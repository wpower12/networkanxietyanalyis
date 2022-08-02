from os import listdir
from os.path import isfile, join
import torch
import random
import pandas as pd
from tqdm import tqdm
from .processing import create_examples_from_raw_mn_sequences


def read_user_mn_examples_from_dir(user_seq_dir, window_size, anx_threshold, limit=None, target_col="ave_pos_anx",
                                   verbose=True):
    user_files = [f for f in listdir(user_seq_dir) if isfile(join(user_seq_dir, f))]

    if limit is not None:
        user_files = user_files[:limit]

    if verbose: print("reading raw user sequences.")

    # Need to build the user_id -> df map.
    user_id_2_df = {}
    for f in tqdm(user_files, delay=0.1, unit="user", disable=(not verbose)):
        user_id = int(f.replace("sequences_", "").replace(".csv", ""))
        user_df = pd.read_csv("{}/{}".format(user_seq_dir, f),
                              parse_dates=[0],
                              infer_datetime_format=True,
                              dtype={"mentioned_users": str})
        user_df['mentioned_users'].fillna("", inplace=True)
        user_id_2_df[user_id] = user_df

    if verbose: print("building example graph sequences")
    example_graph_seqs = []
    example_labels = []
    for _, user_id in enumerate(tqdm(user_id_2_df, delay=0.1, unit="user", disable=(not verbose))):
        df_central_user = user_id_2_df[user_id]
        graphs, labels = create_examples_from_raw_mn_sequences(df_central_user,
                                                               user_id,
                                                               user_id_2_df,
                                                               window_size,
                                                               anx_threshold,
                                                               target_col=target_col)
        example_graph_seqs.extend(graphs)
        example_labels.extend(labels)
    return example_graph_seqs, example_labels


def balance_and_split_data(sequences, labels, test_frac=0.1):
    print("balancing dataset and splitting test/train groups.")
    idx_pos = [idx for idx, l in enumerate(labels) if l.sum() > 0]
    idx_neg = [idx for idx, l in enumerate(labels) if l.sum() == 0]
    print("{}/{} raw positive example sequences".format(len(idx_pos), len(labels)))

    # First we split the raw pos and negative examples into properly sized test/train groups
    cutoff_pos = int((1-test_frac)*len(idx_pos))
    cutoff_neg = int((1-test_frac)*len(idx_neg))
    train_idx_pos = idx_pos[:cutoff_pos]
    train_idx_neg = idx_neg[:cutoff_neg]
    test_idx_pos = idx_pos[cutoff_pos:]
    test_idx_neg = idx_neg[cutoff_neg:]

    num_copies_pos_train = int(len(train_idx_pos)/len(train_idx_neg))
    num_copies_pos_test  = int(len(train_idx_pos)/len(train_idx_neg))

    # Then we actually build the test/train data sets by creating multiple copies of the positive examples.
    train_seqs   = []
    train_labels = []
    # Add copies of positive examples
    for i in range(num_copies_pos_train):
        for p_idx in train_idx_pos:
            train_seqs.append(sequences[p_idx])
            train_labels.append(labels[p_idx])
    # Add negative examples
    for n_idx in train_idx_neg:
        train_seqs.append(sequences[n_idx])
        train_labels.append(labels[n_idx])
    train_data = list(zip(train_seqs, train_labels))
    random.shuffle(train_data)

    test_seqs   = []
    test_labels = []
    # Add copies of positive examples
    for i in range(num_copies_pos_test):
        for p_idx in test_idx_pos:
            test_seqs.append(sequences[p_idx])
            test_labels.append(labels[p_idx])
    # Add negative examples
    for n_idx in test_idx_neg:
        test_seqs.append(sequences[n_idx])
        test_labels.append(labels[n_idx])
    test_data = list(zip(test_seqs, test_labels))
    random.shuffle(test_data)

    return train_data, test_data


def eval_batch(model, loss_func, data_batch):
    loss_acc = 0
    for (seq, y) in data_batch:
        loss_acc += loss_func(model(seq), y).item()
    return loss_acc/len(data_batch)


'''
'Manually' batches sequences. I think this is the only way to batch the graph sequences? Idk. Seems to be working. 
'''
def train_batched_model(model, criteria, optimizer, data_train, data_test, log_file=None, batch_size=16,
                        num_epochs=1, verbose=True, test_interval=100):

    for epoch in range(num_epochs):
        exs_pb = tqdm(data_train, delay=0.25, unit="examples", disable=(not verbose))
        running_loss = 0
        b = 0
        batch = []
        for (gs, y) in exs_pb:
            batch.append((gs, y))

            if len(batch) < batch_size:
                continue
            else:
                optimizer.zero_grad()

                # Create batches of examples and targets such that the operations can still be back proped/autograd-ed?
                outputs = [model(ex) for (ex, _) in batch]
                outputs = torch.stack(outputs)
                targets = [target for (_, target) in batch]
                targets = torch.stack(targets)

                loss = criteria(outputs, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                b += 1
                if verbose:
                    exs_pb.desc = "epoch: {} ave training loss: {:.4}".format(epoch + 1, running_loss / b)
                batch = []

                if log_file is not None:
                    log_file.write("batch_training_loss, {}, {}\n".format(b, running_loss / b))
                    if b % test_interval == 0:
                        log_file.write("batch_test_loss, {}, {}\n".format(b, eval_batch(model, criteria, data_test)))

        epoch_test_loss = eval_batch(model, criteria, data_test)
        if verbose: print("epoch: {} ave test loss: {:.4}".format(epoch + 1, epoch_test_loss))
        if log_file is not None:
            log_file.write("epoch_test_loss, {}, {}\n".format(epoch+1, epoch_test_loss))
