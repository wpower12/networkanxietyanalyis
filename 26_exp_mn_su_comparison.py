import torch
import torch_geometric.nn as pygnn
from torch.nn import Module, RNN, Linear, BCELoss
from torch.nn.functional import relu
from torch.optim import SGD
from naatools import pipeline, processing, utils

# Trying to stay reproducible.
RANDOM_SEED = 0
utils.set_random_seeds(RANDOM_SEED)


USER_SEQ_DIR = "data/prepared/mb_user_sequences_new_targets"
RESULTS_DIR = "data/results/sus_vs_mn_02"
WINDOW_SIZE = 5
ANX_THRESHOLD = 0.0
NUM_EPOCHS = 4
TEST_FRAC = 0.15
TEST_INTERVAL = 4000

graph_sequences, labels = pipeline.read_user_mn_examples_from_dir(USER_SEQ_DIR,
                                                                  WINDOW_SIZE,
                                                                  ANX_THRESHOLD,
                                                                  target_col="max_raw_anx")
user_sequences = processing.extract_cuss_from_gss(graph_sequences)


class GraphSequenceModel(Module):
    def __init__(self):
        super(GraphSequenceModel, self).__init__()
        self.gcn = pygnn.GCNConv(2, 2)
        self.rnn = RNN(2, 3)
        self.linear = Linear(3, 1)

    def forward(self, graphs):
        seq = [self.gcn(g.x, g.edge_index) for g in graphs]
        _, h = self.rnn(torch.cat(seq))
        h = relu(h)
        h = self.linear(h)
        return torch.sigmoid(h)


print("Graph Sequence Model")
gsm_train, gsm_test = pipeline.balance_and_split_data(graph_sequences, labels, test_frac=TEST_FRAC)
gsm_model = GraphSequenceModel()
gsm_criteria = BCELoss()
gsm_optimizer = SGD(gsm_model.parameters(), lr=0.001, momentum=0.9)

with open("{}/gsm_log.csv".format(RESULTS_DIR), 'w') as gsm_log_f:
    pipeline.train_batched_model(gsm_model,
                                 gsm_criteria,
                                 gsm_optimizer,
                                 gsm_train,
                                 gsm_test,
                                 num_epochs=NUM_EPOCHS,
                                 log_file=gsm_log_f,
                                 test_interval=TEST_INTERVAL)


class SingleUserSequenceModel(Module):
    def __init__(self):
        super(SingleUserSequenceModel, self).__init__()
        self.rnn = RNN(2, 3)
        self.linear = Linear(3, 1)

    def forward(self, x):
        _, h = self.rnn(x)
        h = relu(h)
        h = self.linear(h)
        return torch.sigmoid(h)


print("Baseline Single User Sequence Model")
susm_train, susm_test = pipeline.balance_and_split_data(user_sequences, labels, test_frac=TEST_FRAC)
susm_model = SingleUserSequenceModel()
susm_criteria = BCELoss()
susm_optimizer = SGD(susm_model.parameters(), lr=0.001, momentum=0.9)

with open("{}/susm_log.csv".format(RESULTS_DIR), 'w') as susm_log_f:
    pipeline.train_batched_model(susm_model,
                                 susm_criteria,
                                 susm_optimizer,
                                 susm_train,
                                 susm_test,
                                 num_epochs=NUM_EPOCHS,
                                 log_file=susm_log_f,
                                 test_interval=TEST_INTERVAL)
