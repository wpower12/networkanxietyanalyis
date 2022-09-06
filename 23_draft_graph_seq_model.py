import torch
from torch.nn import Module, RNN, Linear, BCELoss
import torch.nn.functional as F
from torch.optim import SGD
from torch_geometric.nn import GCNConv
from naatools import pipeline, utils
from tqdm import tqdm

USER_SEQ_DIR = "data/prepared/mb_user_sequences_new_targets"
WINDOW_SIZE = 5
ANX_THRESHOLD = 0.05
NUM_EPOCHS = 10
TEST_FRAC = 0.35

class GraphSequenceModel(Module):
    def __init__(self):
        super(GraphSequenceModel, self).__init__()
        self.gcn = GCNConv(2, 2)
        self.rnn = RNN(2, 3)
        self.linear = Linear(3, 1)

    def forward(self, graphs):
        seq = [self.gcn(g.x, g.edge_index) for g in graphs]
        _, h = self.rnn(torch.cat(seq))
        h = F.relu(h)
        h = self.linear(h)
        return torch.sigmoid(h)


model = GraphSequenceModel()
criteria = BCELoss()
optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)

graph_sequences, labels = pipeline.read_user_mn_examples_from_dir(USER_SEQ_DIR,
                                                                  WINDOW_SIZE,
                                                                  ANX_THRESHOLD,
                                                                  target_col="max_raw_anx")
data_train, data_test = pipeline.balance_and_split_data(graph_sequences, labels, test_frac=TEST_FRAC)


def eval_batch(loss_func, graph_model, data):
    loss_acc = 0
    for (graph_seq, graph_y) in data:
        loss_acc += loss_func(graph_model(graph_seq), graph_y).item()
    return loss_acc/len(data)


print("training")
for epoch in range(NUM_EPOCHS):
    exs_pb = tqdm(data_train, delay=0.25, unit="examples")
    running_loss = 0
    n = 0
    for (gs, y) in exs_pb:
        optimizer.zero_grad()
        loss = criteria(model(gs), y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        n += 1
        exs_pb.desc = "epoch: {} rloss: {:.4}".format(epoch+1, running_loss/(n))

    print("epoch: {} test loss: {:.4}".format(epoch+1, eval_batch(criteria, model, data_test)))
