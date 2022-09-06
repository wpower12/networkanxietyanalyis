import torch
import torch_geometric.nn as pygnn
from torch.nn import Module, RNN, Linear, BCELoss
from torch.nn.functional import relu
from torch.optim import SGD
from naatools import pipeline, processing, utils

import pickle

CACHED = False

RANDOM_SEED = 0
utils.set_random_seeds(RANDOM_SEED)

DIR_CENTRAL_USERS   = "data/prepared/mb_00_central_users"
DIR_MENTIONED_USERS = "data/prepared/mb_00_mentioned_users"

FN_PF_GRAPHS = "data/prepared/mb_02_graphs.p"
FN_PF_LABELS = "data/prepared/mb_02_labels.p"
FN_PF_BALANCED_GRAPH_DATA = "data/prepared/mb_02_balanced_split.p"
FN_PF_SUS_DATA = "data/prepared/mb_02_balanced_sus.p"

RESULTS_DIR = "data/results/sus_vs_mn_03"
WINDOW_SIZE = 5
ANX_THRESHOLD = 0.0
NUM_EPOCHS = 4
TEST_FRAC = 0.15
TEST_INTERVAL = 4000
USER_LIMIT = 5000
EXP_LIMIT = None

if not CACHED:
    graph_sequences, labels = pipeline.generate_exs_from_cu_mu_dirs_cached(DIR_CENTRAL_USERS,
                                                                           DIR_MENTIONED_USERS,
                                                                           WINDOW_SIZE,
                                                                           ANX_THRESHOLD,
                                                                           limit=USER_LIMIT,
                                                                           target_col="max_raw_anx")
    pickle.dump(graph_sequences, open(FN_PF_GRAPHS, 'wb'))
    pickle.dump(labels, open(FN_PF_LABELS, 'wb'))

class GraphSequenceModel(Module):
    def __init__(self):
        super(GraphSequenceModel, self).__init__()
        self.gcn = pygnn.GCNConv(2, 2)
        self.rnn = RNN(2, 3)
        self.linear = Linear(3, 1)

    def forward(self, graphs):
        seq = [self.gcn(g.x.float(), g.edge_index) for g in graphs]
        _, h = self.rnn(torch.cat(seq))
        h = relu(h)
        h = self.linear(h)
        return torch.sigmoid(h)


print("Graph Sequence Model")
if not CACHED:
    gsm_train, gsm_test = pipeline.balance_and_split_data(graph_sequences, labels, test_frac=TEST_FRAC, limit=EXP_LIMIT)
    pickle.dump([gsm_train, gsm_test], open(FN_PF_BALANCED_GRAPH_DATA, 'wb'))
else:
    gsm_train, gsm_test = pickle.load(open(FN_PF_BALANCED_GRAPH_DATA, 'rb'))

gsm_model = GraphSequenceModel()
gsm_criteria = BCELoss()
gsm_optimizer = SGD(gsm_model.parameters(), lr=0.001, momentum=0.9)

with open("{}/gsm_log.csv".format(RESULTS_DIR), 'w') as gsm_log_f:
    pipeline.train_batched_graph_model(gsm_model,
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
        _, h = self.rnn(x.float())
        h = relu(h)
        h = self.linear(h)
        return torch.sigmoid(h)


print("Baseline Single User Sequence Model")
if not CACHED:
    user_sequences = processing.extract_cuss_from_gss(graph_sequences)
    susm_train, susm_test = pipeline.balance_and_split_data(user_sequences, labels, test_frac=TEST_FRAC, limit=EXP_LIMIT)
    pickle.dump([susm_train, susm_test], open(FN_PF_SUS_DATA, 'wb'))
else:
    susm_train, susm_test = pickle.load(open(FN_PF_SUS_DATA, 'rb'))

susm_model = SingleUserSequenceModel()
susm_criteria = BCELoss()
susm_optimizer = SGD(susm_model.parameters(), lr=0.001, momentum=0.9)
with open("{}/susm_log.csv".format(RESULTS_DIR), 'w') as susm_log_f:
    pipeline.train_batched_graph_model(susm_model,
                                       susm_criteria,
                                       susm_optimizer,
                                       susm_train,
                                       susm_test,
                                       num_epochs=NUM_EPOCHS,
                                       log_file=susm_log_f,
                                       test_interval=TEST_INTERVAL)
