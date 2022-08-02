"""
cus = central user sequence, the sequence of node features for the central (0th) user in the graph
      The convention that the 0th user is the central user is maintained in the data pipeline code.
gs  = graph sequence, the original/input sequence of graphs.
"""
from twitter_utils import pipeline, processing

USER_SEQ_DIR = "data/prepared/mb_user_sequences_new_targets"
ANX_THRESHOLD = 0.0
WINDOW_SIZE = 5

graph_sequences, labels = pipeline.read_user_mn_examples_from_dir(USER_SEQ_DIR,
                                                                  WINDOW_SIZE,
                                                                  ANX_THRESHOLD,
                                                                  target_col="max_raw_anx",
                                                                  limit=200)

user_sequences = processing.extract_cuss_from_gss(graph_sequences, WINDOW_SIZE, 2)

print(user_sequences[0])
