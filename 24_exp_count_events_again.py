from naatools import pipeline

USER_SEQ_DIR = "data/prepared/mb_user_sequences_new_targets"
WINDOW_SIZE = 5
ANX_THRESHOLDS = [0.0, 0.005, 0.01, 0.05, 0.1]

counts = []
for thresh in ANX_THRESHOLDS:
    print("processing {}".format(thresh))
    _, labels = pipeline.read_user_mn_examples_from_dir(USER_SEQ_DIR,
                                                        WINDOW_SIZE,
                                                        thresh,
                                                        target_col="max_raw_anx")

    print(thresh, sum(labels))
    counts.append((thresh, sum(labels)))

print(counts)
