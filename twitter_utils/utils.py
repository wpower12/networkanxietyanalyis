
def str_to_tuple_list(s):
    if s == "[]":
        return []
    tuples = s.replace("[", "").replace("]", "").replace("), (", "):(").split(":")
    tuples = [s.replace("(", "").replace(")", "") for s in tuples]
    tuples = [s.split(", ") for s in tuples]
    tuples = [(int(s[0]), int(s[1])) for s in tuples]
    return tuples


def str_to_lemma_list(s):
    if s == "[]":
        return []
    return s.replace("[", "").replace("]", "").replace('\'', "").split(",")


def str_to_dict(s):
    if s == "{}": return {}
    entries = s.replace("{", "").replace("}", "").split(", ")
    ret_dict = {}
    for entry in entries:
        bowid, weight = entry.split(":")
        ret_dict[bowid] = int(weight)
    return ret_dict


def read_text_dict(fn):
    text_dict = {}
    with open(fn, 'r') as f:
        for line in f.readlines():
            raw = line.replace("\n", "", ).split(", ")
            if len(raw) == 2:
                k, v = raw
                text_dict[int(v)] = k  # Because we need id -> string
    return text_dict


def make_clean_lemmas(topic_terms):
    def clean_lemmas(text):
        raw_lemmas = text.replace("]", "").replace("[", "").split(", ")
        good_lemmas = []
        for l in raw_lemmas:
            l = l.replace("'", "")
            if l in topic_terms and len(l) > 0:
                good_lemmas.append(l)
        return good_lemmas
    return clean_lemmas


def anx_dict_from_df(df):
    ret_dict = {}
    for _, row in df.iterrows():
        ret_dict[row['lemma']] = row['anxiety']
    return ret_dict

def read_lexicon_to_dict(fn):
    lex_dict = {}
    with open(fn, 'r') as f:
        f.readline()
        for l in f.readlines():
            _, lemma, _, score = l.split(",")
            score = float(score)
            lex_dict[lemma] = score
    return lex_dict

