from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel
from .utils import str_to_tuple_list, make_clean_lemmas

"""
df: A pandas DataFrame with [userid, text] columns
tkz: tokenizer
ltz: lemmatizer
stopwords: list of stopwords.
fn_stub: the string that'll go before [lemma, dict, bow].csv. Assumed to already have its id and stuff.
"""
def preprocess_tweets(df, tkz, ltz, stopwords, out_dir, fn_stub, verbose=False, sent=None):
    clean_urls(df)
    if verbose: print("urls cleaned")

    def calc_sent(text):
        if text is not None:
            return sent.polarity_scores(text)['compound']
        else:
            return 0

    if sent is not None:
        if verbose: print("adding sentiment features")
        df['sentiment'] = df['text'].apply(calc_sent)

    df['text'] = df['text'].apply(make_preprocess(tkz, ltz, stopwords))
    fn_lemma = "{}/{}_{}.csv".format(out_dir, fn_stub, "lemmas")
    if verbose: print("saving lemmas to {}".format(fn_lemma))
    df.to_csv(fn_lemma)

    if verbose: print("building dict.")
    text_dict = Dictionary(df.text)
    fn_dict = "{}/{}_{}.csv".format(out_dir, fn_stub, "dict")
    if verbose: print("saving dict/word indexes to {}".format(fn_dict))
    with open(fn_dict, 'w') as f:
        for k in text_dict.token2id:
            v = text_dict.token2id[k]
            f.write("{}, {}\n".format(k, v))

    if verbose: print("building bow features")
    df['bow_features'] = df['text'].apply(lambda t: text_dict.doc2bow(t))
    fn_bow = "{}/{}_{}.csv".format(out_dir, fn_stub, "bow")
    if verbose: print("saving bow features to {}".format(fn_bow))

    df_save = df.drop(columns=["text", "sentiment"]).copy()
    df_save.to_csv(fn_bow)

    return df, text_dict


def preprocess_tweets_w_alex(df, tkz, ltz, stopwords, alex, verbose=False, sent=None):
    clean_urls(df)
    if verbose: print("urls cleaned")

    def calc_sent(text):
        if text is not None:
            return sent.polarity_scores(text)['compound']
        else:
            return 0

    if sent is not None:
        if verbose: print("adding sentiment features")
        df['sentiment'] = df['text'].apply(calc_sent)

    df['lemmas'] = df['text'].apply(make_preprocess(tkz, ltz, stopwords))

    # if verbose: print("building dict.")
    # text_dict = Dictionary(df.text)
    # if verbose: print("building bow features")
    # df['bow_features'] = df['lemmas'].apply(lambda t: text_dict.doc2bow(t))

    # calculate anxiety score
    def anxiety_score(lemmas):
        s = 0
        for lemma in lemmas:
            if lemma in alex:
                s += alex[lemma]
        return s

    if verbose: print("adding anxiety score")
    df['anxiety'] = df['lemmas'].apply(anxiety_score)
    return df


def clean_urls(dframe):
    dframe['text'] = dframe['text'].str.replace(r"http\S+", "", regex=True)


def make_preprocess(tkz, ltz, stopwords):
    def preprocess(text):
        if text is not None:
            tokens = tkz.tokenize(text)
            tokens = [t.lower() for t in tokens if t not in stopwords]
            lemmas = [ltz.lemmatize(t) for t in tokens]
            return lemmas
        return []
    return preprocess


def generate_topic_terms(df_bow, text_dict, fn_out, n_topics=50, r_state=1, n_passes=1, verbose=False):
    df_bow.drop(df_bow[df_bow['bow_features'] == "[]"].index, inplace=True) # Drop 'empty' tweets
    df_bow['bow_features'] = df_bow['bow_features'].apply(str_to_tuplelist)

    if verbose: print("bow feature df loaded. performing LDA")
    tweets_lda = LdaModel(df_bow['bow_features'].to_list(),
                          num_topics=n_topics, # This doesn't seem to be working?
                          id2word=text_dict,
                          random_state=r_state,
                          # alpha="auto",
                          passes=n_passes)

    if verbose: print("saving topics to {}".format(fn_out))
    with open("{}".format(fn_out), 'w') as f:
        for topic in tweets_lda.show_topics(num_topics=n_topics, formatted=True):
            f.write("{}\n".format(topic))

    topic_terms = set()
    for topic in tweets_lda.show_topics(num_topics=n_topics, formatted=False):
        terms = topic[1]
        for term in terms:
            topic_terms.add(term[0])

    return topic_terms


def filter_lemmas(df, good_terms):
    df['lemmas'] = df['text'].apply(make_clean_lemmas(good_terms))
    text_dict = Dictionary(df['lemmas'])
    df['bow'] = df['lemmas'].apply(lambda l: text_dict.doc2bow(l))
    return df, text_dict


def add_anxiety_scores(df, anxiety_dict):
    def sum_anxiety(lemmas):
        score = 0.0
        for l in lemmas:
            if l in anxiety_dict:
                score += anxiety_dict[l]
        return score
    df['anxiety'] = df['lemmas'].apply(sum_anxiety)
