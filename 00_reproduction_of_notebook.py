import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

FN_TWEETS_AFTER = "data/srikar/anxiety_per_tweet_AFTER_2022_01_24.csv"
FN_SCI_DATA = "data/srikar/counties_countries_sci.tsv"

anxiety_tweets = pd.read_csv(FN_TWEETS_AFTER, dtype={'countyfips': str})

sci_data = pd.read_csv(FN_SCI_DATA, sep='\t', dtype={'user_loc': str})
sci_ua = sci_data.query("fr_loc == 'UA'").drop('fr_loc', axis=1)  # Isolate Ukraine data.

county_fips = pd.read_csv('data/srikar/county_fips.csv', dtype={'fips': str})
county_fips = county_fips.query("state == state")
county_fips = county_fips.merge(sci_ua, how='inner', left_on='fips', right_on='user_loc')

anxiety_fips = anxiety_tweets.merge(county_fips, how='left', left_on ='countyfips', right_on='fips')
anxiety_fips['date'] = pd.to_datetime(anxiety_fips.date.values).round('D')
anxiety_fips.anxiety = anxiety_fips.anxiety.apply(lambda x: (x + 1)/2).values

full_datetime_df = pd.DataFrame({'date' : pd.date_range(anxiety_fips.date.min(), anxiety_fips.date.max())})


# function to interpolate anxiety dataframe
def interpolate_df(df, gb_feature, sort_feature='date', interpolate_feature=['anxiety'],
                   full_sort_feature_df=full_datetime_df):
    df_gb = df[gb_feature].values[0]
    df = df.merge(full_sort_feature_df, left_on=sort_feature, right_on=sort_feature, how='outer')

    df = df.sort_values(sort_feature)
    for feature in interpolate_feature:
        n_nas = df[feature].isna().sum()
        if n_nas > df.shape[0] / 1.5:
            # print(str(df_gb) + ' has more than half missing values; exclude')
            return None
        df[feature] = df[feature].interpolate().values
    df[gb_feature] = df_gb
    return df


# lag a feature by `periods` amount
def get_lags(df, lag_feature='anxiety', periods=1, sort_feature='date'):
    df = df.sort_values(sort_feature)
    df[lag_feature + '_lag' + str(periods)] = df[lag_feature].shift(periods).values
    return df


# lag a feature for a longitudinal dataframe
def groupby_get_lags(df, lag_feature='anxiety', periods=1, sort_feature='date', gb_feature='state'):
    df = pd.concat([get_lags(gb_df[1], lag_feature=lag_feature, sort_feature=sort_feature, periods=periods) for gb_df in
                    df.groupby(gb_feature)])
    return df


# create function to interpolate missing values in hashtag features and rescale data to be between 0 and 1
def interpolate_ht_df(df, gb_feature, sort_feature='date', interpolate_feature=['anxiety'],
                      full_sort_feature_df=full_datetime_df):
    df_gb = str(df[gb_feature].values[0])
    df = df.merge(full_sort_feature_df, left_on=sort_feature, right_on=sort_feature, how='outer')

    df = df.sort_values(sort_feature)
    for feature in interpolate_feature:
        n_nas = df[feature].isna().sum()
        if n_nas > df.shape[0] / 2:
            print(df_gb + ' has more than half missing values ' + feature + '; exclude')
        df[feature] = df[feature].interpolate().values
    df[gb_feature] = df_gb
    return df


# interpolate missing anxiety data and remove counties with more than 1/2 values missing
anxiety_cty = anxiety_fips.groupby(['date', 'countyfips'])[['anxiety']].mean().reset_index().sort_values(['countyfips', 'date'])
anxiety_cty = pd.concat([interpolate_df(df = gb_df[1], gb_feature = 'countyfips') for gb_df in anxiety_cty.groupby('countyfips')])

anxiety_lags = anxiety_cty.copy()
for i in range(1,6):
    # anxiety_lags['anxiety_lag' + str(i)] = anxiety_lags.groupby('state').anxiety.shift(i).values
    anxiety_lags = groupby_get_lags(df=anxiety_lags, periods=i, gb_feature='countyfips')


anxiety_lags.to_csv('anxiety_lags.csv', index = False)

# load in state-level hashtag features
ht_features = pd.read_csv('data/srikar/ht_features_state.csv')
ht_features['date'] = pd.to_datetime(ht_features['date'])
ht_features['date_num'] = ht_features['date'].astype(int)
ht_features = pd.concat([interpolate_ht_df(df = gb_df[1],
               gb_feature = 'state',
               sort_feature = 'date',
               interpolate_feature = ['business', 'entertainment', 'food', 'outdoors', 'sports'],
               full_sort_feature_df = full_datetime_df) for gb_df in ht_features.groupby('state')])

# add social connectedness and state-level hashtags to county-level anxiety dataset
anxiety_sci = anxiety_cty.merge(county_fips, left_on = 'countyfips', right_on = 'fips', how = 'inner')
anxiety_ht = anxiety_sci.merge(ht_features, left_on = ['date', 'state'], right_on = ['date', 'state'], how = 'left')
anxiety_ht['treated'] = anxiety_ht.date.values > pd.to_datetime('2022-02-24')
anxiety_ht['anxiety_log'] = anxiety_ht.anxiety.apply(lambda x: np.log(x))
# add 1 time period lag for anxiety
for i in [1]:
    anxiety_ht = groupby_get_lags(df = anxiety_ht, periods = i, gb_feature = 'countyfips')


# identify which lagged periods are related to anxiety
lag_model = smf.logit('anxiety ~ anxiety_lag1 + anxiety_lag2 + anxiety_lag3 + anxiety_lag4', anxiety_lags.dropna()).fit()
lag_model.summary()
anxiety_ht.to_csv('anxiety_ht_county.csv', index = False)
