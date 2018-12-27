import pandas as pd
import numpy as np
from util import create_count_features, replace_feature_values
from constants import TARGET, IDS


def preprocess(df):

    # Divide features into 4 groups: DOES_EXIST_FEATURES, COUNT_FEATURES, TO_DUMMIES_FEATURES, NUMERIC

    BOOLEAN_FEATURES = [col for col in df.drop([TARGET], axis=1).columns
                        if set(df[col].unique()) == set([1, 0])]
    COUNT_FEATURES = [col for col in df.select_dtypes(include=[object]).columns
                      if len(df[col].unique()) > 2]
    TO_DUMMIES_FEATURES = [col for col in df.select_dtypes(include=[object]).columns
                           if len(df[col].unique()) == 2]
    NUMERIC = df.drop(BOOLEAN_FEATURES + IDS + [TARGET], axis=1). \
        select_dtypes(include=[np.int, np.float]).columns.tolist()

    # replace all non popular categories with "Other"
    for feature_name in COUNT_FEATURES + TO_DUMMIES_FEATURES:
        df = replace_feature_values(df, feature_name)

    # create count features
    for feature_name in COUNT_FEATURES:
        df = create_count_features(df, feature_name)

    # create dummies features
    for feature_name in TO_DUMMIES_FEATURES:
        df = pd.concat(
            [df,
             pd.get_dummies(df[feature_name], prefix=feature_name)], axis=1)

    # replace NA with -999
    df.fillna(-999, inplace=True)

    # drop unnecessary columns
    df = df.select_dtypes(include=['float64', 'int64'])

    # drop highly correlated features
    predictors = sorted([x for x in df.columns if x != TARGET], reverse=True)
    corr_matrix = df[predictors].corr().abs()
    correlated_features = []
    for feature_name in predictors:
        if (corr_matrix.loc[feature_name] > 0.95).sum() > 1:
            correlated_features.append(feature_name)
            corr_matrix.drop([feature_name], axis=1, inplace=True)
    df.drop(correlated_features, axis=1, inplace=True)

    # drop near zero features
    freq = (df == 0).sum() / len(df)
    nero_zero_features = [x for x in freq[freq > 0.985].index.tolist() if x != TARGET]
    df.drop(nero_zero_features, axis=1, inplace=True)

    return df
