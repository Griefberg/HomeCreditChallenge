import datetime
import pickle
import pandas as pd
import xgboost as xgb
from sklearn import metrics


def replace_feature_values(df, feature_name):
    freq = df[feature_name].fillna(-999).value_counts(normalize=True)
    others = freq[freq < 0.01].index
    df.loc[df[feature_name].isin(others), feature_name] = 'Other'
    return df


def create_count_features(df, feature_name):
    freq = df.groupby(feature_name)[feature_name].count().to_dict()
    df[feature_name + '_count'] = df[feature_name].replace(freq)
    return df


def top_features_and_performance_metrics(clf, xtrain, ytrain, xtest, ytest, cv_performance):
    dtrain_predprob = clf.predict_proba(xtrain)
    dtest_predprob = clf.predict_proba(xtest)

    auc_train = metrics.roc_auc_score(ytrain.values, dtrain_predprob[:, 1])
    auc_test = metrics.roc_auc_score(ytest.values, dtest_predprob[:, 1])

    print("AUC Score (CV): %f" % cv_performance['test-auc-mean'])
    print("AUC Score STD (CV): %f" % cv_performance['test-auc-std'])
    print("AUC Score (Train): %f" % auc_train)
    print("AUC Score (Test): %f" % auc_test)

    ft_importance = pd.DataFrame.from_dict(clf.get_booster().get_score(importance_type='gain'),
                                           orient='index', columns=['importance'])
    ft_importance.sort_values(by='importance', ascending=False, inplace=True)
    ft_importance = ft_importance.iloc[:15]

    performance = pd.DataFrame(
        {
            'cv_auc_mean': cv_performance['test-auc-mean'],
            'cv_map_std': cv_performance['test-auc-std'],
            'auc_train': auc_train,
            'auc_test': auc_test,
        })

    return ft_importance, performance


def modelfit(alg, xtrain, ytrain, xtest, ytest, cv_folds=5, early_stopping_rounds=25,
             return_fitted=False):
    xgb_param = alg.get_xgb_params()
    xgtrain = xgb.DMatrix(xtrain.values, label=ytrain.values)
    cv_result = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'],
                       nfold=cv_folds, stratified=True, metrics=alg.get_params()['feval'],
                       early_stopping_rounds=early_stopping_rounds, verbose_eval=True)
    alg.set_params(n_estimators=cv_result.shape[0])

    # Fit the algorithm on the data
    alg.fit(xtrain, ytrain, eval_metric=alg.get_params()['feval'])
    cv_performance = cv_result[-1:]
    ft_importance, model_performance = top_features_and_performance_metrics(alg, xtrain, ytrain,
                                                                            xtest, ytest,
                                                                            cv_performance)

    if return_fitted:
        return alg, model_performance, ft_importance
    else:
        return cv_result.shape[0]
