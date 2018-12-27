import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import GridSearchCV
from constants import TARGET, CV, METRIC, RANDOM_SEED, N_JOBS, EARLY_STOP
from util import modelfit


def train_model(preprocessed_train):

    # get train and test set
    PREDICTORS = [x for x in preprocessed_train.columns if x not in ['SK_ID_CURR', TARGET]]
    x_train, x_test, y_train, y_test = train_test_split(preprocessed_train[PREDICTORS],
                                                        preprocessed_train[TARGET], test_size=0.1,
                                                        random_state=RANDOM_SEED,
                                                        stratify=preprocessed_train[TARGET])
    weight = (preprocessed_train[TARGET] == 0).sum() / (preprocessed_train[TARGET] == 1).sum()

    # MODELING

    # Step 1. Fix learning rate and choose n_estimators
    xgb1 = XGBClassifier(
        learning_rate=0.1,
        n_estimators=1000,
        max_depth=5,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        feval=METRIC,
        n_jobs=N_JOBS,
        scale_pos_weight=1,
        random_state=RANDOM_SEED)
    best_n = modelfit(xgb1, x_train, y_train, x_test, y_test, cv_folds=CV,
                      early_stopping_rounds=EARLY_STOP)

    # Step 2. Tune max_depth and min_child_weight
    param_test1 = {
        'max_depth': range(3, 8, 1),
        'min_child_weight': range(1, 7, 1)
    }
    gsearch1 = GridSearchCV(
        estimator=XGBClassifier(learning_rate=0.1, n_estimators=best_n, max_depth=5,
                                min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                objective='binary:logistic', n_jobs=N_JOBS, scale_pos_weight=1,
                                random_state=RANDOM_SEED),
        param_grid=param_test1, scoring='roc_auc', n_jobs=N_JOBS, iid=False, cv=CV)
    gsearch1.fit(x_train, y_train)
    print(gsearch1.best_params_, gsearch1.best_score_)

    # Step 3 Tune weights
    param_test2 = {'scale_pos_weight': [1, weight]}
    gsearch2 = GridSearchCV(
        estimator=XGBClassifier(learning_rate=0.1, n_estimators=best_n,
                                max_depth=gsearch1.best_params_['max_depth'],
                                min_child_weight=gsearch1.best_params_['min_child_weight'],
                                gamma=0, subsample=0.8, colsample_bytree=0.8,
                                objective='binary:logistic', n_jobs=N_JOBS, scale_pos_weight=1,
                                random_state=RANDOM_SEED),
        param_grid=param_test2, scoring='roc_auc', n_jobs=N_JOBS, iid=False, cv=CV)
    gsearch2.fit(x_train, y_train)
    print(gsearch2.best_params_, gsearch2.best_score_)

    # Step 4: Tune gamma
    param_test3 = {
        'gamma': [0, 0.5, 1, 2, 3, 5, 7]
    }
    gsearch3 = GridSearchCV(
        estimator=XGBClassifier(learning_rate=0.1, n_estimators=best_n,
                                max_depth=gsearch1.best_params_['max_depth'],
                                min_child_weight=gsearch1.best_params_['min_child_weight'],
                                gamma=0, subsample=0.8, colsample_bytree=0.8,
                                objective='binary:logistic', n_jobs=N_JOBS,
                                scale_pos_weight=gsearch2.best_params_['scale_pos_weight'],
                                random_state=RANDOM_SEED),
        param_grid=param_test3, scoring='roc_auc', n_jobs=N_JOBS, iid=False, cv=CV)
    gsearch3.fit(x_train, y_train)
    print(gsearch3.best_params_, gsearch3.best_score_)

    # recalculate
    xgb2 = XGBClassifier(
        learning_rate=0.1,
        n_estimators=1000,
        max_depth=gsearch1.best_params_['max_depth'],
        min_child_weight=gsearch1.best_params_['min_child_weight'],
        gamma=gsearch3.best_params_['gamma'],
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        feval=METRIC,
        n_jobs=N_JOBS,
        scale_pos_weight=gsearch2.best_params_['scale_pos_weight'],
        random_state=RANDOM_SEED)
    best_n = modelfit(xgb2, x_train, y_train, x_test, y_test,
                      cv_folds=CV, early_stopping_rounds=EARLY_STOP)

    # Step 5: Tune subsample and colsample_bytree
    param_test4 = {
        'subsample': [i / 10.0 for i in range(6, 11)],
        'colsample_bytree': [i / 10.0 for i in range(6, 11)]
    }
    gsearch4 = GridSearchCV(
        estimator=XGBClassifier(learning_rate=0.1, n_estimators=best_n,
                                max_depth=gsearch1.best_params_['max_depth'],
                                min_child_weight=gsearch1.best_params_['min_child_weight'],
                                gamma=gsearch3.best_params_['gamma'],
                                subsample=0.8, colsample_bytree=0.8,
                                objective='binary:logistic', n_jobs=N_JOBS,
                                scale_pos_weight=gsearch2.best_params_['scale_pos_weight'],
                                random_state=RANDOM_SEED),
        param_grid=param_test4, scoring='roc_auc', n_jobs=N_JOBS, iid=False, cv=CV)
    gsearch4.fit(x_train, y_train)
    print(gsearch4.best_params_, gsearch4.best_score_)

    # Step 6: Tuning Regularization Parameters
    param_test5 = {
        'reg_alpha': [1e-5, 1e-2, 0.1, 1, 100],
        'reg_lambda': [1e-5, 1e-2, 0.1, 1, 100]
    }
    gsearch5 = GridSearchCV(
        estimator=XGBClassifier(learning_rate=0.1, n_estimators=best_n,
                                max_depth=gsearch1.best_params_['max_depth'],
                                min_child_weight=gsearch1.best_params_['min_child_weight'],
                                gamma=gsearch3.best_params_['gamma'],
                                subsample=gsearch4.best_params_['subsample'],
                                colsample_bytree=gsearch4.best_params_['colsample_bytree'],
                                objective='binary:logistic', n_jobs=N_JOBS,
                                scale_pos_weight=gsearch2.best_params_['scale_pos_weight'],
                                random_state=RANDOM_SEED),
        param_grid=param_test5, scoring='roc_auc', n_jobs=N_JOBS, iid=False, cv=CV)
    gsearch5.fit(x_train, y_train)
    print(gsearch5.best_params_, gsearch5.best_score_)

    # Step 6: Reducing Learning Rate
    xgb3 = XGBClassifier(
        learning_rate=0.01,
        n_estimators=5000,
        max_depth=gsearch1.best_params_['max_depth'],
        min_child_weight=gsearch1.best_params_['min_child_weight'],
        gamma=gsearch3.best_params_['gamma'],
        subsample=gsearch4.best_params_['subsample'],
        colsample_bytree=gsearch4.best_params_['colsample_bytree'],
        reg_alpha=gsearch5.best_params_['reg_alpha'],
        reg_lambda=gsearch5.best_params_['reg_lambda'],
        objective='binary:logistic',
        feval=METRIC,
        n_jobs=N_JOBS,
        scale_pos_weight=gsearch2.best_params_['scale_pos_weight'],
        random_state=RANDOM_SEED)

    alg, model_performance, ft_importance = modelfit(xgb3, x_train, y_train, x_test, y_test, cv_folds=CV,
                                                     early_stopping_rounds=EARLY_STOP, return_fitted=True)

    return alg, model_performance, ft_importance
