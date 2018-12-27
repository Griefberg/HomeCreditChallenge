import pandas as pd
from preprocessing import preprocess
from modeling import train_model

train = pd.read_csv('data/application_train.csv')

preprocessed_train = preprocess(train)

alg, model_performance, ft_importance = train_model(preprocessed_train)
