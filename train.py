import pickle

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# parameters

C = 1.0
n_splits = 5
output_file = f'model_C={C}.bin'

# data preparation

df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

df.columns = df.columns.str.lower().str.replace(' ', '_')

categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)

for c in categorical_columns:
    df[c] = df[c].str.lower().str.replace(' ', '_')

df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce')
df.totalcharges = df.totalcharges.fillna(0)

df.churn = (df.churn == 'yes').astype(int)

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)

numerical = ['tenure', 'monthlycharges', 'totalcharges']

categorical = [
    'gender',
    'seniorcitizen',
    'partner',
    'dependents',
    'phoneservice',
    'multiplelines',
    'internetservice',
    'onlinesecurity',
    'onlinebackup',
    'deviceprotection',
    'techsupport',
    'streamingtv',
    'streamingmovies',
    'contract',
    'paperlessbilling',
    'paymentmethod',
]


# training and returning dv and model

def train(df_train, y_train, C=1.0):
    dicts = df_train[categorical + numerical].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    model = LogisticRegression(C=C, max_iter=1000)
    model.fit(X_train, y_train)

    return dv, model

# makes predictions
def predict(df, dv, model):
    dicts = df[categorical + numerical].to_dict(orient='records')

    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred


# validation

print(f'doing validation with C={C}')

#iteration over folds and training
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)


scores = []    # stores auc for each each fold

fold = 0

for train_idx, val_idx in kfold.split(df_full_train):
    
#selects the training data for the current fold by indexing the df_full_train
    df_train = df_full_train.iloc[train_idx]
    
#selects the validation data for the current fold using the validation
    df_val = df_full_train.iloc[val_idx]

#extracts target variable for training and validation
    y_train = df_train.churn.values
    y_val = df_val.churn.values


#to train logistic model
    dv, model = train(df_train, y_train, C=C)
 
 #make pred on val data
    y_pred = predict(df_val, dv, model)

#evaluate models performance on val data
    auc = roc_auc_score(y_val, y_pred)
    scores.append(auc)

#accuracy of current fold  and incremented
    print(f'auc on fold {fold} is {auc}')
    fold = fold + 1

print('validation results:')
print('C=%s %.3f +- %.3f' % (C, np.mean(scores), np.std(scores)))

# training the final model

print('training the final model')

#call train function
dv, model = train(df_full_train, df_full_train.churn.values, C=1.0)
#call predict function
y_pred = predict(df_test, dv, model)


#true churn labels from the test dataset are extracted
y_test = df_test.churn.values
auc = roc_auc_score(y_test, y_pred)

print(f'auc={auc}')

# Save the model

with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)

print(f'the model is saved to {output_file}')