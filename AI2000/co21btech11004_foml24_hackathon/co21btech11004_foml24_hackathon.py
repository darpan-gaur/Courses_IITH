#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Team Member
# Darpan Gaur (CO21BTECH11004)
# Bishwashri Roy (CS24RESCH11013)


# In[2]:


import numpy as np
import random

import pandas as pd
import argparse

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier


# In[3]:


# set seed
seed = 42
random.seed(seed)
np.random.seed(seed)


# In[4]:


# ignore warnings
import warnings
warnings.filterwarnings("ignore")


# In[5]:


def preprocessing(train_data, test_data):
    # Count missing values for each column
    missing_values = train_data.isnull().sum()

    # Create a DataFrame to store the count of missing values
    missing_df = pd.DataFrame({
        'Column': missing_values.index,
        'MissingCount': missing_values.values
    })

    # Add a column to show the percentage of missing values
    missing_df['MissingPercentage'] = (missing_df['MissingCount'] / len(train_data)) * 100

    # Sort the DataFrame by the number of missing values in descending order
    missing_df.sort_values(by='MissingCount', ascending=False, inplace=True)

    # Reset index for readability
    missing_df.reset_index(drop=True, inplace=True)

    # Display the DataFrame
    # print(missing_df)

    null_threshold = 10

    # Drop columns with missing percentage greater than 60%
    columns_to_drop = missing_df[missing_df['MissingPercentage'] > null_threshold]['Column'].tolist()

    # Drop the identified columns from the DataFrame
    train_data.drop(columns=columns_to_drop, axis=1, inplace=True)

    # Display the updated DataFrame shape after dropping columns
    # print(f"Updated shape of the DataFrame: {train_data.shape}")

    # drop same columns from test data
    test_data.drop(columns=columns_to_drop, axis=1, inplace=True)

    # Display the updated DataFrame shape after dropping columns
    # print(f"Updated shape of the DataFrame: {test_data.shape}")

    # Set the 'UID' column as the index
    train_data.set_index('UID', inplace=True)

    # Display the updated DataFrame to confirm the change
    # print(train_data.head())

    # Set the 'UID' column as the index
    test_data.set_index('UID', inplace=True)

    # Display the updated DataFrame to confirm the change
    # print(test_data.head())

    # Define the mapping for 'Target' column
    target_mapping = {'low': 0, 'medium': 1, 'high': 2}

    # Apply the mapping to the 'Target' column
    train_labels = train_data['Target'].map(target_mapping)
    train_data['Target'] = train_labels

    # Display the first few rows of the labels to verify the mapping
    # print(train_labels.head())

    train_data = train_data.drop(columns=['TownId','Target','DistrictId'])

    test_data = test_data.drop(columns=['TownId','DistrictId'])  

    # use knn imupter to fill missing values
    imputer = KNNImputer(n_neighbors=2)
    train_data = pd.DataFrame(imputer.fit_transform(train_data), columns = train_data.columns, index=train_data.index)

    test_data = pd.DataFrame(imputer.transform(test_data), columns = test_data.columns, index=test_data.index)

    # Standardize the features before training
    scaler = StandardScaler()
    train_data = pd.DataFrame(scaler.fit_transform(train_data), columns=train_data.columns, index=train_data.index)
    test_data = pd.DataFrame(scaler.transform(test_data), columns=test_data.columns, index=test_data.index)

    return train_data, test_data, train_labels


# In[6]:


def col_with_null(train_data, null_threshold):
    missing_values = train_data.isnull().sum()

    # Create a DataFrame to store the count of missing values
    missing_df = pd.DataFrame({
        'Column': missing_values.index,
        'MissingCount': missing_values.values
    })

    # Add a column to show the percentage of missing values
    missing_df['MissingPercentage'] = (missing_df['MissingCount'] / len(train_data)) * 100

    # Sort the DataFrame by the number of missing values in descending order
    missing_df.sort_values(by='MissingCount', ascending=False, inplace=True)

    # Reset index for readability
    missing_df.reset_index(drop=True, inplace=True)

    # drop columns with missing percentage greater than threshold
    columns_to_drop = missing_df[missing_df['MissingPercentage'] > null_threshold]['Column'].tolist()

    return columns_to_drop


# In[7]:


def preprocessing_func(data, columns_to_drop):

    data.drop(columns=columns_to_drop, axis=1, inplace=True)

    # Set the 'UID' column as the index
    data.set_index('UID', inplace=True)

    train_labels = None

    if 'Target' in data.columns:
        # Define the mapping for 'Target' column
        target_mapping = {'low': 0, 'medium': 1, 'high': 2}

        # Apply the mapping to the 'Target' column
        train_labels = data['Target'].map(target_mapping)
        data['Target'] = train_labels

        data = data.drop(columns=['TownId','Target','DistrictId'])
    else:
        data = data.drop(columns=['TownId','DistrictId'])

    # use knn imupter to fill missing values
    imputer = KNNImputer(n_neighbors=2)
    data = pd.DataFrame(imputer.fit_transform(data), columns = data.columns, index=data.index)

    # Standardize the features before training
    scaler = StandardScaler()
    data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns, index=data.index)

    if train_labels is None:
        return data
    else:
        return data, train_labels


# In[8]:


def get_model(train_data, train_labels):
    cat = CatBoostClassifier(
        random_seed=seed,
        class_weights=[3, 1, 3],
        iterations=2000,
        learning_rate=0.1,
    )

    # fit the model
    cat.fit(train_data, train_labels)

    # get f1
    y_pred = cat.predict(train_data)
    cat_f1 = f1_score(train_labels, y_pred, average='macro')

    lgbm = LGBMClassifier(
        random_state=seed,
        # class weight
        class_weight='balanced',
        n_estimators=300,
        subsample=0.8,
        reg_lambda=0.1,
        reg_alpha=0.1,
        learning_rate=0.1,
        num_leaves=300,
        max_depth=10,
        min_child_samples=50,
        colsample_bytree=0.8,
    )

    lgbm.fit(train_data, train_labels)

    # get f1 score
    y_pred = lgbm.predict(train_data)
    lgbm_f1 = f1_score(train_labels, y_pred, average='macro')

    # Initialize the Random Forest Classifier
    rf = RandomForestClassifier(
        n_estimators=200,
        min_samples_split=10,
        min_samples_leaf=4,
        max_features=5,
        max_depth=10,
        bootstrap=False,
        random_state=seed,
        n_jobs=-1, 
        class_weight='balanced',
    )

    # fit the model
    rf.fit(train_data, train_labels)

    # get f1
    y_pred = rf.predict(train_data)
    rf_f1 = f1_score(train_labels, y_pred, average='macro')

    # weighted ensemble
    model = VotingClassifier(
        estimators=[
            ('cat', cat),
            ('lgbm', lgbm),
            ('rf', rf),
            # ('xgb', xgb)
        ],
        voting='soft',
        n_jobs=-1,
        weights=[cat_f1, lgbm_f1, rf_f1]
    )

    # Fit the model
    model.fit(train_data, train_labels)
    
    return model


# In[9]:


def make_predictions(model, test_fname, predictions_fname, columns_to_drop):
    #TODO: complete this function to save predictions to the csv file predictions_fname
    #this is an example, you need to modify the code below to fit your workflow
    #### start code ####
    test = pd.read_csv(test_fname)
    # fill_na_values(test, features, vals)
    test_X = preprocessing_func(test, columns_to_drop)
    # test_X = test[features].to_numpy()
    preds = model.predict(test_X)
    # test_uid = test[["UID"]].copy()
    # test_uid["Target"] = preds.reshape(-1)
    # test_uid.to_csv(predictions_fname, index=False)
    # Save the predictions to a CSV file
    output = pd.DataFrame({'UID': test_X.index, 'Target': preds})

    output['Target'] = output['Target'].map({0: 'low', 1: 'medium', 2: 'high'})

    output.to_csv(f'{predictions_fname}', index=False)
    #### end code ####


# In[ ]:


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--train-file", type=str, help='file path of train.csv')
    parser.add_argument("--test-file", type=str, help='file path of test.csv')
    parser.add_argument("--predictions-file", type=str, help='save path of predictions')
    args = parser.parse_args()

    # train_data = pd.read_csv(args.train_file)
    train_data = pd.read_csv("./train.csv")
    columns_to_drop = col_with_null(train_data, 10)
    train_data, train_labels = preprocessing_func(train_data, columns_to_drop)
    model = get_model(train_data, train_labels)

    make_predictions(model, args.test_file, args.predictions_file, columns_to_drop)

