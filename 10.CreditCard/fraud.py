import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

def load_datasets():
    '''
    Load csv's into dataframes
    '''
    fraud = pd.read_csv('./Fraud_Data.csv')
    ip_country = pd.read_csv('./IpAddress_to_Country.csv')
    return fraud, ip_country

def match_country_to_ip(ip=None):
    '''
    Match an IP address with country name
    INPUT:
        ip - ip address, float
    '''
    match = (ip <= ip_country['upper_bound_ip_address']) & ( ip >= ip_country['lower_bound_ip_address'])
    if match.any():
        return ip_country['country'][match].to_string(index=False)
    else:
        return 'unknown'


def add_countries():
    '''
    Add countries to fraud dataframe
    '''
    fraud['country'] = fraud['ip_address'].apply(lambda x: match_country_to_ip(x))

def clean_up_data_frame():
    '''
    Add features to data frame,
    Remove unused features from dataframe
    '''
    purchase = fraud['purchase_time'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    signup = fraud['signup_time'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    fraud['time_to_purchase'] = purchase - signup
    fraud['time_to_purchase'] = fraud['time_to_purchase'].apply(lambda x: x.days)
    fraud.drop([' "user_id"', 'signup_time', 'purchase_time',
        'device_id', 'ip_address', 'sex', 'country'], axis=1, inplace=True)

def dummify_data_frame(fraud):
    '''
    Create dummy variables for categorical features
    INPUT:
        - fraud: fraud data framerame
    OUTPUT:
        - new_fraud: fraud data frame
            with dummified variables
    '''
    cols = ['source', 'browser']
    new_fraud = pd.get_dummies(fraud, columns=cols)
    return new_fraud


def make_train_test_split():
    '''
    Create training and testing sets
    '''
    y = fraud['class']
    X = fraud.drop('class', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    return X_train, X_test, y_train, y_test

def fit_random_forest_classifier():
    '''
    Create random forest classifier
    '''
    rf = RandomForestClassifier(class_weight = {0: .9, 1: .1})
    rf.fit(X_train, y_train)
    return rf

def make_prediction_give_score():
    '''
    Make a prediction with the fit random forest
    model and score is using accuracy, recall,
    precision, and f1
    '''
    predictions = rf.predict(X_test)
    f1 = f1_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    accuracy = rf.score(X_test, y_test)
    
    print "f1 score: {}".format(f1)
    print "#######################"
    print "recall: {}".format(recall)
    print "#######################"
    print "precision: {}".format(precision)
    print "#######################"
    print "accuracy: {}".format(accuracy)

def display_feature_importance():
    '''
    Print most important features of random forest model
    '''
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    print "Feature Ranking"

    for f in range(X_train.shape[1]):
        print "{}. {}: {}".format(f+1, X_train.columns[f], importances[indices[f]])

def make_percent_fraud_by_country_plot():
    '''
    Plot precentage of fraud by country
    '''
    total_fraud = fraud['class'].sum()
    country_fraud = fraud.groupby('country').sum()
    sorted_country_fraud = country_fraud['class'].sort_values(ascending=False)
    
    y = sorted_country_fraud[:10]/float(total_fraud)
    x = np.arange(len(y))
    
    fig, ax = plt.subplots()

    ax.bar(x, y)
    ax.bar(x, y)
    ax.set_ylabel('Percentage Of Total Fraud')
    ax.set_title('Percentage Of Total Fraud By Country')
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_country_fraud.index.values[:10], rotation=45)
    ax.grid(False)
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.savefig('percent_fraud_by_country')


if __name__ == '__main__':

    fraud, ip_country = load_datasets()

    add_countries()

    fraud.to_csv('fraud_with_features.csv')

    make_percent_fraud_by_country_plot()

    clean_up_data_frame()

    fraud = dummify_data_frame(fraud)

    X_train, X_test, y_train, y_test = make_train_test_split()

    rf = fit_random_forest_classifier()

    make_prediction_give_score()

    display_feature_importance()
