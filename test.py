import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import pickle
import re
import numpy as np
import tensorflow as tf

# Define a function to run anomaly detection
def run_anomaly_detection1(df):
    df2 = df.drop(['Time'], axis=1)
    df2 = df2.astype(np.float64)
    


    mean = df2.mean(axis=0)
    std = df2.std(axis=0)
    df2 = (df2 - mean) / std

    df2 = df2.to_numpy()

    model = tf.keras.models.load_model('./autoencoder_model.keras')
    y_pred = model.predict(df2)
    mse = np.mean(np.power(df2 - y_pred, 2), axis=1)
    
    y_pred = [1 if e>24 else 0 for e in mse]

    df['anomaly'] = y_pred
    return df
def run_anomaly_detection(df):
    df1 = df.rename(columns={"Amount":"scaled_amount","Time":"scaled_time"})

    # Preprocess the data if necessary
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df1)
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
        pred = model.predict(df_scaled)
        df['anomaly'] = [1 if label==-1 else 0 for label in pred]
    # Anomaly is labeled as -1, normal as 1
    
    
    return df

df = pd.read_csv('./creditcard_test.csv',encoding='utf-8')
inds=[]
print(df.info())
print(df.head())

def is_numeric_string(x):
    if isinstance(x, str):
        try:
            float(x)
            return True
        except:
            return False
    return False

strs=[]
for c in df.columns:
    for i in df[c]:
        if type(i)==str:
            if re.search('[a-zA-Z]', i):
                strs.append(i)
inds=[]
for c in df.columns:
    for s in strs:
        inds.extend(list( df[(df[c]==s)].index))  
df.drop(inds,axis=0,inplace=True)
df = run_anomaly_detection1(df)