import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import pickle
import re
# Define a function to run anomaly detection
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
df = run_anomaly_detection(df)