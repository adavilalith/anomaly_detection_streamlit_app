import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
import pickle
import re
import numpy as np

# model = keras.models.load_model('./autoencoder_model.keras')

# Define a function to run anomaly detection
# autoencoer model but wont load in streamlit, works in test
# def run_anomaly_detection1(df):
#     df2 = df.drop(['Time'], axis=1)
#     df2 = df2.astype(np.float64)
#     mean = df2.mean(axis=0)
#     std = df2.std(axis=0)
#     df2 = (df2 - mean) / std

#     df2 = df2.to_numpy()

#     y_pred = model.predict(df2)
#     mse = np.mean(np.power(df2 - y_pred, 2), axis=1)
    
#     y_pred = [1 if e>24 else 0 for e in mse]

#     df['anomaly'] = y_pred
#     return df


def run_anomaly_detection(df):
    df1 = df.rename(columns={"Amount":"scaled_amount","Time":"scaled_time"})

    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df1)
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
        pred = model.predict(df_scaled)
        df['anomaly'] = [1 if label==-1 else 0 for label in pred]
    # Anomaly is labeled as -1, normal as 1
    
    
    return df

def run_linear_model(df):
    # df1 = df.rename(columns={"Amount":"scaled_amount","Time":"scaled_time"})

    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    with open('lmodel.pkl', 'rb') as f:
        model = pickle.load(f)
        pred = model.predict(df_scaled)
        df['anomaly'] = [1 if label==-1 else 0 for label in pred]
    df['anomaly'] = pred
    return df
# Define the Streamlit app
def main():
    st.title('Credit Card Transactions Anomaly Detection')
    
    st.write("Upload a CSV file with credit card transactions.")

    uploaded_file = st.file_uploader("Isolation Forest", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
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
        df = run_linear_model(df)
        # Display the results
        st.write("Data with Anomalies Detected:")
        st.write(df[:100])
        print(sum(df['anomaly']==1),sum(df['anomaly']==0))
        
        # Visualize the anomalies
        st.write("Visualization of Anomalies:")
        fig, ax = plt.subplots()
        sns.barplot(x='anomaly', y=[i for i in range(len(df))],data=df)
        plt.title('Anomaly Detection')
        st.pyplot(fig)
  
    


if __name__ == "__main__":
    main()
