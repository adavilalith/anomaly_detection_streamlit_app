import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import pickle

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

# Define the Streamlit app
def main():
    st.title('Credit Card Transactions Anomaly Detection')
    
    st.write("Upload a CSV file with credit card transactions.")

    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="xlsx")
    
    if uploaded_file is not None:
        # Read the CSV file
        df = pd.read_excel(uploaded_file, sheet_name='creditcard_test')
        inds=[]
        for c in df.columns:
            s=set()
            for i in df[c].unique():
                s.add(type(i))
            if len(s)>1:
                inds.extend(df[df[c].apply(lambda x: isinstance(x, str))].index)
        df.drop(inds,axis=0,inplace=True)
        df = run_anomaly_detection(df)
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
