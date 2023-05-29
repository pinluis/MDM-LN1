import streamlit as st
import pandas as pd

# Load data
df = pd.read_csv("airlines_delay.csv")

st.title("Airline Delay Data")

# streamlit radio button to select the column
col = st.radio("Select a column", df.columns)

# streamlit slider to select the number of rows
rows = st.slider("Select number of rows", 1, 100, 10)

# Display the data
st.write(df[col].head(rows))
