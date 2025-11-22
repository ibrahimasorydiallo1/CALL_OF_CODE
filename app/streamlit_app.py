import streamlit as st
import pandas as pd

st.set_page_config(page_title="Data analyse", page_icon="🌕", layout="wide")

st.title("Call Of Code ⚔️")

# Initialize connection.
conn = st.connection("postgresql", type="sql")

# Perform query.
st.subheader("Table turbine")
df = conn.query('SELECT * FROM turbine;', ttl="10m")

st.write(df)

df_csv = pd.read_csv("app\production_2025_10.csv", sep=";")
st.write("Ceci est le csv affiché")
st.write(df_csv)