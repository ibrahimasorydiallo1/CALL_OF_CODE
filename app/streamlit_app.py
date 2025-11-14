import streamlit as st
import pandas as pd

st.set_page_config(page_title="Data analyse", page_icon="🌕", layout="wide")

st.title("Call Of Code ⚔️")

df_csv = pd.read_csv("app\\assets\production_2025_10.csv", sep=";")
st.write("Ceci est le csv affiché")
st.write(df_csv)