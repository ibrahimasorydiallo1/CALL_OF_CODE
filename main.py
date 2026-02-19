import streamlit as st

from routes import routes

st.set_page_config(page_title="Data analyse: Projet ETL", page_icon="🌕", layout="wide")

pages = st.navigation(routes)
pages.run()
