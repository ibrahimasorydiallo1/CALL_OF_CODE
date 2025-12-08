import os
import streamlit as st
from routes import routes
import logging

st.set_page_config(page_title="Data analyse", page_icon="🌕", layout="wide")

logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
logging.getLogger("sqlalchemy.pool").setLevel(logging.WARNING)

os.system("python app/api.py")
pages = st.navigation(routes)
pages.run()
