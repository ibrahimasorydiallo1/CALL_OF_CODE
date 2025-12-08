from streamlit import Page
import streamlit as st

routes = [
    Page("pages/1_Accueil.py", title="Accueil", icon="🏠"),
    Page("pages/2_Exploration.py", title="Exploration et Traitements", icon="🔍"),
]

def redirection(titre, nom_de_page):
    # Redirection page suivante
    st.write("***")
    st.markdown("Vous pouvez maintenant passer à la page")
    if st.button(titre):
        st.switch_page(page=f"pages/{nom_de_page}.py")