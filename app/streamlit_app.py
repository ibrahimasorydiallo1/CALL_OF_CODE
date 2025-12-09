import subprocess

import streamlit as st

from routes import routes
from modules.mod_connexion import login_screen, logout

st.set_page_config(page_title="Data analyse", page_icon="🌕", layout="wide")

# --- Initialisation de la session ---
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if "user" not in st.session_state:
    st.session_state["user"] = None


# os.system("python app/api.py")  # Pas idéal pour la rapide exécution
subprocess.Popen(["python", "app/api.py"])  # Meilleure approche pour lancer en parallèle

# --- GESTION DE LA SIDEBAR ---

if st.session_state["authenticated"]:
    # --- GESTION DE LA SIDEBAR (SI CONNECTÉ) ---
    user = st.session_state["user"]
    st.sidebar.success(
        f"Connecté : {user.get('f_name', 'Utilisateur')} ({user.get('role', 'N/A')})"
    )

    # Bouton de DÉCONNEXION
    st.sidebar.button("Déconnexion 🔓", on_click=logout, key="sidebar_logout")

    # --- AFFICHAGE DE L'APPLICATION MULTIPAGE ---
    pages = st.navigation(routes)
    pages.run()

else:
    st.sidebar.warning("Veuillez vous connecter.")

    # --- AFFICHAGE DE L'ÉCRAN DE CONNEXION ---
    login_screen()
