import streamlit as st
import bcrypt

from sqlalchemy import text   # sert à exécuter des requêtes SQL brutes

# from routes import connection_db

# --- CONFIG ---
st.set_page_config(page_title="Login", page_icon="🔐", layout="centered")

# --- DB CONNECTION (Assurez-vous que cette connexion est configurée dans votre secrets.toml) ---
try:
    # conn = connection_db()
    conn = st.connection("postgresql", type="sql")
except Exception as e:
    st.error(f"Erreur de connexion à la base de données : {e}")
    st.stop()


# --- UTILS (SÉCURITÉ ET DB) ---
def hash_password_bcrypt(password: str) -> str:
    """Hashe le mot de passe en utilisant bcrypt pour plus de sécurité (avec salage auto)."""
    # bcrypt.gensalt() génère le sel, hashpw hache le mot de passe avec le sel
    # .decode('utf-8') est nécessaire pour stocker la chaîne dans PostgreSQL
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password_bcrypt(plain_password: str, hashed_password: str) -> bool:
    """Vérifie le mot de passe clair contre le hachage stocké."""
    try:
        # Vérifie si le mot de passe clair correspond au hachage (le sel est inclus dans le hachage stocké)
        return bcrypt.checkpw(
            plain_password.encode("utf-8"), hashed_password.encode("utf-8")
        )
    except ValueError:
        # Gère les cas où le hachage stocké n'est pas un hachage bcrypt valide
        return False


def user_exists(email: str) -> bool:
    """Vérifie l'existence d'un utilisateur par email."""
    query = "SELECT 1 FROM users WHERE email = :email"
    result = conn.query(query, params={"email": email.strip()})
    return not result.empty


def register_user(
    email: str, f_name: str, l_name: str, password: str, role: str = "reader"
):
    """Insère un nouvel utilisateur dans la base de données après hachage du mot de passe."""
    # Hachage sécurisé
    hashed_password = hash_password_bcrypt(password)

    query = """
        INSERT INTO users (email, f_name, l_name, password, role)
        VALUES (:email, :f_name, :l_name, :password, :role) 
    """

    # Assainissement des entrées avant l'insertion
    params = {
        "email": email.strip(),
        "f_name": f_name.strip(),
        "l_name": l_name.strip(),
        "password": hashed_password,
        "role": role.strip(),
    }

    try:
        # Obtenir la session SQL brute de la connexion Streamlit
        session = conn.session

        session.execute(text(query), params)

        # Committer la transaction à la base de données
        session.commit()

        return True
    except Exception as e:
        session.rollback()   # Annule la transaction en cas d'erreur
        st.error(f"Erreur lors de l'enregistrement en base de données : {e}")
        return False


def check_login(email: str, password: str) -> dict:
    """Vérifie les identifiants et retourne les informations de l'utilisateur."""

    # 1. Récupérer le hachage et les données utilisateur
    query = """
        SELECT f_name, role, password FROM users
        WHERE email = :email
    """

    # Assainissement de l'email
    result = conn.query(query, params={"email": email.strip()}, ttl=0)

    if result.empty:
        return None  # Utilisateur non trouvé

    user_data = result.iloc[0].to_dict()
    hashed_password_db = user_data.pop("password")  # Récupère le hachage stocké

    # 2. Vérifier le mot de passe avec bcrypt
    if verify_password_bcrypt(password, hashed_password_db):
        return user_data  # Connexion réussie, retourne le prénom et le rôle
    else:
        return None  # Mot de passe incorrect


# --- SESSION INIT ---
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if "user" not in st.session_state:
    st.session_state["user"] = None


# --- LOGOUT ---
def logout():
    """Déconnecte l'utilisateur et recharge l'application."""
    st.session_state["authenticated"] = False
    st.session_state["user"] = None
    st.success("Déconnexion réussie.")
    st.rerun()


# --- LOGIN SCREEN ---
def login_screen():
    """Affiche les formulaires de connexion et d'enregistrement."""
    st.title("🔐 Connexion & Enregistrement")

    tab_login, tab_register = st.tabs(["Se connecter", "Créer un compte"])

    with tab_login:
        with st.form("login_form"):
            email = st.text_input("Email", key="login_email")
            password = st.text_input(
                "Mot de passe", type="password", key="login_password"
            )
            submit = st.form_submit_button("Connexion")

            if submit:
                if not email or not password:
                    st.error("Veuillez remplir tous les champs.")
                    st.stop()

                user = check_login(email, password)

                if user:
                    st.session_state["authenticated"] = True
                    st.session_state["user"] = user
                    st.success("Connexion réussie")
                    st.rerun()
                else:
                    st.error("Identifiants incorrects.")

    with tab_register:
        with st.form("register_form", clear_on_submit=True):
            email = st.text_input("Email", key="reg_email")
            f_name = st.text_input("Prénom", key="reg_f_name")
            l_name = st.text_input("Nom", key="reg_l_name")
            password = st.text_input(
                "Mot de passe", type="password", key="reg_password"
            )
            submit = st.form_submit_button("Créer le compte")

            if submit:
                # Validation côté client
                if not email or not f_name or not l_name or not password:
                    st.error("Veuillez remplir tous les champs.")
                    st.stop()

                if user_exists(email):
                    st.error("Utilisateur avec cet email déjà existant.")
                else:
                    if register_user(email, f_name, l_name, password):
                        st.success(
                            "Compte créé avec succès. Vous pouvez maintenant vous connecter."
                        )
                    # Pas de st.rerun ici, on laisse l'utilisateur basculer sur l'onglet de connexion
