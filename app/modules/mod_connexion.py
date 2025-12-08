# from PIL import Image
# import streamlit as st
# import base64
# import re

# # USERS = st.secrets["users"]

# st.set_page_config(
#     page_title="Page login",
#     page_icon="🔐",
#     layout="wide",
#     initial_sidebar_state="collapsed"
# )

# sidebar_logo = Image.open("photo/ai_logo_4.png")
# st.sidebar.image(sidebar_logo, use_container_width=True)

# def get_base64_image(image_path):
#     with open(image_path, "rb") as img_file:
#         encoded = base64.b64encode(img_file.read()).decode()
#     return f"data:image/png;base64,{encoded}"

# def get_users_from_sheet():
#     """Fetch usernames and passwords from Google Sheets"""
#     records = sheet.get_all_records()
#     users = {record["username"]: record["password"] for record in records}
#     return users


# def register_user(username, password):
#     """Write a new user to the Google Sheet"""
#     sheet.append_row([username, password])
#     # sheet.append_row([username.strip(), password.strip()])


# def user_exists(email):
#     users = get_users_from_sheet()
#     return email in users


# def login_screen():
#     col_left, col_main1, col_right = st.columns([1, 8, 1])

#     with col_main1:
#         col1, col2 = st.columns([7, 4])

#         with col2:
#             # Create two tabs: Login and Register
#             tab_login, tab_register = st.tabs(["Login", "Register"])
#             # tab_login = st.tabs("Login")

#             with tab_login:
#                 with st.form("login_form"):
#                     username = st.text_input("Username")
#                     password = st.text_input("Password", type="password")
#                     login_button = st.form_submit_button("Se connecter")

#                     if login_button:
#                         USERS = get_users_from_sheet()
#                         if username.strip() in USERS and USERS[username.strip()] == password.strip():
#                             st.session_state["authenticated"] = True
#                             st.session_state["username"] = username.strip()
#                             st.success("Login successful. Redirecting...")
#                             st.switch_page("pages/AITC.py")
#                         else:
#                             st.error("Incorrect username or password")

#             with tab_register:
#                 with st.form("register_form"):
#                     new_username = st.text_input("Choose a username")
#                     new_password = st.text_input("Choose a password", type="password")
#                     register_button = st.form_submit_button("Register")

#                     if register_button:
#                         if user_exists(new_username):
#                             st.error("Username already exists. Please choose a different one.")
#                         else:
#                             register_user(new_username, new_password)
#                             st.success("Registration successful! You can now log in.")

#         with col1:
#             ai_logo = get_base64_image("photo/ai_logo_4.png")
#             st.markdown(f"""
#                     <div style='display: flex; align-items: center; gap: 16px; margin-bottom: 1px;'>
#                         <img src='{ai_logo}' width='900'>
#                     </div>
#                     """, unsafe_allow_html=True)


# # Auth check
# if "authenticated" not in st.session_state:
#     st.session_state["authenticated"] = False

# if not st.session_state["authenticated"]:
#     st.markdown("""
#         <style>
#             section[data-testid="stSidebar"] {
#                 display: none;
#             }
#             div[data-testid="collapsedControl"] {
#                 display: none;
#             }
#         </style>
#     """, unsafe_allow_html=True)

#     login_screen()
# else:
#     st.switch_page("./pages/1_Accueil.py")