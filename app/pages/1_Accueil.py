import streamlit as st
# from routes import redirection


# En-tête principal avec image
st.title("Call Of Code ⚔️")

st.subheader("Application d'analyse et de prédiction de données")   
st.markdown("Bienvenue sur notre pipeline d'analyse et prédiction")

with st.expander("ℹ️ Comment ça marche ?"):
    st.info("""
        Tout au long de votre parcours, des onglets rétractables comme celui-ci vous aideront dans l'utilisation de l'application.\n
        Vous utiliserez le menu de gauche pour naviguer dans les pages de l'application et certaines pages comportent également des onglets.
    """)

st.markdown("""
            # Dans cette application, voici ce que vous allez pouvoir faire dans les différentes pages :\n
            ## Partie 1 du projet - Data Analyse \n
            - ### 📥 Chargement des données : 
                * Charger des données depuis une base de données SQL,
                * Charger des données depuis un fichier CSV local,
                * Charger des données depuis une API externe,
                * Visualiser un aperçu des données chargées.     
            - ### 🔍 Exploration et Traitements : 
                * Choisir la colonne cible et si vous voulez faire de la Classification ou de la Régression,
                * Observer la distribution des variables,
                * Encoder la cible si besoin puis observer les corrélations et choisir les colonnes à conserver en fonction,
                * Effectuer la gestion des valeurs manquantes et des valeurs aberrantes,
                * Standardiser les données si nécessaire,
                * Exporter le résultat en CSV ou XLSX et générer un rapport PDF des observations et traitements effectués. 

            ## Partie 2 du projet - IA \n          
            - ### 🦾 Entraînement d'un modèle : 
                * Effectuer la séparation du jeu de données (entraînement/test) puis sélectionner le meilleur modèle pour votre modélisation,
                * Entraîner le modèle sélectionné et l'exporter au format pickles,
                * Optimiser automatiquement les Hyperparamètres puis exporter le modèle optimisé au format pickles.                
            - ### 📝 Évaluations : 
                * Évaluer les performances du modèle
            - ### 🔮 Prédictions : 
                * Effectuer des prédictions sur de nouvelles données
            """)
