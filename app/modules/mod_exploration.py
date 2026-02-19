from datetime import datetime
import pandas as pd
import json
import streamlit as st
from functools import reduce
from dateutil.relativedelta import relativedelta
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
import tempfile
from io import BytesIO
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


# Initialize connection.
conn = st.connection("postgresql", type="sql")
references = st.secrets["filename"]

if "target_corr" not in st.session_state:
    st.session_state["target_corr"] = None


def load_sql_table(table: str) -> pd.DataFrame:
    """
    Charge une table SQL complète dans un DataFrame pandas.

    Args:
        conn (conn):
            Connexion SQLAlchemy vers la base de données.
        table (str):
            Nom de la table SQL à charger.

    Returns:
        pd.DataFrame:
            Le contenu complet de la table.

    Raises:
        Exception: Si la requête SQL échoue.

    Example:
        >>> df = load_sql_table(conn, "turbine")
    """
    return conn.query(f"SELECT * FROM {table};")

def load_clean_csv(path: str) -> pd.DataFrame:
    """
    Charge un fichier CSV et supprime les lignes contenant des valeurs manquantes.

    Args:
        path (str):
            Chemin du fichier CSV à charger.

    Returns:
        pd.DataFrame:
            Le DataFrame propre lu depuis le CSV.

    Raises:
        FileNotFoundError: Si le fichier n'existe pas.
        pd.errors.ParserError: En cas de fichier mal formé.

    Example:
        >>> df = load_clean_csv("app/production_2025_10.csv")
    """
    # return pd.read_csv(path, sep=";").dropna(inplace=True)
    return pd.read_csv(path, sep=";")


def store_table_in_database(df_merged: pd.DataFrame, table_name: str) -> None:
    """
    Stocke un DataFrame pandas dans une table SQL.

    La table est *entièrement remplacée* si elle existe déjà.

    Args:
        df_merged (pd.DataFrame):
            Le DataFrame à insérer dans la base.
        table_name (str):
            Nom de la table SQL cible.

    Returns:
        None

    Raises:
        ValueError: Si le DataFrame est vide.
        Exception: Si l'insertion SQL échoue.

    Example:
        >>> store_table_in_database(df, "combined_data")
    """
    if df_merged.empty:
        raise ValueError("Le DataFrame fourni est vide et ne peut pas être stocké.")

    df_merged.to_sql(
        table_name,
        con=conn,
        if_exists="replace",
        index=False
    )

@st.cache_data
def combine_data_sources() -> pd.DataFrame:
    """
    Combine plusieurs sources de données (SQL + CSV) en un seul DataFrame unifié.

    Cette fonction récupère automatiquement :
    - la table des turbines
    - la table d’inventaire des capteurs
    - la table brute des mesures
    - le fichier CSV de production du mois précédent

    Elle merge ensuite toutes ces sources sur la clé commune `turbine_id`
    en utilisant des jointures internes (INNER JOIN).

    Returns:
        pd.DataFrame:
            Un DataFrame contenant l’ensemble des données fusionnées.
            La table finale contient uniquement les lignes dont `turbine_id`
            existe dans toutes les sources.

    Raises:
        ValueError: Si une source n’a pas de colonne `turbine_id`.
        FileNotFoundError: Si le fichier CSV du mois précédent n’existe pas.
        Exception: Pour toute autre erreur de chargement ou de fusion.

    Notes:
        - Le mois précédent est calculé dynamiquement à partir de la date du jour.
        - La fusion utilise `reduce` afin d'appliquer `pd.merge` successivement.
        - La fonction suppose que les helpers `load_sql_table` et `load_clean_csv`
          sont définis ailleurs dans le projet.

    Example:
        >>> df = combine_data_sources()
        >>> print(df.head())
    """

    now = datetime.now()
    prev = now - relativedelta(months=1)

    # Chargement des sources SQL
    df_turbine   = load_sql_table(references["table_1"])
    df_inventory = load_sql_table(references["table_2"])
    df_raw       = load_sql_table(references["table_3"])

    # df_turbine = df_turbine.dropna(inplace=True)
    # df_inventory = df_inventory.dropna(inplace=True)
    # df_raw = df_raw.dropna(inplace=True)

    # Fichier CSV du mois précédent
    csv_path = f"app/{references['csv']}_{prev.year}_{prev.month}.csv"
    df_prod = load_clean_csv(csv_path)

    # Liste des DataFrames
    dfs = [df_turbine, df_inventory, df_raw, df_prod]

    # Vérification de la présence de la clé turbine_id
    for i, df in enumerate(dfs):
        if "turbine_id" not in df.columns:
            raise ValueError(f"La source n°{i+1} ne contient pas 'turbine_id'.")

    # Fusion progressive
    df_merged = reduce(
        lambda left, right: pd.merge(left, right, on="turbine_id", how="inner"),
        dfs
    )

    # Limiter à 200 lignes
    df_merged = df_merged.head(200)

    return df_merged

def print_api_data():
    """
    Charge, transforme et affiche les données météo horaires depuis un fichier JSON local.

    Cette fonction lit un fichier JSON météo généré à la date UTC du jour,
    extrait les données horaires, les convertit en DataFrame pandas,
    renomme certaines colonnes pour plus de clarté, puis affiche le tout
    dans une interface Streamlit.

    Données affichées :
    - Température (°C)
    - Humidité (%)
    - Vitesse du vent (km/h)
    - Pression atmosphérique (hPa)
    """
    # Charger le JSON brut
    name = f"meteo_{datetime.utcnow().date().isoformat()}.json"
    with open(f"app/data/tmp/{name}", "r", encoding="utf-8") as f:
        data = json.load(f)

    # Extraire les données horaires
    hourly = data["hourly"]

    # Convertir en DataFrame
    df = pd.DataFrame(hourly)

    # Convertir la colonne "time" en datetime
    df["time"] = pd.to_datetime(df["time"])
    df.rename(columns={
                "time": "date",
                "temperature_2m": "temperature",
                "relativehumidity_2m": "humidity",
                "windspeed_10m": "windspeed",
                "pressure_msl": "pressure",
                }, inplace=True)

    print("Température en dégré celsius, humidité en %, vitesse du vent en km/h et pression en hPa (hectoPascal)")
    return df


### Onglet 3 - Corrélations ###

def encoder_cible(df, target, methode, drop_first=False):
    """Encode la target selon la méthode choisie par l'utilisateur"""
    df_copy = df.copy()
    encoded_target_name = "target_encoded"

    if methode == "Label Encoding":
        encoder = LabelEncoder()
        df_copy["target_encoded"] = encoder.fit_transform(df_copy[target])
        # Stockage du mapping inverse pour Label Encoding pour afficher les valeurs initiales avant encodage lors de la prédiction
        st.session_state["mapping_target"] = {i: label for i, label in enumerate(encoder.classes_)}

    elif methode == "One-Hot Encoding":
        ohe = OneHotEncoder(sparse_output=False, drop='first' if drop_first else None)
        encoded_data = ohe.fit_transform(df_copy[[target]])
        encoded_cols = ohe.get_feature_names_out([target])
        df_encoded = pd.DataFrame(encoded_data, columns=encoded_cols, index=df_copy.index)
        # df_copy = pd.concat([df_copy.drop(columns=[target]), df_encoded], axis=1)
        df_copy = pd.concat([df_copy, df_encoded], axis=1)
        df_copy = df_copy.loc[:, ~df_copy.columns.duplicated()].copy()
        encoded_target_name = encoded_cols.tolist()  # Liste de colonnes encodées
        
        # Stockage du mapping inverse pour One-Hot pour afficher les valeurs initiales avant encodage lors de la prédiction
        mapping_inv = {}
        for full_col in encoded_cols:
            # Exemple : 'target_classname'
            if "_" in full_col:
                original_value = full_col.split("_", 1)[1]
                mapping_inv[full_col] = original_value
        st.session_state["mapping_target"] = mapping_inv

    elif methode == "get_dummies":
        df_copy = pd.get_dummies(df_copy, columns=[target], drop_first=drop_first)
        df_copy = df_copy.loc[:, ~df_copy.columns.duplicated()].copy()
        encoded_cols = [col for col in df_copy.columns if col.startswith(target + "_")]
        encoded_target_name = encoded_cols  # Liste des colonnes générées
        
        # Stockage du mapping inverse pour get_dummies
        mapping_inv = {}
        for col_name in encoded_cols:
            original_value = col_name.split("_", 1)[1]
            mapping_inv[col_name] = original_value
        st.session_state["mapping_target"] = mapping_inv

    else:
        raise ValueError("Méthode d'encodage non reconnue.")

    st.session_state["target_corr"] = encoded_target_name  # MAJ de la variable
    return df_copy, encoded_target_name


### Onglet 4 - NaN et Outliers ###

def detecter_outliers_zscore(df, seuil=3.0):
    """Retourne un DataFrame booléen où True indique un outlier selon le Z-score."""
    z_scores = np.abs((df - df.mean()) / df.std(ddof=0))
    return z_scores > seuil

def detecter_outliers_iqr(df, seuil=1.5):
    """Retourne un DataFrame booléen où True indique un outlier selon la méthode IQR."""
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - seuil * IQR
    upper_bound = Q3 + seuil * IQR
    return (df < lower_bound) | (df > upper_bound)

def detection_outliers(df, numeric_cols, methode_outlier, key=None):
    if methode_outlier == "Z-score":
        seuil = st.slider("Seuil Z-score", 2.0, 5.0, 3.0, key=key)
        outliers_bool = detecter_outliers_zscore(df[numeric_cols], seuil)
    else:
        seuil = 1.5
        outliers_bool = detecter_outliers_iqr(df[numeric_cols], seuil)
    return outliers_bool

def traiter_outliers(df, numeric_cols, outliers_bool, action="Supprimer les lignes"):
    """Traite les outliers en fonction de l'action choisie."""
    if action == "Supprimer les lignes":
        to_drop = outliers_bool.any(axis=1)
        return df[~to_drop]
    elif action == "Remplacer par la médiane":
        df_copy = df.copy()
        for col in numeric_cols:
            median = df_copy[col].median()
            df_copy.loc[outliers_bool[col], col] = median
        return df_copy
    return df

### Onglet 5 - Standardisation ###

def standardisation(df, colonne_target):
    # Affichage du df avant standardisation
    st.markdown("**Aperçu du Dataframe AVANT Standardisation**")
    st.dataframe(df.head())
    st.write("***")
    # définir un seuil de proximité de 0
    threshold = 0.1
    # tester si la deviation std et la moyenne sont proche de 0
    cols_to_test = [col for col in df.columns if col != colonne_target and pd.api.types.is_numeric_dtype(df[col])]
    close_to_zero_std = (df[cols_to_test].std().sub(1).abs() < threshold).all()
    close_to_zero_mean = (df[cols_to_test].mean().abs() < threshold).all()
    
    # Si les 2 sont déjà proches de 0 
    if close_to_zero_std and close_to_zero_mean:
        st.write("Vos données semblent déjà standardisées")
    else:
        st.write("Vos données ne semblent pas standardisées")
        standard_box = st.checkbox('Standardiser', key="appli_standardisation")
        if standard_box:
            standardize_data(df, colonne_target)

def standardize_data(df, colonne_target):
    # Convertir en liste si besoin
    if isinstance(colonne_target, str):
        colonne_target = [colonne_target]
    
    non_numeric_columns = [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])]
    non_standardizable_columns = [col for col in non_numeric_columns if col not in colonne_target]
    
    if non_standardizable_columns:
        st.write("Colonnes qui ne sont pas numériques et ne peuvent pas être standardisées :")
        for col in non_standardizable_columns:
            st.write(col)
            
    standardizable_columns = [
        col for col in df.columns 
        if col not in non_standardizable_columns 
        and col not in colonne_target
        and pd.api.types.is_numeric_dtype(df[col])
                              ]
    
    if standardizable_columns:
        scaler = StandardScaler()
        df[standardizable_columns] = scaler.fit_transform(df[standardizable_columns])
        st.success("✅ Les colonnes standardisables ont été standardisées avec succès.")
        # Stockage des session_states pour le résumé et pour la prédiction sur les nouvelles données
        st.session_state["standardized"] = True
        st.session_state["scaler"] = scaler
        st.session_state["standardized_columns"] = standardizable_columns
        st.session_state["standardized_stats"] = df[standardizable_columns].agg(['mean', 'std']).T.round(2)
        # Affichage du df après standardisation
        st.write("***")
        st.markdown("**Aperçu du Dataframe APRÈS Standardisation**")
        st.dataframe(df[standardizable_columns].head())
    else:
        st.warning("Aucune colonne standardisable n'a été trouvée.")
        st.session_state["standardized"] = False
    
    st.session_state["df_clean"] = df
            
            

    ### Onglet 6 - Résumé & Exports ###

def telecharger_donnees(df_clean):
    st.subheader("Télécharger les données")
    format_choisi = st.selectbox("Quel format désirez vous ?", ["CSV", "Excel (.xlsx)"])

    if format_choisi == "CSV":
        data = df_clean.to_csv(index=False).encode('utf-8')
        file_name="donnees_nettoyees.csv"
        mime="text/csv"
        
    else:
        excel_buffer = BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            df_clean.to_excel(writer, sheet_name="données nettoyées", index=False)
        excel_buffer.seek(0)
        data=excel_buffer
        file_name="donnees_nettoyees.xlsx"
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    
    st.success(f"✅ Fichier {file_name} prêt à être téléchargé.")
    st.download_button("📥 Télécharger les données nettoyées", data=data, file_name=file_name, mime=mime)


class PDF(FPDF):
    def header(self):
        self.set_font("Arial", 'B', 12)
        self.cell(0, 10, 'Rapport d\'Exploration de Données', 0, 1, 'C')
        self.ln(10)

    def chapter_title(self, title):
        self.set_font("Arial", 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(4)

    def chapter_body(self, text):
        self.set_font("Arial", '', 10)
        self.multi_cell(0, 10, text)
        self.ln()

def generer_rapport_pdf(df, df_clean, target, to_keep, task, to_drop, corr):
    pdf = PDF()
    pdf.add_page()

    tmpimg_paths = []

    # Graphique de distribution
    if pd.api.types.is_numeric_dtype(df[target]):
        fig1, ax1 = plt.subplots()
        sns.histplot(df[target].dropna(), kde=True, ax=ax1)
        ax1.set_title(f"Distribution de {target}")
        fig1.tight_layout()
        tmpimg = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        fig1.savefig(tmpimg.name)
        plt.close(fig1)
        pdf.image(tmpimg.name, w=180)
        tmpimg_paths.append(tmpimg.name)

    # Heatmap de corrélation
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, annot=False, cmap="coolwarm", ax=ax2)
    ax2.set_title("Matrice de corrélation")
    fig2.tight_layout()    # Ajustement du padding layout
    tmpimg2 = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    fig2.savefig(tmpimg2.name)
    plt.close(fig2)
    pdf.image(tmpimg2.name, w=180)
    tmpimg_paths.append(tmpimg2.name)

    # Bar chart d'une variable catégorielle
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    if cat_cols:
        top_cat = max(cat_cols, key=lambda c: df[c].nunique())
        counts = df[top_cat].value_counts()
        fig3, ax3 = plt.subplots()
        sns.barplot(x=counts.index, y=counts.values, ax=ax3)
        ax3.set_title(f"Répartition de {top_cat}")
        plt.xticks(rotation=45)
        fig3.tight_layout()
        tmpimg3 = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        fig3.savefig(tmpimg3.name)
        plt.close(fig3)
        pdf.image(tmpimg3.name, w=180)
        tmpimg_paths.append(tmpimg3.name)

    # Ajout des sections texte
    pdf.chapter_title("1. Informations générales")
    pdf.chapter_body(f"Nombre de lignes : {df_clean.shape[0]}\nNombre de colonnes : {df_clean.shape[1]}")

    pdf.chapter_title("2. Colonnes conservées")
    pdf.chapter_body(", ".join(to_keep))

    pdf.chapter_title("3. Type de tâche choisi")
    pdf.chapter_body(task)

    pdf.chapter_title("4. Colonnes très corrélées proposées à l'exclusion")
    pdf.chapter_body(", ".join(to_drop) if to_drop else "Aucune")

    pdf.chapter_title("5. Statistiques descriptives")
    desc = df_clean.describe().round(2).T
    pdf.set_font("Arial", "", 8)

    # Entêtes
    col_names = desc.columns.tolist()
    pdf.cell(30, 10, "Statistique", border=1)
    for col in col_names:
        pdf.cell(30, 10, str(col), border=1)
    pdf.ln()

    # Valeurs
    for index, row in desc.iterrows():
        pdf.cell(30, 10, str(index), border=1)
        for val in row:
            pdf.cell(30, 10, str(val), border=1)
        pdf.ln()

    # Sauvegarde du PDF
    tmp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(tmp_pdf.name)

    return tmp_pdf.name, tmpimg_paths
