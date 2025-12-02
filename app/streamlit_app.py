from datetime import datetime
import streamlit as st
import pandas as pd
import json
from functools import reduce
from dateutil.relativedelta import relativedelta

st.set_page_config(page_title="Data analyse", page_icon="🌕", layout="wide")

st.title("Call Of Code ⚔️")

# Initialize connection.
conn = st.connection("postgresql", type="sql")
references = st.secrets["filename"]

# Perform query.
st.subheader("Table turbine")

def load_sql_table(conn, table: str) -> pd.DataFrame:
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
    return pd.read_csv(path, sep=";").dropna()


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
    df_turbine   = load_sql_table(conn, references["table_1"])
    df_inventory = load_sql_table(conn, references["table_2"])
    df_raw       = load_sql_table(conn, references["table_3"])

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

    return df_merged



# Charger le JSON brut
name = f"meteo_{datetime.utcnow().date().isoformat()}.json"
with open(f"tmp/{name}", "r", encoding="utf-8") as f:
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

st.info("Température en dégré celsius, humidité en %, vitesse du vent en km/h et pression en hPa (hectoPascal)")
st.write(df)