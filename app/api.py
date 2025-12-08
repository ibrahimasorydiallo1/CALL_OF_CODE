import requests
import json
import shutil
from datetime import datetime, timedelta
from pathlib import Path

def fetch_daily_weather(lat, lon):
    date = (datetime.utcnow() - timedelta(days=1)).date().isoformat()
    # https:// indique que la communication est chiffrée par TLS/SSL.

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": date,
        "end_date": date,
        "hourly": "temperature_2m,relativehumidity_2m,windspeed_10m,pressure_msl",
        "timezone": "Europe/Paris"
    }

    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()

TMP_DIR = Path("app/data/tmp")

# Supprimer le dossier s'il existe (même s'il n'est pas vide)
if TMP_DIR.exists():
    shutil.rmtree(TMP_DIR)

# Recréer le dossier
TMP_DIR.mkdir(parents=True, exist_ok=True)

def save_json(data, filename):
    filepath = TMP_DIR / filename
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return filepath

if __name__ == "__main__":
    print("Récupération des données météo...")

    lat = 45.764043  # Lyon
    lon = 4.835659

    data = fetch_daily_weather(lat, lon)
    name = f"meteo_{datetime.utcnow().date().isoformat()}.json"
    output_file = save_json(data, name)

    print("Données météo récupérées avec succès !")
    print("Fichier sauvegardé :", output_file)
