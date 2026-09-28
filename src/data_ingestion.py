import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

def fetch_openagenda_events():
    """
    Récupère les événements depuis l'API OpenDataSoft d'OpenAgenda.
    Filtre par ville et période selon le .env.
    """
    base_url = os.getenv("OPENDATA_API_URL")
    city = os.getenv("CITY", "Lille")
    history_years = int(os.getenv("HISTORY_YEARS", 1))
    
    # Date actuelle (2026-09-24)
    current_date = datetime.now().date()
    # On peut aussi fixer la date si nécessaire pour le POC comme demandé (2026)
    # current_date = datetime(2026, 9, 24).date()
    start_date = current_date - timedelta(days=history_years * 365)
    
    # Format ISO pour l'API
    start_date_str = start_date.strftime("%Y-%m-%dT%H:%M:%SZ")
    
    # Construction de la requête Opendatasoft V2.1
    # On filtre par ville (location_city ou location_name contient la ville)
    # Et par date (lastdate_end >= start_date)
    where_query = f"location_name like '{city}' or location_address like '{city}'"
    # Note: L'API peut avoir location_city mais dans l'exemple il était null. 
    # Souvent location_name contient l'adresse complète.
    
    # Pour être plus précis sur la ville :
    where_query = f"location_address like '{city}'"
    # Et la date :
    where_query += f" and lastdate_end >= '{start_date_str}'"

    max_events = int(os.getenv("MAX_EVENTS", 100))
    limit = 100  # Limite par requête imposée par l'API
    all_events = []
    offset = 0
    
    while len(all_events) < max_events:
        params = {
            "where": where_query,
            "limit": min(limit, max_events - len(all_events)),
            "offset": offset
        }
        
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        
        data = response.json()
        results = data.get("results", [])
        
        if not results:
            break
            
        all_events.extend(results)
        offset += len(results)
        
        # Sécurité pour éviter les boucles infinies si le nombre total est atteint
        if len(results) < params["limit"]:
            break
            
    return all_events

def process_events(events):
    """
    Nettoie et structure les événements avec Pandas.
    """
    if not events:
        return pd.DataFrame()
        
    df = pd.DataFrame(events)
    
    # Sélection des colonnes pertinentes
    columns = [
        'uid', 'title_fr', 'description_fr', 'longdescription_fr', 
        'location_name', 'location_address', 'firstdate_begin', 'lastdate_end'
    ]
    # Vérifier si les colonnes existent
    available_columns = [col for col in columns if col in df.columns]
    df = df[available_columns]
    
    # Nettoyage simple : supprimer les balises HTML si présentes dans longdescription
    if 'longdescription_fr' in df.columns:
        df['longdescription_fr'] = df['longdescription_fr'].str.replace(r'<[^>]*>', '', regex=True)
        
    # Combiner description et longdescription pour la vectorisation
    df['full_description'] = df['title_fr'].fillna('') + ". " + \
                             df['description_fr'].fillna('') + ". " + \
                             df.get('longdescription_fr', pd.Series(['']*len(df))).fillna('')
                             
    return df

if __name__ == "__main__":
    events = fetch_openagenda_events()
    print(f"Nombre d'événements récupérés : {len(events)}")
    if events:
        df = process_events(events)
        print(df.head())
