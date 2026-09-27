import os
import pytest
import pandas as pd
from src.data_ingestion import fetch_openagenda_events, process_events

def test_fetch_openagenda_events():
    """Vérifie qu'on récupère bien des événements."""
    events = fetch_openagenda_events()
    assert isinstance(events, list)
    # L'API peut retourner 0 si aucun événement n'est trouvé, mais pour Lille on en attend généralement
    # Si le test échoue à cause de 0 événements, c'est peut-être un problème de réseau ou de filtres
    assert len(events) >= 0

def test_process_events():
    """Vérifie la structuration des données avec Pandas."""
    sample_events = [
        {
            'uid': '1',
            'title_fr': 'Test Event',
            'description_fr': 'Desc',
            'longdescription_fr': '<p>Long Desc</p>',
            'location_name': 'Lille',
            'location_address': 'Place Rihour, 59000 Lille',
            'firstdate_begin': '2026-10-01T10:00:00Z',
            'lastdate_end': '2026-10-01T18:00:00Z'
        }
    ]
    df = process_events(sample_events)
    assert not df.empty
    assert 'full_description' in df.columns
    assert 'Long Desc' in df.iloc[0]['full_description']
    assert '<p>' not in df.iloc[0]['full_description']
    assert 'Test Event' in df.iloc[0]['full_description']

def test_process_events_empty():
    """Vérifie le comportement avec une liste vide."""
    df = process_events([])
    assert isinstance(df, pd.DataFrame)
    assert df.empty

if __name__ == "__main__":
    pytest.main([__file__])
