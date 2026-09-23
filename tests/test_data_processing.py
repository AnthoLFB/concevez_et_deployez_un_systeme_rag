import os
import pytest
import pandas as pd
from src.data_ingestion import fetch_openagenda_events, process_events

def test_fetch_openagenda_events():
    """Vérifie qu'on récupère bien des événements."""
    events = fetch_openagenda_events()
    assert isinstance(events, list)
    # On devrait au moins avoir quelques événements pour Lille en 2026/2025
    assert len(events) > 0

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

if __name__ == "__main__":
    pytest.main([__file__])
