import os
import requests
from datetime import datetime
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv


load_dotenv()


BASE_URL = os.getenv("OPENDATA_API_URL")

now = datetime.now()

start = now - relativedelta(years=1)
end = now + relativedelta(years=1)

start_str = start.strftime("%Y-%m-%dT%H:%M:%S+00:00")
end_str = end.strftime("%Y-%m-%dT%H:%M:%S+00:00")


def count_events(where):
    response = requests.get(
        BASE_URL,
        params={
            "select": "count(*)",
            "where": where,
            "limit": 1,
        },
        timeout=30,
    )

    response.raise_for_status()

    data = response.json()

    return data["results"][0]["count(*)"]


filters = {
    "Ville Lille": (
        f'location_city = "Lille" '
        f'and lastdate_end >= "{start_str}" '
        f'and firstdate_begin <= "{end_str}"'
    ),

    "INSEE Lille": (
        f'location_insee = "59350" '
        f'and lastdate_end >= "{start_str}" '
        f'and firstdate_begin <= "{end_str}"'
    ),

    "Ville Lille début période": (
        f'location_city = "Lille" '
        f'and firstdate_begin >= "{start_str}" '
        f'and firstdate_begin <= "{end_str}"'
    ),

    "Ville Lille fin période": (
        f'location_city = "Lille" '
        f'and lastdate_end >= "{start_str}" '
        f'and lastdate_end <= "{end_str}"'
    ),

    "Ville Lille sans date": (
        'location_city = "Lille"'
    ),
}


print("=" * 60)
print("COMPARAISON DES FILTRES OPENAGENDA")
print("=" * 60)

for name, where in filters.items():
    count = count_events(where)
    print(f"{name:<35} : {count}")

print()
print(f"Période : {start_str} -> {end_str}")