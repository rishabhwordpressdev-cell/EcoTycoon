from __future__ import annotations
import io
import pandas as pd
import requests
from dataclasses import dataclass
from typing import Optional


@dataclass
class GBIFQuery:
    scientific_name: Optional[str] = None
    country: Optional[str] = None
    limit: int = 5000


class GBIFClient:
    BASE_URL = "https://api.gbif.org/v1/occurrence/search"

    def search(self, query: GBIFQuery) -> pd.DataFrame:
        params = {
            "limit": min(query.limit, 20000),
        }
        if query.scientific_name:
            params["speciesKey"] = self._get_species_key(query.scientific_name)
        if query.country:
            params["country"] = query.country.upper()

        resp = requests.get(self.BASE_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        records = []
        for result in data.get("results", []):
            records.append(
                {
                    "gbif_id": result.get("gbifID"),
                    "species": result.get("species"),
                    "scientific_name": result.get("scientificName"),
                    "country": result.get("country"),
                    "decimal_latitude": result.get("decimalLatitude"),
                    "decimal_longitude": result.get("decimalLongitude"),
                    "event_date": result.get("eventDate"),
                    "basis_of_record": result.get("basisOfRecord"),
                    "coordinate_uncertainty": result.get("coordinateUncertaintyInMeters"),
                }
            )
        return pd.DataFrame(records)

    def _get_species_key(self, name: str) -> Optional[int]:
        """Look up species key from GBIF species API."""
        try:
            resp = requests.get(
                "https://api.gbif.org/v1/species/match",
                params={"name": name},
                timeout=10,
            )
            if resp.ok:
                return resp.json().get("usageKey")
        except Exception:
            pass
        return None


def load_tabular_file(file_obj) -> pd.DataFrame:
    """Load CSV, Parquet, or JSON from a Streamlit file uploader."""
    name = file_obj.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(file_obj)
    elif name.endswith(".parquet"):
        return pd.read_parquet(file_obj)
    elif name.endswith(".json"):
        return pd.read_json(file_obj)
    else:
        # Try to detect from content
        content = file_obj.read()
        file_obj.seek(0)
        if content.startswith(b"{"):
            return pd.read_json(io.BytesIO(content))
        else:
            return pd.read_csv(io.BytesIO(content))