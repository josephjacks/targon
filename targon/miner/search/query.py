import os

from serpapi import GoogleSearch
from pydantic import BaseModel

class QueryParams(BaseModel):
    q: str
    location: str = "US"
    hl: str = "Google UI Language",
    gl: str = "Google Country",
    google_domain: str = "google.com",
    api_key: str = os.environ['SERP_API_KEY'] if os.environ['SERP_API_KEY'] else None


        
def query(params: QueryParams):
    assert params.api_key, "SERP_API_KEY not found in .env"


    search = GoogleSearch(params)
    results = search.get_dict()

    return results