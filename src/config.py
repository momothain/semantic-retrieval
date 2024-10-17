import os
from pathlib import Path

from dotenv import load_dotenv

from openai import OpenAI

load_dotenv()
# region TYPES
Embedding = list[float]


"""CONSTANTS"""
class Config:
    BASE_DIR = Path().cwd()
    DATA_DIR = BASE_DIR / "src" / "data"
    HOME_DEPOT_DATA_PATH = DATA_DIR / "home_depot_data_2021.csv"
    PRODUCTS_PATH = DATA_DIR / "products.csv"
    PRODUCTS_EMBEDDED_PATH = DATA_DIR / "products_embedded.csv"
    PRODUCTS_INDEX_PATH = DATA_DIR / "products.index"

    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    EMBEDDINGS_MODEL = "text-embedding-3-small"
    EMBEDDINGS_DIM = 1536
    CHAT_MODEL = "gpt-4o"

    client = OpenAI(api_key=OPENAI_API_KEY)
