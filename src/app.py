import faiss
import pandas as pd

from src.product_retrieval.config import Config
from src.product_retrieval.index_faiss import retrieve_products_semantic


products = pd.read_csv(Config.PRODUCTS_PATH, header=0, index_col=0)
index = faiss.read_index(Config.PRODUCTS_INDEX_PATH.__str__())


def search(text: str, k=6):
    results = retrieve_products_semantic(text=text, products=products, k=k, index=index)
    print(results)


search("red paint")
