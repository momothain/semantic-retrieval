import pandas as pd

from src.product_retrieval.config import Config


products = pd.read_csv(Config.PRODUCTS_PATH, header=0, index_col=0)
print(products.info())
products = products.drop("Unnamed: 0", axis=1)
print(products.info())
products.to_csv(Config.PRODUCTS_PATH, index=True, index_label="index") 
