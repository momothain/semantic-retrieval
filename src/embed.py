import ast
import pandas as pd

from src.product_retrieval.config import Config, Embedding


def generate_embedding(text: str) -> Embedding:
    response = Config.client.embeddings.create(input=text, model=Config.EMBEDDINGS_MODEL)
    return response.data[0].embedding

if __name__ == "__main__":
    products = pd.read_csv(Config.PRODUCTS_PATH, header=0, index_col=0)
    # Generate embeddings for all rows in 'text_column'
    products["embeddings"] = products["name"].apply(generate_embedding)
    products["embeddings"] = products["embeddings"].apply(ast.literal_eval)

    # Now 'embeddings' should be lists of floats
    print(type(products["embeddings"][0]))  # This should now return <class 'list'>
    products.to_csv(Config.PRODUCTS_EMBEDDED_PATH, index=True, index_label="index")
