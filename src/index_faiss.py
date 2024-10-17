import ast
from numpy._typing._array_like import NDArray
import pandas as pd
import faiss
from faiss import IndexFlatL2
import numpy as np

from src.product_retrieval.config import Config
from src.product_retrieval.embed import generate_embedding

FAISS_Embedding = NDArray[np.float32]
FAISS_Query = NDArray


def validate_faiss_query(q: FAISS_Query):
    num_queries, embed_dim = q.shape
    assert (
        embed_dim == Config.EMBEDDINGS_DIM
    ), f"Expected embedding dimension {Config.EMBEDDINGS_DIM}, but got {embed_dim}"


def generate_faiss_embedding_query(text: str) -> FAISS_Query:
    embedding: list[float] = generate_embedding(text)
    query_embedding: FAISS_Embedding = np.array(object=embedding).astype("float32")
    query_matrix: FAISS_Query = query_embedding.reshape(1, -1)

    return query_matrix


def retrieve_products_semantic(text: str, products: pd.DataFrame, index: IndexFlatL2, k = 5):
    # Query FAISS index with a random query vector
    embedding_matrix: FAISS_Query = generate_faiss_embedding_query(text)
    # Number of nearest neighbors to retrieve
    distances, indices = index.search(embedding_matrix, k)  # Perform the search
    # Get the indices of the most similar rows
    print("FAISS returned indices:", indices)

    # Retrieve the corresponding rows from the DataFrame
    matching_rows = products.iloc[
        indices[0]
    ]  # indices[0] gives the list of matched indices
    print("Matching rows from DataFrame:")
    print(matching_rows["name"])
    return matching_rows


def index_embeddings(embeddings: FAISS_Embedding, index: IndexFlatL2):
    # Insert
    index.add(embeddings)
    print(f"Total embeddings in index: {index.ntotal}")

    # Write
    faiss.write_index(index, Config.PRODUCTS_INDEX_PATH.__str__())
    print(f"Index saved to {Config.PRODUCTS_INDEX_PATH}")


if __name__ == "__main__":
    ### PREPARE EMBEDDINGS
    products = pd.read_csv(Config.PRODUCTS_EMBEDDED_PATH, header=0, index_col=0)

    # Step 2: Convert the 'embeddings' column from strings to lists (because it was saved as a string)
    products["embeddings"] = products["embeddings"].apply(ast.literal_eval)
    assert isinstance(
        products["embeddings"][0], list
    ), f"The first embedding is {type(products["embeddings"][0])} instead of a list!"
    # Convert the list of embeddings to a NumPy array (float32 type is required by FAISS)
    embeddings: FAISS_Embedding = np.array(products["embeddings"].tolist()).astype(
        "float32"
    )
    print(embeddings.__len__())
    print(
        f"Embeddings shape: {embeddings.shape}"
    )  # Should be (num_rows, embedding_dim)

    ### FAISS INDEX
    index = faiss.IndexFlatL2(Config.EMBEDDINGS_DIM)  # L2 distance (Euclidean distance)
    print(index.is_trained)

    index_embeddings(embeddings, index)

    # TEST
    retrieve_products_semantic("drywall", products, index)
