from pydantic import BaseModel, Field
from typing import List, Optional

from src.product_retrieval.archive.dataframe import schema, sample_data

# Define a Pydantic model for a product
class Product(BaseModel):
    name: str #ID
    url: str
    description: str
    brand: Optional[str]
    price: float
    currency: str
    breadcrumbs: List[str]
    overview: str
    specifications: List[str]

from typing import List, Optional, Dict, Any

# Map Pandas data types to Pydantic/Python types
pandas_to_pydantic_type = {
    'string': str,
    'float': float,
    'object': Any  # General object type, could also specify List[Any] for specific lists
}

# Function to dynamically create Pydantic model from Pandas schema
def create_pydantic_model(model_name: str, schema: Dict[str, str]):
    # Dictionary to store the fields of the Pydantic model
    fields = {}
    
    # Iterate over the schema and convert to Pydantic fields
    for field_name, field_type in schema.items():
        pydantic_type = pandas_to_pydantic_type.get(field_type, Any)
        fields[field_name] = (Optional[pydantic_type], None)  # Optional fields with default None

    # Dynamically create the Pydantic model class
    return type(model_name, (BaseModel,), fields)


# Dynamically create a Pydantic model based on the schema
ProductModel = create_pydantic_model('ProductModel', schema)


# Validate the data using the dynamically created model
product = ProductModel(**sample_data)
print(product)
