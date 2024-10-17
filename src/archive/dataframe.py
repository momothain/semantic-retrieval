import json
from pathlib import Path
import pandas as pd

schema: dict[str, str] = {
    "url": "string",
    "name": "string",
    "description": "string",
    "brand": "string",
    "price": "float",
    "currency": "string",
    "breadcrumbs": "object",  # Use 'object' for lists
    "overview": "string",
    "specifications": "object",  # Use 'object' for lists
}


