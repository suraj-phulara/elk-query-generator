from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Dict, Optional


class ProductQuerySchema(BaseModel):
    actual_price: Optional[str] = Field(description="Actual price of the product if mentioned in user query")
    discount_price: Optional[str] = Field(description="Discounted price of the product if mentioned in user query")
    main_category: Optional[str] = Field(description="Main category of the product if mentioned in user query")
    name: Optional[str] = Field(description="Name of the product if mentioned in user query")
    no_of_ratings: Optional[str] = Field(description="Number of ratings the product has received if mentioned in user query")
    ratings: Optional[str] = Field(description="Ratings of the product if mentioned in user query")
    sub_category: Optional[str] = Field(description="Sub-category of the product if mentioned in user query")
    name_synonyms: Optional[Dict[str, List[str]]] = Field(description="Synonyms for each word in the name field")

