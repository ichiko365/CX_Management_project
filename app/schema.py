from pydantic import BaseModel, Field
from typing import Optional

# ---------------------------------------------------------------------------
# INPUT MODEL: What a user sends to your API.
# ---------------------------------------------------------------------------
class ReviewInput(BaseModel):
    """Defines the data structure for a new review submitted by a user."""
    ASIN: str = Field(..., example="B002K6AHQY")
    Title: str = Field(..., example="Great Product!")
    Review: str = Field(..., example="This is the best product ever.")
    Region: Optional[str] = Field(None, example="Bengaluru")


# ---------------------------------------------------------------------------
# DATABASE & RESPONSE MODEL: The complete record in your 'raw_reviews' table.
# ---------------------------------------------------------------------------
class ReviewDB(BaseModel):
    """
    Represents a full review record as it exists in the database.
    This schema now exactly matches your table columns.
    """
    id: int
    ASIN: str
    Title: Optional[str] = None
    Description: Optional[str] = None
    ImageURL: Optional[str] = None
    Rating: Optional[float] = None
    Verified: Optional[bool] = None
    ReviewTime: Optional[str] = None
    Review: str
    Summary: Optional[str] = None
    # Use Field(alias=...) for column names with spaces
    Domestic_Shipping: Optional[str] = Field(None, alias="Domestic Shipping")
    International_Shipping: Optional[str] = Field(None, alias="International Shipping")
    Sentiment: Optional[float] = None
    Region: Optional[str] = None
    analysis_status: str = "pending"

    class Config:
        from_attributes = True
        # Allow Pydantic to work with field names that have spaces via their alias
        populate_by_name = True