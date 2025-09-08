from typing import Optional

from pydantic import BaseModel, Field, model_validator

# Support both package-style and script-style imports
try:
    from .db_engine import fetch_product_metadata
except Exception:  # pragma: no cover - fallback when executed as a script
    from db_engine import fetch_product_metadata

# ---------------------------------------------------------------------------
# INPUT MODEL: What a user sends to your API.
# ---------------------------------------------------------------------------
class ReviewInput(BaseModel):
    """Incoming payload for a new review.

    Title and Description are optional on input; if omitted, they will be
    auto-filled from the database based on the provided ASIN when possible.
    """
    ASIN: str = Field(..., example="B002K6AHQY")
    Title: Optional[str] = Field(None, example="Great Product!")
    Description: Optional[str] = Field(None, example="Moisturizing serum with vitamin C")
    Review: str = Field(..., example="This is the best product ever.")
    Region: Optional[str] = Field(None, example="Bengaluru")
    ReviewTime: Optional[str] = Field(None, example="2024-01-15T10:30:00")

    @model_validator(mode="after")
    def autofill_from_db(self) -> "ReviewInput":
        """If Title/Description are missing, pull them from DB by ASIN."""
        try:
            meta = fetch_product_metadata(self.ASIN)
        except Exception:
            meta = {}
        if not self.Title and meta.get("Title"):
            self.Title = meta.get("Title")
        if not self.Description and meta.get("Description"):
            self.Description = meta.get("Description")
        return self


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