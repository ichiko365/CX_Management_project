from typing import TypedDict, Literal
from pydantic import BaseModel, Field

# Graph state
class State(TypedDict):
    """Customer service state."""
    messages: list
    order_id: str | None
    order_id_verified: bool | None
    complaint_details: str | None
    complaint_summary: str | None
    department: str | None
    complaint_logged: bool | None

# Classification schema
class DepartmentClassification(BaseModel):
    """Classification of the complaint department."""
    department: Literal["Billing", "Technical Support", "Product Quality", "General Inquiry"] = Field(
        description="The department that should handle this complaint"
    )
