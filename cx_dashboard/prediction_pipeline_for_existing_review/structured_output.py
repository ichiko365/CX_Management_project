from pydantic import BaseModel, Field
from typing import Dict, List

class DashboardAnalysis(BaseModel):
    """
    Defines the rich, structured data the LLM must generate to power the dashboard.
    """
    sentiment: str = Field(description="The overall sentiment: 'Positive', 'Negative', or 'Neutral'")
    
    summary: str = Field(description="A concise, single-sentence summary of the review, written from a neutral, third-person perspective.")
    
    key_drivers: Dict[str, str] = Field(
        description="A dictionary mapping specific features to their sentiment ('Positive', 'Negative'). E.g., {'Color Match': 'Positive'}"
    )
    
    # For the 'Urgent Issues' KPI and queue severity
    urgency_score: int = Field(
        description="An integer from 1 (low) to 5 (critical) indicating how urgently this review needs a human response."
    )
    
    # For the 'Urgent Issues' queue tag pills
    issue_tags: List[str] = Field(
        description="A list of 1-3 short, relevant keyword tags that categorize the main issues, e.g., ['allergic reaction', 'wrong shade']."
    )
    
    # For the 'Category Distribution' chart
    primary_category: str = Field(
        description="The single best product category from the review, e.g., 'Skincare', 'Makeup', 'Haircare', 'Fragrance'."
    )