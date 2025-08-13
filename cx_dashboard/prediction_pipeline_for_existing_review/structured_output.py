from pydantic import BaseModel, Field
from typing import Dict, List

class DashboardAnalysis(BaseModel):
    """
    Defines the rich, structured data the LLM must generate to power the dashboard.
    """
    sentiment: str = Field(description="Overall sentiment: 'Positive', 'Negative', 'Neutral', or 'Mixed'.")
    
    summary: str = Field(description="A concise, single-sentence summary of the review, written from a neutral, third-person perspective.")
    
    key_drivers: Dict[str, str] = Field(
        default_factory=dict,
        description="A dictionary mapping beauty-related features to their sentiment ('Positive' or 'Negative'). E.g., {'Shade Accuracy': 'Negative'}"
    )
    
    # For the 'Urgent Issues' KPI and queue severity
    urgency_score: int = Field(
        description="An integer from 1 (low) to 5 (critical) indicating how urgently this review needs a human response."
    )
    
    # For the 'Urgent Issues' queue tag pills
    issue_tags: List[str] = Field(
        description="A list (0–2) of short, beauty-relevant issue keywords (1–2 words each), e.g., ['allergic reaction', 'wrong shade']."
    )
    
    # For the 'Category Distribution' chart
    primary_category: str = Field(
        description="Top-level beauty category, e.g., 'Makeup', 'Skincare', 'Haircare', 'Fragrance', 'Beauty Tools', or 'other'."
    )