from pydantic import BaseModel, Field
from typing import Dict

class ReviewAnalysis(BaseModel):
    """
    Defines the structured data contract for the LLM's analysis output.
    """
    sentiment: str = Field(description="The overall sentiment: 'Positive', 'Negative', or 'Neutral'")
    main_topic: str = Field(description="The single most fitting category from ['Durability', 'Performance', 'Shipping', 'Price', 'Features', 'Usability', 'Customer Service', 'Other']")
    
    # --- THIS IS THE UPDATED LINE ---
    key_drivers: Dict[str, str] = Field(description="A dictionary where keys are specific features mentioned and values are their corresponding sentiment ('Positive', 'Negative', or 'Neutral'). For example: {'Battery Life': 'Negative', 'Screen Quality': 'Positive'}.")
    
    is_actionable: bool = Field(description="A boolean (true or false). Set to true only if the review contains specific, concrete feedback.")
    summary: str = Field(description="A concise, single-sentence summary of the review's main point.")