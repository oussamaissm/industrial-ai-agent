from pydantic import BaseModel, Field, field_validator
from typing import Optional

class ScheduleProposal(BaseModel):
    order_id: str = Field(description="The ID of the order being scheduled")
    machine_id: str = Field(description="The ID of the assigned machine")
    estimated_hours: float = Field(description="Estimated processing time")
    reasoning: str = Field(description="Brief justification based on retrieved data")

    @field_validator('machine_id')
    @classmethod
    def validate_machine_format(cls, v):
        if not v.startswith("M"):
            raise ValueError("Machine ID must start with 'M'")
        return v

class AgentResponse(BaseModel):
    answer: str
    structured_data: Optional[ScheduleProposal] = None
    confidence: float = Field(ge=0.0, le=1.0, description="Model's self-assessed confidence")
