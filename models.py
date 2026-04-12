
from typing import List, Optional
from pydantic import BaseModel, Field


class Region(BaseModel):
    population: int = Field(..., description="Initial population of the region")
    severity: float = Field(..., ge=0.0, le=1.0, description="Reported disaster severity (may be noisy on hard task)")
    delay: int = Field(..., ge=0, description="Steps since last resource dispatch")
    resource_need: List[float] = Field(..., description="Relative need for [food, medical, rescue]")
    alive: float = Field(..., ge=0.0, description="Current surviving population estimate")
    sensor_note: Optional[str] = Field(None, description="Non-null on hard task when sensor data may be corrupted")
    phantom_note: Optional[str] = Field(None, description="Non-null on hard task when resource demand may be spurious")


class Observation(BaseModel):
    regions: List[Region] = Field(..., description="Current state of all disaster regions")
    available_resources: List[float] = Field(..., description="Remaining [food, medical, rescue] supply")
    time_step: int = Field(..., ge=0, description="Current episode step")


class Action(BaseModel):
    region_id: int = Field(..., ge=0, description="Index of region to dispatch resources to")
    resource_type: str = Field(..., description="One of: food, medical, rescue")
    quantity: float = Field(..., ge=0.0, le=2.0, description="Amount to dispatch")


class StepResult(BaseModel):
    observation: Observation
    reward: float = Field(..., ge=-1.0, le=1.0)
    done: bool
    info: dict = Field(default_factory=dict)


class GradeResult(BaseModel):
    score: float = Field(..., ge=0.01, le=0.99)
    success: bool
    survival_rate: float
    efficiency: float
    utilization: float
    cost_penalty: float
    breakdown: str