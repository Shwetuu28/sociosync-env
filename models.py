from pydantic import BaseModel
from typing import List, Literal


class Observation(BaseModel):
    low_skill: float
    mid_skill: float
    high_skill: float                  
    unemployment_rate: float          

    learning_efficiency: float        
    project_exposure: float           

    open_jobs: int

    economic_growth: float
    inequality: float                 

    budget: float


class Action(BaseModel):
    action_type: Literal[
        "education_policy",
        "hiring_policy",
        "economic_policy"
    ]
    intensity: float  