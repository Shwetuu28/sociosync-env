from typing import Literal


class Observation:
    def __init__(
        self,
        low_skill: float,
        mid_skill: float,
        high_skill: float,
        unemployment_rate: float,
        learning_efficiency: float,
        project_exposure: float,
        open_jobs: int,
        economic_growth: float,
        inequality: float,
        budget: float
    ):
        self.low_skill = low_skill
        self.mid_skill = mid_skill
        self.high_skill = high_skill
        self.unemployment_rate = unemployment_rate
        self.learning_efficiency = learning_efficiency
        self.project_exposure = project_exposure
        self.open_jobs = open_jobs
        self.economic_growth = economic_growth
        self.inequality = inequality
        self.budget = budget


class Action:
    def __init__(
        self,
        action_type: Literal[
            "education_policy",
            "hiring_policy",
            "economic_policy"
        ],
        intensity: float
    ):
        self.action_type = action_type
        self.intensity = intensity