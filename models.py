# models.py

from typing import List


# -----------------------------
# REGION
# -----------------------------
class Region:
    def __init__(self, population, severity, delay, resource_need, alive):
        self.population = population
        self.severity = severity
        self.delay = delay
        self.resource_need = resource_need
        self.alive = alive


# -----------------------------
# OBSERVATION
# -----------------------------
class Observation:
    def __init__(self, regions, available_resources, time_step):
        self.regions: List[Region] = regions
        self.available_resources = available_resources
        self.time_step = time_step


# -----------------------------
# ACTION
# -----------------------------
class Action:
    def __init__(self, region_id, resource_type, quantity):
        self.region_id = region_id
        self.resource_type = resource_type
        self.quantity = quantity