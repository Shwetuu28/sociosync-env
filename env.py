import math
import random
from models import Observation, Region

class RescueNetEnv:
    def __init__(self, mode="easy"):
        self.mode = mode

        self.severity_multiplier = 1.0

        # CONFIG
        self.max_steps = 20
        self.current_step = 0

        self.num_regions = {
            "easy": 5,
            "medium": 7,
            "hard": 10
        }.get(mode, 5)

        self.reset()

    # -----------------------------
    # RESET
    # -----------------------------
    def reset(self, seed=None):
        if seed is not None:
            random.seed(seed)

        self.current_step = 0

        self.regions = []
        self.total_population = 0
        self.total_survived = 0

        for _ in range(self.num_regions):
            population = random.randint(50, 200)
            severity = round(random.uniform(0.1, 1.0) * self.severity_multiplier, 2)

            self.regions.append({
                "population": population,
                "severity": severity,
                "delay": 1,
                "resource_need": [1.0, 1.0, 1.0],  # food, medical, rescue
                "alive": population
            })

            self.total_population += population

        self.available_resources = [10.0, 10.0, 10.0]  # [food, medical, rescue]
        self.used_resources = 0
        self.total_resources = sum(self.available_resources)

        self.tasks_completed = 0
        self.total_cost = 0

        return self._get_obs()

    # -----------------------------
    # STEP
    # -----------------------------
    def step(self, action):
        self.current_step += 1

        region_id = action.region_id
        resource_map = {"food": 0, "medical": 1, "rescue": 2}

        if region_id >= len(self.regions):
            return self._get_obs(), -0.1, False, {"error": "invalid_region"}

        resource_idx = resource_map.get(action.resource_type, 0)
        quantity = max(0.0, min(action.quantity, 2.0))

        region = self.regions[region_id]

        # Apply resource if available
        if self.available_resources[resource_idx] >= quantity:
            self.available_resources[resource_idx] -= quantity
            self.used_resources += quantity
            region["delay"] = max(1, region["delay"] - 1)
            self.tasks_completed += 1
        else:
            self.total_cost += 0.05  # penalty for invalid allocation

        # -----------------------------
        # SURVIVAL MODEL
        # -----------------------------
        step_survival = 0

        for r in self.regions:
            S = r["severity"]
            D = r["delay"]
            P = r["alive"]

            survival_prob = math.exp(-S * D)
            survived = P * survival_prob

            # deaths this step
            deaths = P - survived

            r["alive"] = survived
            r["delay"] += 1

            step_survival += survived

        # -----------------------------
        # REWARD
        # -----------------------------
        delta_survival = step_survival - self.total_survived
        self.total_survived = step_survival

        unused_resources = sum(self.available_resources)

        reward = (
            delta_survival / self.total_population
            - 0.05 * unused_resources / (self.total_resources + 1e-6)
        )

        # clamp reward (safe for validator)
        reward = max(-1.0, min(reward, 1.0))

        # -----------------------------
        # DONE CONDITION
        # -----------------------------
        done = self.current_step >= self.max_steps

        return self._get_obs(), reward, done, {}

    # -----------------------------
    # STATE (REQUIRED)
    # -----------------------------
    def state(self):
        return {
            "regions": self.regions,
            "resources": self.available_resources,
            "step": self.current_step
        }

    # -----------------------------
    # OBSERVATION
    # -----------------------------
    def _get_obs(self):
        region_objs = []

        for r in self.regions:
            region_objs.append(
                Region(
                    population=r["population"],
                    severity=r["severity"],
                    delay=r["delay"],
                    resource_need=r["resource_need"],
                    alive=r["alive"]
                )
            )

        return Observation(
            regions=region_objs,
            available_resources=self.available_resources,
            time_step=self.current_step
        )

    # -----------------------------
    # CLOSE (SAFE)
    # -----------------------------
    def close(self):
        pass
