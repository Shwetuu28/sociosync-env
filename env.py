from models import Observation, Action
from typing import List
import random


class SocioSyncEnv:
    def __init__(self,seed=42, mode="easy"):
        self.seed = seed
        self.rng = random.Random(seed)
        self.mode = mode
        self.current_step = 0
        self.max_steps = 50
        self.learning_queue = []  # 🔥 delayed effects

        self.state_data = None

    def reset(self) -> Observation:
        if self.mode == "easy":
            unemployment = self.rng.uniform(0.2, 0.3)
            budget = 120

        elif self.mode == "medium":
            unemployment = self.rng.uniform(0.3, 0.5)
            budget = 100

        elif self.mode == "hard":
            unemployment = self.rng.uniform(0.4, 0.6)
            budget = 80

        self.state_data = Observation(
            low_skill=self.rng.uniform(0.4, 0.6),
            mid_skill=self.rng.uniform(0.2, 0.4),
            high_skill=self.rng.uniform(0.1, 0.3),
            unemployment_rate=unemployment,
            learning_efficiency=0.5,
            project_exposure=0.1,
            open_jobs=50,
            economic_growth=0.2,
            inequality=0.5,
            budget=budget
        )

        return self.state_data

    def state(self) -> Observation:
        obs = self.state_data

        return Observation(
            low_skill=obs.low_skill,
            mid_skill=obs.mid_skill,
            high_skill=obs.high_skill,

            unemployment_rate=max(0, min(1, obs.unemployment_rate + self.rng.uniform(-0.02, 0.02))),
            learning_efficiency=obs.learning_efficiency,
            project_exposure=obs.project_exposure,
            open_jobs=obs.open_jobs,

            economic_growth=obs.economic_growth,
            inequality=max(0, min(1, obs.inequality + self.rng.uniform(-0.02, 0.02))),
            budget=obs.budget
        )
    
    def step(self, action: Action):
        self.current_step += 1

        new_queue = []
        for item in self.learning_queue:
            item["delay"] -= 1
            if item["delay"] <= 0:
                # 🎓 Education pipeline with diminishing returns
                flow_low_to_mid = 0.2 * self.state_data.low_skill * item["effect"]
                flow_mid_to_high = 0.1 * self.state_data.mid_skill * item["effect"]

                self.state_data.low_skill += 0.5 * item["effect"] * (1 - self.state_data.low_skill)
                self.state_data.mid_skill += 0.3 * item["effect"] * (1 - self.state_data.mid_skill)
                self.state_data.high_skill += 0.2 * item["effect"] * (1 - self.state_data.high_skill)

                # pipeline flow
                self.state_data.low_skill -= flow_low_to_mid
                self.state_data.mid_skill += flow_low_to_mid

                self.state_data.mid_skill -= flow_mid_to_high
                self.state_data.high_skill += flow_mid_to_high
            else:
                new_queue.append(item)
        self.learning_queue = new_queue

        self.state_data.low_skill = max(0, min(1, self.state_data.low_skill))
        self.state_data.mid_skill = max(0, min(1, self.state_data.mid_skill))
        self.state_data.high_skill = max(0, min(1, self.state_data.high_skill))

        if action.action_type == "education_policy":
            skill_gain = 0.05 * action.intensity * self.state_data.learning_efficiency

            self.learning_queue.append({
                "effect": skill_gain,
                "delay": self.rng.choice([2, 3, 4, 5])
            })

            self.state_data.project_exposure += 0.1 * action.intensity
            self.state_data.budget -= 10 * action.intensity

        elif action.action_type == "hiring_policy":
            self.state_data.open_jobs += int(20 * action.intensity)
            self.state_data.budget -= 15 * action.intensity

        elif action.action_type == "economic_policy":
            self.state_data.learning_efficiency += 0.02 * action.intensity
            self.state_data.budget -= 20 * action.intensity

        demand_low = 0.4 + self.rng.uniform(-0.05, 0.05)
        demand_mid = 0.35 + self.rng.uniform(-0.05, 0.05)
        demand_high = 0.25 + self.rng.uniform(-0.05, 0.05)

        total_demand = demand_low + demand_mid + demand_high
        demand_low /= total_demand
        demand_mid /= total_demand
        demand_high /= total_demand

        if self.state_data.budget < 30:
            self.state_data.learning_efficiency *= 0.97


        demand_low = 0.4
        demand_mid = 0.35
        demand_high = 0.25

        alignment = (
            abs(self.state_data.low_skill - demand_low) +
            abs(self.state_data.mid_skill - demand_mid) +
            abs(self.state_data.high_skill - demand_high)
        )

        alignment_score = max(0, 1 - alignment)

        employment_gain = min(
            alignment_score,
            self.state_data.open_jobs / 100
        )

        self.state_data.unemployment_rate -= 0.05 * employment_gain
        self.state_data.unemployment_rate = max(0, min(1, self.state_data.unemployment_rate))
        self.state_data.inequality = max(0, min(1, self.state_data.inequality))

        self.state_data.economic_growth += 0.02 * employment_gain
        self.state_data.inequality += 0.01 * (1 - employment_gain)

        reward = self.compute_reward()

        done = False

        if self.current_step >= self.max_steps:
            done = True

        if self.state_data.budget < -50:
            reward -= 1
            done = True

        if self.state_data.unemployment_rate > 0.8:
            reward -= 1
            done = True

        job_noise = self.rng.uniform(-2, 2)
        self.state_data.open_jobs = max(0, int(self.state_data.open_jobs + job_noise))

        return self.state_data, reward, done, {}  
        
    
    def compute_reward(self):
        obs = self.state_data

        avg_skill = (
            obs.low_skill +
            obs.mid_skill +
            obs.high_skill
        ) / 3

        demand_low = 0.4
        demand_mid = 0.35
        demand_high = 0.25

        alignment = (
            abs(obs.low_skill - demand_low) +
            abs(obs.mid_skill - demand_mid) +
            abs(obs.high_skill - demand_high)
        )

        alignment_score = max(0, 1 - alignment)

        employment_score = (1 - obs.unemployment_rate) ** 1.5
        skill_score = avg_skill ** 1.2

        reward = (
            0.5 * employment_score +
            0.2 * skill_score +
            0.2 * alignment_score +
            0.1 * obs.economic_growth -
            0.2 * obs.inequality
        )

        if obs.budget < 0:
            reward -= 0.2 + 0.01 * abs(obs.budget)

        if obs.unemployment_rate > 0.6:
            reward -= 0.1

        if hasattr(self, "prev_unemployment"):
            change = abs(obs.unemployment_rate - self.prev_unemployment)
            reward -= 0.05 * change  # penalize volatility

        self.prev_unemployment = obs.unemployment_rate

        return reward
    
    def close(self):
        pass
        
