
import math
import random
from models import Observation, Region


# ---------------------------------------------------------------------------
# ADVERSARIAL NOISE HELPERS (issue #2 fix)
# ---------------------------------------------------------------------------

def _inject_noise(regions, rng, mode):
    """
    Hard task only: corrupt the observation a LLM agent receives so that
    greedy-by-severity no longer trivially wins.

      - One region gets an inflated severity report (sensor malfunction).
        Its *displayed* severity is boosted by 0.25–0.4, but the underlying
        dynamics are unchanged — so dispatching there is actively wasteful.
      - One region emits a phantom resource_need spike (+2.0 on a random
        resource type) signalling urgency that isn't real.

    Easy/Medium: no noise added.
    """
    if mode != "hard" or len(regions) < 3:
        return regions

    # Pick two distinct victim indices
    victims = rng.sample(range(len(regions)), 2)

    # Inflated severity (display only — stored in a separate field)
    r0 = regions[victims[0]]
    r0["reported_severity"] = round(min(1.0, r0["severity"] + rng.uniform(0.25, 0.40)), 2)
    r0["sensor_note"] = "⚠ sensor spike — cross-check with field report"

    # Phantom resource demand
    r1 = regions[victims[1]]
    phantom_idx = rng.randint(0, 2)
    r1["resource_need"] = list(r1["resource_need"])
    r1["resource_need"][phantom_idx] = round(r1["resource_need"][phantom_idx] + 2.0, 1)
    r1["phantom_note"] = "📡 unverified surge — comms may be corrupted"

    return regions


# ---------------------------------------------------------------------------
# ENVIRONMENT
# ---------------------------------------------------------------------------

class RescueNetEnv:

    RESOURCE_TYPES = ["food", "medical", "rescue"]

    def __init__(self, mode="easy"):
        self.mode = mode
        self.severity_multiplier = 1.0

        self.max_steps = 20
        self.current_step = 0

        self.num_regions = {"easy": 5, "medium": 7, "hard": 10}.get(mode, 5)

        # Will be properly initialised in reset()
        self.regions = []
        self.available_resources = []
        self.total_population = 0
        self.total_survived = 0
        self.used_resources = 0
        self.total_resources = 0
        self.tasks_completed = 0
        self.total_cost = 0
        self._rng = random.Random()

        self.reset()

    # -----------------------------------------------------------------------
    # RESET  (issue #7 fix: always seed-aware)
    # -----------------------------------------------------------------------

    def reset(self, seed=None):
        """
        Reset the environment.  Pass seed=<int> for fully reproducible episodes.
        Baseline inference uses seed=42 so scores are deterministic.
        """
        if seed is not None:
            self._rng.seed(seed)
            random.seed(seed)

        self.current_step = 0
        self.regions = []
        self.total_population = 0
        self.total_survived = 0
        self.used_resources = 0
        self.tasks_completed = 0
        self.total_cost = 0

        for _ in range(self.num_regions):
            population = self._rng.randint(50, 200)
            severity = round(
                self._rng.uniform(0.1, 1.0) * self.severity_multiplier, 2
            )
            self.regions.append({
                "population": population,
                "severity": severity,
                "reported_severity": severity,   # may be corrupted on hard
                "delay": 1,
                "resource_need": [1.0, 1.0, 1.0],
                "alive": float(population),
                "sensor_note": None,
                "phantom_note": None,
            })
            self.total_population += population

        # Hard task gets scarce resources (unchanged from original)
        base = 6.0 if self.mode == "hard" else 10.0
        self.available_resources = [base, base, base]
        self.total_resources = sum(self.available_resources)

        # Inject adversarial noise for hard task (issue #2 fix)
        self.regions = _inject_noise(self.regions, self._rng, self.mode)

        return self._get_obs()

    # -----------------------------------------------------------------------
    # STEP  (issue #1 fix: richer per-step reward signal)
    # -----------------------------------------------------------------------

    def step(self, action):
        self.current_step += 1

        region_id = action.region_id
        resource_map = {"food": 0, "medical": 1, "rescue": 2}

        if region_id < 0 or region_id >= len(self.regions):
            obs = self._get_obs()
            return obs, -0.1, self.current_step >= self.max_steps, {"error": "invalid_region"}

        resource_idx = resource_map.get(action.resource_type, 1)
        quantity = max(0.0, min(action.quantity, 2.0))

        region = self.regions[region_id]

        # ---- apply dispatch ----
        if self.available_resources[resource_idx] >= quantity and quantity > 0:
            self.available_resources[resource_idx] -= quantity
            self.used_resources += quantity
            region["delay"] = max(1, region["delay"] - 1)
            self.tasks_completed += 1
        elif quantity > self.available_resources[resource_idx]:
            # Trying to dispatch more than available → penalty
            self.total_cost += 0.05

        # ---- survival model (unchanged — exponential decay) ----
        prev_total_alive = sum(r["alive"] for r in self.regions)

        # Identify the highest TRUE severity region (not the reported/noisy one)
        max_true_sev = max(r["severity"] for r in self.regions)

        # After the survival loop, increment all delays EXCEPT the dispatched region
        for i, r in enumerate(self.regions):
            survived = r["alive"] * math.exp(-r["severity"] * r["delay"])
            r["alive"] = max(0.0, survived)
            if i == region_id:
                r["delay"] =  0 
            else:
                r["delay"] += 1

        new_total_alive = sum(r["alive"] for r in self.regions)
        self.total_survived = new_total_alive
        # ---- reward shaping (issue #1 fix) ----
        # Primary: normalised population change this step
        delta = (new_total_alive - prev_total_alive) / (self.total_population + 1e-6)

        # Triage bonus: extra +0.05 if agent dispatched to the region with
        # highest TRUE severity (rewards correct prioritisation)
        dispatched_region_sev = self.regions[region_id]["severity"]
        triage_bonus = 0.05 if dispatched_region_sev >= max_true_sev - 0.05 else 0.0

        # Idle-resource penalty (unchanged)
        unused_resources = sum(self.available_resources)
        idle_penalty = 0.05 * unused_resources / (self.total_resources + 1e-6)

        # Delay-accumulation penalty: discourage repeatedly ignoring a region
        max_delay = max(r["delay"] for r in self.regions)
        delay_penalty = 0.02 * min(max_delay, 10) / 10.0

        reward = delta + triage_bonus - idle_penalty - delay_penalty
        reward = max(-1.0, min(reward, 1.0))

        done = self.current_step >= self.max_steps
        return self._get_obs(), reward, done, {}

    # -----------------------------------------------------------------------
    # STATE  (issue #6 fix: fully serialisable, required by openenv validate)
    # -----------------------------------------------------------------------

    def state(self):
        """Returns full current state as a plain dict (OpenEnv spec requirement)."""
        return {
            "mode": self.mode,
            "current_step": self.current_step,
            "max_steps": self.max_steps,
            "total_population": self.total_population,
            "total_survived": round(self.total_survived, 2),
            "used_resources": round(self.used_resources, 2),
            "total_resources": self.total_resources,
            "tasks_completed": self.tasks_completed,
            "total_cost": round(self.total_cost, 4),
            "available_resources": [round(r, 2) for r in self.available_resources],
            "regions": [
                {
                    "population": r["population"],
                    "severity": r["severity"],
                    "reported_severity": r["reported_severity"],
                    "delay": r["delay"],
                    "alive": round(r["alive"], 2),
                    "resource_need": r["resource_need"],
                    "sensor_note": r["sensor_note"],
                    "phantom_note": r["phantom_note"],
                }
                for r in self.regions
            ],
        }

    # -----------------------------------------------------------------------
    # OBSERVATION
    # -----------------------------------------------------------------------

    def _get_obs(self):
        region_objs = []
        for r in self.regions:
            region_objs.append(
                Region(
                    population=r["population"],
                    # Agents see reported_severity (may be noisy on hard task)
                    severity=r["reported_severity"],
                    delay=r["delay"],
                    resource_need=r["resource_need"],
                    alive=round(r["alive"], 2),
                    sensor_note=r.get("sensor_note"),
                    phantom_note=r.get("phantom_note"),
                )
            )
        return Observation(
            regions=region_objs,
            available_resources=[round(x, 2) for x in self.available_resources],
            time_step=self.current_step,
        )

    # -----------------------------------------------------------------------
    # CLOSE
    # -----------------------------------------------------------------------

    def close(self):
        pass