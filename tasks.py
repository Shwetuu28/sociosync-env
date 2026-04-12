
from env import RescueNetEnv

BASELINE_SEED = 42   # change to None for random episodes during training


def easy_1(seed=BASELINE_SEED):
    env = RescueNetEnv(mode="easy")
    env.max_steps = 15
    env.severity_multiplier = 0.7
    env.reset(seed=seed)   # seed applied HERE so region generation is deterministic
    return env


def medium_1(seed=BASELINE_SEED):
    env = RescueNetEnv(mode="medium")
    env.max_steps = 20
    env.severity_multiplier = 1.0
    env.reset(seed=seed)
    return env


def hard_1(seed=BASELINE_SEED):
    env = RescueNetEnv(mode="hard")
    env.max_steps = 25
    env.severity_multiplier = 1.3
    # Note: available_resources for hard mode (6.0 per type) is set inside
    # RescueNetEnv.reset() based on mode == "hard", so no override needed here.
    env.reset(seed=seed)
    return env


TASK_REGISTRY = {
    "easy_1":   easy_1,
    "medium_1": medium_1,
    "hard_1":   hard_1,
}


def get_task(name, seed=BASELINE_SEED):
    if name not in TASK_REGISTRY:
        raise ValueError(f"Unknown task '{name}'. Available: {list(TASK_REGISTRY)}")
    return TASK_REGISTRY[name](seed=seed)