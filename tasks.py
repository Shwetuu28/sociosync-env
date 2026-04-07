from env import SocioSyncEnv
from models import Action


# 🔥 Common runner for all tasks
def run_task(mode):
    env = SocioSyncEnv(mode=mode, seed=42)
    obs = env.reset()

    total_reward = 0

    for _ in range(50):
        # Simple stable policy (deterministic for validator)
        if obs.unemployment_rate > 0.3:
            action = Action("education_policy", 0.6)
        elif obs.budget > 50:
            action = Action("economic_policy", 0.5)
        else:
            action = Action("hiring_policy", 0.5)

        obs, reward, done, _ = env.step(action)
        total_reward += reward

        if done:
            break

    # ✅ REQUIRED: binary-like scoring (NOT continuous)
    score = 0.99 if total_reward > 0 else 0.01

    return {
        "success": score > 0.5,
        "score": score
    }


# 🟢 EASY TASKS
def easy_1():
    return run_task("easy")

def easy_2():
    return run_task("easy")

def easy_3():
    return run_task("easy")


# 🟡 MEDIUM TASKS
def medium_1():
    return run_task("medium")

def medium_2():
    return run_task("medium")

def medium_3():
    return run_task("medium")


# 🔴 HARD TASKS
def hard_1():
    return run_task("hard")

def hard_2():
    return run_task("hard")

def hard_3():
    return run_task("hard")