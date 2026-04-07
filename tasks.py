from env import SocioSyncEnv
from models import Action


def easy():
    env = SocioSyncEnv(mode="easy", seed=42)
    obs = env.reset()

    total_reward = 0

    for _ in range(20):
        action = Action("hiring_policy", 0.5)
        obs, reward, done, _ = env.step(action)
        total_reward += reward

        if done:
            break

    return {
        "success": True,
        "score": float(total_reward / 10)
    }


def medium():
    env = SocioSyncEnv(mode="medium", seed=42)
    obs = env.reset()

    total_reward = 0

    for _ in range(30):
        if obs.unemployment_rate > 0.3:
            action = Action("education_policy", 0.6)
        else:
            action = Action("hiring_policy", 0.5)

        obs, reward, done, _ = env.step(action)
        total_reward += reward

        if done:
            break

    return {
        "success": True,
        "score": float(total_reward / 15)
    }


def hard():
    env = SocioSyncEnv(mode="hard", seed=42)
    obs = env.reset()

    total_reward = 0

    for _ in range(50):
        if obs.budget > 50:
            action = Action("education_policy", 0.6)
        elif obs.unemployment_rate > 0.3:
            action = Action("hiring_policy", 0.6)
        else:
            action = Action("economic_policy", 0.4)

        obs, reward, done, _ = env.step(action)
        total_reward += reward

        if done:
            break

    return {
        "success": True,
        "score": float(total_reward / 20)
    }