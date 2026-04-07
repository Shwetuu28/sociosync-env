from env import SocioSyncEnv
import env
from models import Action


def run_easy_task():
    env = SocioSyncEnv()
    obs = env.reset()

    total_reward = 0

    for _ in range(20):
        action = Action(action_type="hiring_policy", intensity=0.5)
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            break

    score = max(0, min(1, total_reward / 10))

    return {
        "success": True,
        "score": score
    }