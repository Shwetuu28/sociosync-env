from env import SocioSyncEnv
import env
from models import Action


def run_easy_task():
    env = SocioSyncEnv()
    obs = env.reset()

    env.state_data.open_jobs = 80
    env.state_data.budget = 9999

    for _ in range(50):
        action = Action(
            action_type="hiring_policy",
            intensity=0.7
        )

        obs, reward, done, _ = env.step(action)

        if done:
            break

    score = 1 - obs.unemployment_rate

    return min(score, 1.0)