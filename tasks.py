from env import SocioSyncEnv
from models import Action
from grader import grade_environment
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def run_task(mode):
    env = SocioSyncEnv(mode=mode, seed=42)
    obs = env.reset()

    for _ in range(30):
        action = Action(
            action_type="hiring_policy",
            intensity=0.5
        )
        obs, _, done, _ = env.step(action)

        if done:
            break

    return grade_environment(env)


def easy():
    return run_task("easy")


def medium():
    return run_task("medium")


def hard():
    return run_task("hard")