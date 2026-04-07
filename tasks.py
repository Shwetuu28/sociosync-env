import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from env import SocioSyncEnv
from models import Action
from grader import grade_environment

__all__ = ["easy", "medium", "hard"]


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

    result = grade_environment(env)

    # 🔥 STRICT FORMAT (VERY IMPORTANT)
    return {
        "success": bool(result["success"]),
        "score": float(result["score"])
    }


def easy():
    return run_task("easy")


def medium():
    return run_task("medium")


def hard():
    return run_task("hard")