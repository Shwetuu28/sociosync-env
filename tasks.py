from env import SocioSyncEnv
from models import Action
from grader import grade_environment


def easy():
    env = SocioSyncEnv(mode="easy", seed=42)
    obs = env.reset()

    for _ in range(20):
        action = Action(action_type="hiring_policy", intensity=0.5)
        obs, _, done, _ = env.step(action)
        if done:
            break

    return grade_environment(env)


def medium():
    env = SocioSyncEnv(mode="medium", seed=42)
    obs = env.reset()

    for _ in range(50):
        if obs.unemployment_rate > 0.3:
            action = Action("education_policy", 0.7)
        else:
            action = Action("hiring_policy", 0.5)

        obs, _, done, _ = env.step(action)
        if done:
            break

    return grade_environment(env)


def hard():
    env = SocioSyncEnv(mode="hard", seed=42)
    obs = env.reset()

    for step in range(50):
        if obs.budget > 50:
            action = Action("education_policy", 0.6)
        elif obs.unemployment_rate > 0.25:
            action = Action("hiring_policy", 0.6)
        else:
            action = Action("economic_policy", 0.4)

        obs, _, done, _ = env.step(action)
        if done:
            break

    return grade_environment(env)