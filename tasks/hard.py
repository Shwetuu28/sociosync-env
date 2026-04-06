from env import SocioSyncEnv
from models import Action
from grader import grade_environment


def run_hard_task():
    env = SocioSyncEnv()
    obs = env.reset()

    for step in range(50):
        if env.rng.random() < 0.1:

            env.state_data.open_jobs = int(env.state_data.open_jobs * 0.7)
            env.state_data.economic_growth -= 0.05

        if step == 35:
            env.state_data.budget -= 30

        if obs.budget > 50:
            action = Action(
                action_type="education_policy",
                intensity=0.6
            )
        elif obs.unemployment_rate > 0.25:
            action = Action(
                action_type="hiring_policy",
                intensity=0.6
            )
        else:
            action = Action(
                action_type="economic_policy",
                intensity=0.4
            )

        obs, reward, done, _ = env.step(action)

        if done:
            break

    score = grade_environment(env)
    return score