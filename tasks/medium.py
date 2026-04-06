from env import SocioSyncEnv
from models import Action


def run_medium_task():
    env = SocioSyncEnv()
    obs = env.reset()

    for _ in range(50):
        if obs.unemployment_rate > 0.3:
            action = Action(
                action_type="education_policy",
                intensity=0.7
            )
        else:
            action = Action(
                action_type="hiring_policy",
                intensity=0.5
            )

        obs, reward, done, _ = env.step(action)

        if done:
            break

    employment = 1 - obs.unemployment_rate
    skill_alignment = (
        obs.low_skill +
        obs.mid_skill +
        obs.high_skill
    ) / 3

    score = 0.6 * employment + 0.4 * skill_alignment

    return min(score, 1.0)