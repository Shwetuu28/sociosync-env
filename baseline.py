from env import SocioSyncEnv
from models import Action


def run_baseline():
    env = SocioSyncEnv()
    obs = env.reset()

    total_reward = 0

    avg_skill = (obs.low_skill + obs.mid_skill + obs.high_skill) / 3

    for step in range(50):

        if obs.unemployment_rate > 0.3:
            action = Action(
                action_type="education_policy",
                intensity=0.7
            )

        elif avg_skill < 0.5:
            action = Action(
                action_type="economic_policy",
                intensity=0.6
            )

        else:
            action = Action(
                action_type="hiring_policy",
                intensity=0.5
            )

        obs, reward, done, _ = env.step(action)
        total_reward += reward

        print(f"Step {step}:")
        print(f"  Skill: {obs.avg_skill:.3f}")
        print(f"  Unemployment: {obs.unemployment_rate:.3f}")
        print(f"  Budget: {obs.budget:.2f}")
        print(f"  Reward: {reward:.3f}")
        print("----------")

        if done:
            break

    print("\nFinal State:")
    print(f"Skill: {obs.avg_skill:.3f}")
    print(f"Unemployment: {obs.unemployment_rate:.3f}")
    print(f"Total Reward: {total_reward:.3f}")

    return total_reward


if __name__ == "__main__":
    run_baseline()