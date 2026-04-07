import asyncio
import os
from openai import OpenAI
from env import SocioSyncEnv
from models import Action
from grader import grade_environment

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
API_KEY = os.getenv("HF_TOKEN")
IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

if not API_KEY:
    raise ValueError("HF_TOKEN is required")

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY
)


def choose_action(obs):
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{
                "role": "user",
                "content": f"""
State:
unemployment: {obs.unemployment_rate}
budget: {obs.budget}

Choose action: education_policy, hiring_policy, economic_policy
Return JSON: {{"action_type": "...", "intensity": 0.5}}
"""
            }],
            temperature=0
        )

        import json
        parsed = json.loads(response.choices[0].message.content)

        return Action(
            action_type=parsed["action_type"],
            intensity=float(parsed["intensity"])
        )

    except Exception:
        return Action("hiring_policy", 0.5)


async def run_task(task_name):
    env = await SocioSyncEnv.from_docker_image(IMAGE_NAME)

    result = await env.reset()

    rewards = []
    steps = 0

    print(f"[START] task={task_name} env=sociosync-env model={MODEL_NAME}")

    try:
        for step in range(1, 51):
            action = choose_action(result.observation)

            result = await env.step(action)

            reward = result.reward or 0.0
            done = result.done

            rewards.append(reward)
            steps = step

            action_str = f"{action.action_type}({action.intensity:.2f})"

            print(
                f"[STEP] step={step} action={action_str} reward={reward:.2f} done={str(done).lower()} error=null"
            )

            if done:
                break

    except Exception as e:
        print(f"[STEP] step={steps} action=none reward=0.00 done=true error={str(e)}")

    finally:
        score = min(max(sum(rewards) / (len(rewards) + 1e-6), 0.0), 1.0)
        success = score > 0.3

        await env.close()

        print(
            f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={','.join([f'{r:.2f}' for r in rewards])}"
        )


async def main():
    for task in ["easy", "medium", "hard"]:
        await run_task(task)


if __name__ == "__main__":
    asyncio.run(main())