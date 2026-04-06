import os
from openai import OpenAI
from env import SocioSyncEnv
from models import Action
from grader import grade_environment

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

USE_LLM = HF_TOKEN is not None

if not USE_LLM:
    print("[WARNING] HF_TOKEN missing, using fallback policy")

client = None
if USE_LLM:
    try:
        client = OpenAI(
            base_url=API_BASE_URL,
            api_key=HF_TOKEN
        )
    except Exception:
        USE_LLM = False


def choose_action(obs):
    if USE_LLM:
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
            text = response.choices[0].message.content
            parsed = json.loads(text)

            return Action(
                action_type=parsed["action_type"],
                intensity=float(parsed["intensity"])
            )

        except Exception:
            pass  

    if obs.unemployment_rate > 0.3:
        return Action(action_type="education_policy", intensity=0.7)
    else:
        return Action(action_type="hiring_policy", intensity=0.5)


def run():
    if LOCAL_IMAGE_NAME:
        try:
            env = SocioSyncEnv.from_docker_image(LOCAL_IMAGE_NAME)
        except:
            env = SocioSyncEnv()
    else:
        env = SocioSyncEnv()

    obs = env.reset()

    rewards = []
    success = True
    step = 0

    print(f"[START] task=hard env=sociosync-env model={MODEL_NAME}")

    try:
        for step in range(1, 51):
            action = choose_action(obs)

            obs, reward, done, _ = env.step(action)

            rewards.append(round(reward, 2))

            action_str = f"{action.action_type}({action.intensity:.2f})"

            error_msg = "null"

            print(
                f"[STEP] step={step} action={action_str} reward={reward:.2f} done={str(done).lower()} error={error_msg}"
            )

            if done:
                break

    except Exception as e:
        success = False
        error_msg = str(e)

        print(
            f"[STEP] step={step} action=none reward=0.00 done=true error={error_msg}"
        )
        
    finally:
        score = grade_environment(env)
        score = max(0, min(score, 1))

        success = score >= 0.5

        try:
            env.close()
        except:
            pass

        print(
            f"[END] success={str(success).lower()} "
            f"steps={step} "
            f"score={score:.2f} "
            f"rewards={','.join([f'{r:.2f}' for r in rewards])}"
        )


if __name__ == "__main__":
    run()