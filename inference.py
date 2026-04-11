import os
from openai import OpenAI
from env import RescueNetEnv
from models import Action
from grader import grade_environment


# -----------------------------
# OPENAI SETUP (OPTIONAL)
# -----------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

client = None
if API_KEY:
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY
    )


# -----------------------------
# SAFE POLICY
# -----------------------------
def fallback_policy(obs):
    max_severity = -1
    target_region = 0

    for i, region in enumerate(obs.regions):
        if region.severity > max_severity:
            max_severity = region.severity
            target_region = i

    return Action(
        region_id=target_region,
        resource_type="medical",
        quantity=0.5
    )


# -----------------------------
# CHOOSE ACTION
# -----------------------------
def choose_action(obs):

    # Try LLM (optional, safe)
    if client:
        try:
            region_info = [
                f"(id={i}, severity={r.severity:.2f}, alive={r.alive:.1f})"
                for i, r in enumerate(obs.regions)
            ]

            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{
                    "role": "user",
                    "content": f"""
You are managing disaster response.

Regions:
{region_info}

Choose best action.

Return JSON:
{{"region_id": 0, "resource_type": "medical", "quantity": 0.5}}
"""
                }],
                temperature=0
            )

            import json
            parsed = json.loads(response.choices[0].message.content)

            return Action(
                region_id=int(parsed["region_id"]),
                resource_type=str(parsed["resource_type"]),
                quantity=float(parsed["quantity"])
            )

        except Exception:
            pass  # fallback will handle

    # -----------------------------
    # FALLBACK (CRITICAL)
    # -----------------------------
    return fallback_policy(obs)


# -----------------------------
# RUN SINGLE TASK
# -----------------------------
def run_single_task(task_name):
    mode = task_name.split("_")[0]

    env = RescueNetEnv(mode=mode)

    obs = env.reset()

    rewards = []
    step = 0

    print(f"[START] task={task_name} env=rescuenet-env model={MODEL_NAME}")

    try:
        for step in range(1, env.max_steps + 1):
            action = choose_action(obs)

            obs, reward, done, _ = env.step(action)

            rewards.append(round(reward, 2))

            action_str = f"{action.resource_type}(r{action.region_id},{action.quantity:.2f})"

            print(
                f"[STEP] step={step} action={action_str} reward={reward:.2f} done={str(done).lower()} error=null"
            )

            if done:
                break

    except Exception as e:
        print(f"[STEP] step={step} action=none reward=0.00 done=true error={str(e)}")

    finally:
        result = grade_environment(env)

        score = max(0.01, min(result["score"], 0.99))
        success = result["success"]

        try:
            env.close()
        except:
            pass

        print(
            f"[END] success={str(success).lower()} "
            f"steps={step} "
            f"score={score:.3f} "
            f"rewards={','.join([f'{r:.2f}' for r in rewards])}"
        )


# -----------------------------
# RUN ALL TASKS
# -----------------------------
def run():
    tasks = ["easy_1", "medium_1", "hard_1"]

    for task in tasks:
        run_single_task(task)


# -----------------------------
# ENTRY POINT
# -----------------------------
if __name__ == "__main__":
    run()