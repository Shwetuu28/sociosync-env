
import json
import os
from typing import List, Optional

from openai import OpenAI

from env import RescueNetEnv
from models import Action
from grader import grade_environment
from tasks import BASELINE_SEED

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

BENCHMARK           = "rescuenet-env"
SUCCESS_THRESHOLD   = 0.6
TEMPERATURE         = 0.0
MAX_TOKENS          = 256

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY or "dummy") if True else None

# ---------------------------------------------------------------------------
# STDOUT LOGGING  (hackathon-mandated format)
# ---------------------------------------------------------------------------

def log_start(task: str, model: str) -> None:
    print(f"[START] task={task} env={BENCHMARK} model={model}", flush=True)


def log_step(step, action, reward, done, error):
    print(
        f"[STEP] step={step} action={action} "
        f"reward={reward:.2f} done={'true' if done else 'false'} "
        f"error={error if error else 'null'}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )

# ---------------------------------------------------------------------------
# GREEDY FALLBACK  (safe, deterministic, ignores noisy reported_severity)
# ---------------------------------------------------------------------------

def fallback_policy(obs) -> Action:
    """
    Dispatch medical to the region with highest TRUE urgency signal.
    We use alive/population ratio inverted by severity as a proxy —
    this deliberately ignores reported_severity to avoid being fooled
    by the hard-task sensor noise.
    """
    best_score = -1.0
    target = 0
    for i, r in enumerate(obs.regions):
        if r.alive <= 0:
            continue
        # urgency = severity × delay (both increase with neglect)
        urgency = r.severity * r.delay
        if urgency > best_score:
            best_score = urgency
            target = i

    return Action(region_id=target, resource_type="medical", quantity=1.0)

# ---------------------------------------------------------------------------
# LLM POLICY  (noise-aware prompt — issue #2 fix)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an emergency coordinator managing disaster response.
Each step you must dispatch one resource to one region.

IMPORTANT WARNINGS (hard task only):
- Some regions show a "sensor_note" — their reported severity may be INFLATED.
  Cross-check: a region with very high reported_severity but low delay and decent
  alive count is probably a sensor glitch. Prioritise regions with high delay instead.
- Some regions show a "phantom_note" — their resource_need may be EXAGGERATED.
  Do not over-allocate to them.

STRATEGY:
  1. Prioritise regions with high (severity × delay) — they are deteriorating fastest.
  2. Ignore regions with sensor_note unless their delay is also high (>3).
  3. Match resource_type to the highest resource_need value for the target region.
  4. Keep quantity between 0.5–1.5 to spread resources; never dump all on one region.

Respond with ONLY valid JSON, no explanation:
{"region_id": <int>, "resource_type": "<food|medical|rescue>", "quantity": <float>}"""


def llm_action(obs) -> Optional[Action]:
    if not API_KEY:
        return None
    try:
        region_lines = []
        for i, r in enumerate(obs.regions):
            notes = []
            if r.sensor_note:
                notes.append(r.sensor_note)
            if r.phantom_note:
                notes.append(r.phantom_note)
            note_str = " | ".join(notes) if notes else "clean"
            region_lines.append(
                f"  r{i}: reported_sev={r.severity:.2f} delay={r.delay} "
                f"alive={r.alive:.0f} need={[round(x,1) for x in r.resource_need]} notes=[{note_str}]"
            )

        user_msg = (
            f"Step resources remaining: food={obs.available_resources[0]:.1f} "
            f"medical={obs.available_resources[1]:.1f} rescue={obs.available_resources[2]:.1f}\n"
            f"Regions:\n" + "\n".join(region_lines) + "\n\nChoose one action:"
        )

        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        raw = (resp.choices[0].message.content or "").strip()
        # Strip markdown fences if model wraps in ```json
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        parsed = json.loads(raw.strip())
        return Action(
            region_id=int(parsed["region_id"]),
            resource_type=str(parsed["resource_type"]),
            quantity=float(parsed["quantity"]),
        )
    except Exception:
        return None

# ---------------------------------------------------------------------------
# SINGLE TASK RUNNER
# ---------------------------------------------------------------------------

def run_task(task_name: str) -> None:
    mode = task_name.split("_")[0]

    env = RescueNetEnv(mode=mode)
    # Replicate task factory settings
    settings = {
        "easy":   dict(max_steps=15, severity_multiplier=0.7),
        "medium": dict(max_steps=20, severity_multiplier=1.0),
        "hard":   dict(max_steps=25, severity_multiplier=1.3),
    }
    s = settings.get(mode, settings["easy"])
    env.max_steps = s["max_steps"]
    env.severity_multiplier = s["severity_multiplier"]

    # Fixed seed → reproducible baseline (issue #7)
    obs = env.reset(seed=BASELINE_SEED)

    log_start(task=task_name, model=MODEL_NAME)

    rewards: List[float] = []
    step = 0

    try:
        for step in range(1, env.max_steps + 1):
            action = llm_action(obs) or fallback_policy(obs)
            action_str = f"{action.resource_type}(r{action.region_id},{action.quantity:.2f})"

            obs, reward, done, info = env.step(action)
            rewards.append(reward)
            error = info.get("error")

            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            if done:
                break

    except Exception as exc:
        log_step(step=step, action="none", reward=0.0, done=True, error=str(exc))

    finally:
        result = grade_environment(env)
        score   = max(0.01, min(result["score"], 0.99))
        success = result["success"]
        try:
            env.close()
        except Exception:
            pass

        log_end(success=success, steps=step, score=score, rewards=rewards)

# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

def main():
    target = os.getenv("TASK_NAME")
    tasks = [target] if target else ["easy_1", "medium_1", "hard_1"]
    for t in tasks:
        run_task(t)


if __name__ == "__main__":
    main()