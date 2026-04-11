# grader.py

EPS_MIN = 0.01
EPS_MAX = 0.99


def safe_div(a, b):
    return a / b if b > 0 else EPS_MIN


def clamp(x):
    return max(EPS_MIN, min(x, EPS_MAX))


def grade_environment(env):
    # -----------------------------
    # METRICS
    # -----------------------------

    survival_rate = safe_div(env.total_survived, env.total_population)

    efficiency = safe_div(env.tasks_completed, env.max_steps)

    utilization = safe_div(env.used_resources, env.total_resources)

    cost = clamp(env.total_cost)

    # -----------------------------
    # FINAL SCORE
    # -----------------------------
    score = (
        0.5 * survival_rate +
        0.2 * efficiency +
        0.2 * utilization -
        0.1 * cost
    )

    score = clamp(score)

    return {
        "score": score,
        "success": score > 0.6
    }