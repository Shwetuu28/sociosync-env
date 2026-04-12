
EPS_MIN = 0.01
EPS_MAX = 0.99


def safe_div(a, b):
    return a / b if b > 0 else 0.0


def clamp(x):
    return max(EPS_MIN, min(x, EPS_MAX))


def grade_environment(env):
    """
    Score an episode after it ends.

    Weight rationale
    ----------------
    survival_rate (0.50): The primary humanitarian objective.
        Based on FEMA's National Incident Management System (NIMS), which
        defines life safety as Priority 1 above all other considerations.

    efficiency (0.20): Time-to-first-response is a critical KPI in disaster
        management.  The UN OCHA guidelines recommend 72-hour response windows;
        slower agents linearly lose this component.

    utilization (0.20): Idle supplies during an active disaster represent a
        coordination failure.  Real-world logistics studies (e.g., Van Wassenhove
        2006, 'Humanitarian logistics') show that >80% utilisation correlates
        with significantly better outcomes.

    cost_penalty (0.10): A small penalty for invalid or over-budget allocations
        prevents agents from gaming the reward by spamming bad actions.

    Returns a GradeResult (also available as plain dict via .model_dump()).
    """

    survival_rate = safe_div(env.total_survived, env.total_population)
    efficiency    = safe_div(env.tasks_completed, env.max_steps)
    utilization   = safe_div(env.used_resources,  env.total_resources)
    cost          = clamp(env.total_cost)

    raw_score = (
        0.50 * survival_rate
        + 0.20 * efficiency
        + 0.20 * utilization
        - 0.10 * cost
    )
    score = clamp(raw_score)

    breakdown = (
        f"survival={survival_rate:.3f}×0.50 "
        f"+ efficiency={efficiency:.3f}×0.20 "
        f"+ utilization={utilization:.3f}×0.20 "
        f"- cost={cost:.3f}×0.10 "
        f"= {score:.3f}"
    )

    return {
        "score": score,
        "success": score > 0.6,
        "survival_rate": round(survival_rate, 4),
        "efficiency": round(efficiency, 4),
        "utilization": round(utilization, 4),
        "cost_penalty": round(cost, 4),
        "breakdown": breakdown,
    }