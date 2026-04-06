def grade_environment(env):
    obs = env.state()

    avg_skill = (
        obs.low_skill +
        obs.mid_skill +
        obs.high_skill
    ) / 3

    demand_low = 0.4
    demand_mid = 0.35
    demand_high = 0.25

    alignment = (
        abs(obs.low_skill - demand_low) +
        abs(obs.mid_skill - demand_mid) +
        abs(obs.high_skill - demand_high)
    )

    alignment_score = max(0, 1 - alignment)

    employment_score = (1 - obs.unemployment_rate) ** 1.5
    skill_score = avg_skill ** 1.2

    score = (
        0.5 * employment_score +
        0.2 * skill_score +
        0.2 * alignment_score +
        0.1 * obs.economic_growth -
        0.3 * obs.inequality
    )

    if obs.budget < -20:
        score -= 0.2

    if obs.unemployment_rate > 0.7:
        score -= 0.2

    return max(0, min(score, 1))