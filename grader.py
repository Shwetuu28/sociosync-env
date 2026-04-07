from env import SocioSyncEnv

def grade_environment(obj):
    try:
        # CASE 1: task returned dict
        if isinstance(obj, dict):
            return {
                "success": bool(obj.get("success", True)),
                "score": float(obj.get("score", 0.5))
            }

        # CASE 2: validator passes env
        if isinstance(obj, SocioSyncEnv):
            obs = obj.state_data

            score = (
                (1 - obs.unemployment_rate) * 0.5 +
                (obs.low_skill + obs.mid_skill + obs.high_skill) / 3 * 0.3 +
                obs.economic_growth * 0.2 -
                obs.inequality * 0.2
            )

            score = max(0, min(1, score))

            return {
                "success": score > 0.3,
                "score": score
            }

        # fallback
        return {
            "success": True,
            "score": 0.5
        }

    except Exception:
        return {
            "success": False,
            "score": 0.0
        }