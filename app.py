# server/app.py

from fastapi import FastAPI
from pydantic import BaseModel
from env import RescueNetEnv
from models import Action

app = FastAPI()

env = RescueNetEnv()


# -----------------------------
# REQUEST MODEL
# -----------------------------
class StepRequest(BaseModel):
    region_id: int
    resource_type: str
    quantity: float


# -----------------------------
# RESET
# -----------------------------
@app.post("/reset")
def reset():
    obs = env.reset()

    return {
        "regions": [
            {
                "population": r.population,
                "severity": r.severity,
                "delay": r.delay,
                "alive": r.alive
            } for r in obs.regions
        ],
        "available_resources": obs.available_resources,
        "time_step": obs.time_step
    }


# -----------------------------
# STEP
# -----------------------------
@app.post("/step")
def step(req: StepRequest):
    action = Action(
        region_id=req.region_id,
        resource_type=req.resource_type,
        quantity=req.quantity
    )

    obs, reward, done, _ = env.step(action)

    return {
        "observation": {
            "regions": [
                {
                    "population": r.population,
                    "severity": r.severity,
                    "delay": r.delay,
                    "alive": r.alive
                } for r in obs.regions
            ],
            "available_resources": obs.available_resources,
            "time_step": obs.time_step
        },
        "reward": reward,
        "done": done
    }

@app.get("/state")
def state():
    return env.state()

# -----------------------------
# ENTRY POINT
# -----------------------------
def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()