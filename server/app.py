from fastapi import FastAPI
from sociosync_env import SocioSyncEnv
from pydantic import BaseModel

app = FastAPI()

env = SocioSyncEnv()

class ActionInput(BaseModel):
    action_type: str
    intensity: float

@app.post("/reset")
def reset():
    obs = env.reset()
    return {"observation": obs.__dict__}

@app.post("/step")
def step(action: ActionInput):
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs.__dict__,
        "reward": reward,
        "done": done,
        "info": info
    }