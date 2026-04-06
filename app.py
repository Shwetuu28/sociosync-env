from fastapi import FastAPI
from env import SocioSyncEnv
from models import Action

app = FastAPI()

env = SocioSyncEnv()

@app.post("/reset")
def reset():
    obs = env.reset()
    return obs.dict()

@app.post("/step")
def step(action: dict):
    act = Action(**action)
    obs, reward, done, _ = env.step(act)

    return {
        "observation": obs.dict(),
        "reward": reward,
        "done": done,
        "info": {}
    }