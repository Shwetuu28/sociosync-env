from fastapi import FastAPI
from env import SocioSyncEnv
from models import Action

app = FastAPI()

env = SocioSyncEnv()
obs = env.reset()


@app.get("/")
def home():
    return {"message": "SocioSync-Env is running 🚀"}


@app.post("/step")
def step(action: dict):
    global obs

    action_obj = Action(**action)

    obs, reward, done, _ = env.step(action_obj)

    return {
        "observation": obs.dict(),
        "reward": reward,
        "done": done
    }


@app.post("/reset")
def reset():
    global obs
    obs = env.reset()

    return {
        "observation": obs.dict()
    }