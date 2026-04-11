# tasks.py

from env import RescueNetEnv

def easy_1():
    env = RescueNetEnv(mode="easy")
    env.max_steps = 15
    env.severity_multiplier = 0.7
    return env


def medium_1():
    env = RescueNetEnv(mode="medium")
    env.max_steps = 20
    env.severity_multiplier = 1.0
    return env


def hard_1():
    env = RescueNetEnv(mode="hard")
    env.max_steps = 25
    env.severity_multiplier = 1.3
    env.available_resources = [6.0, 6.0, 6.0]
    return env