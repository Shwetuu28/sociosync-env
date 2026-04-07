---
title: RescueNet Env
emoji: 🚨
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
---
 
# 🚨 RescueNet-Env
 
### Disaster Response Resource Allocation — OpenEnv Hackathon Submission
 
---
 
## 🌍 Problem
 
When disasters strike — earthquakes, floods, epidemics — emergency responders must allocate limited resources (food, medical supplies, rescue teams) across multiple affected regions simultaneously. Every delayed decision costs lives.
 
This is a genuinely hard sequential decision-making problem:
 
- Resources are scarce and shared across competing regions
- Severity changes over time — inaction compounds damage
- Response delay increases mortality exponentially
- No single greedy strategy works across all scenarios
 
Real emergency management organizations face exactly this problem. **RescueNet-Env** simulates it as a reinforcement learning environment where agents must learn to dispatch resources intelligently under pressure.
 
---
 
## 💡 Solution
 
**RescueNet-Env** is a fully OpenEnv-compliant RL environment where an AI agent acts as an emergency coordinator. At each timestep the agent selects a region and dispatches a resource. The environment tracks survival dynamics using an exponential decay model driven by severity and response delay.
 
The agent must learn to prioritize high-severity regions, minimize dispatch delay, and avoid wasting limited supplies — all simultaneously.
 
---
 
## 🧠 Environment Design
 
### Survival Model
 
At every step, each region's alive population evolves as:
 
```
alive(t+1) = alive(t) × exp(−severity × delay)
```
 
Where `delay` increases by 1 each step a region goes without aid, and resets when resources are dispatched. This means inaction is never neutral — every missed step accelerates deaths.
 
### Observation Space
 
```yaml
regions:
  - population: int        # initial population of the region
    severity: float        # disaster severity in [0.0, 1.0]
    delay: int             # steps since last resource dispatch
    alive: float           # current surviving population
available_resources:
  - food: float            # remaining food supply
  - medical: float         # remaining medical supply
  - rescue: float          # remaining rescue supply
time_step: int             # current step in episode
```
 
### Action Space
 
```yaml
region_id: int             # index of region to dispatch to (0 to N-1)
resource_type: string      # one of: "food", "medical", "rescue"
quantity: float            # amount to dispatch in [0.0, 2.0]
```
 
### Reward Function
 
Each step:
 
```
reward = (Δalive / total_population) − 0.05 × (unused_resources / total_resources)
```
 
- **Primary signal:** change in total alive population (positive when agent saves lives, negative when population declines)
- **Secondary penalty:** idle resources — hoarding supplies while people die is penalized
- **Clipped** to `[−1.0, 1.0]` for training stability
 
This reward is **dense** — every step provides a meaningful signal, not just end-of-episode.
 
---
 
## 🎮 Tasks
 
### 🟢 Easy — `easy_1`
 
| Property | Value |
|---|---|
| Regions | 5 |
| Max Steps | 15 |
| Severity Multiplier | 0.7× (reduced) |
| Resources | 10 per type |
 
Full observability, lower severity, small map. A well-tuned greedy agent can achieve ~0.65–0.75. Designed to verify basic dispatch logic works.
 
### 🟡 Medium — `medium_1`
 
| Property | Value |
|---|---|
| Regions | 7 |
| Max Steps | 20 |
| Severity Multiplier | 1.0× (baseline) |
| Resources | 10 per type |
 
More regions competing for the same resource budget. Agent must learn prioritization — dispatching to the wrong region costs survival in others. Greedy-by-severity performs suboptimally here.
 
### 🔴 Hard — `hard_1`
 
| Property | Value |
|---|---|
| Regions | 10 |
| Max Steps | 25 |
| Severity Multiplier | 1.3× (amplified) |
| Resources | 6 per type (scarce) |
 
High-severity disasters across 10 regions with only 6 units per resource type. Agents must make true trade-off decisions — some regions will not receive aid. GPT-4-level greedy reasoning breaks down here because optimal allocation requires multi-step lookahead under scarcity.
 
### 📊 Baseline Scores (Greedy Fallback Agent)
 
| Task | Baseline Score | Success Threshold |
|---|---|---|
| easy_1 | ~0.68 | > 0.60 |
| medium_1 | ~0.55 | > 0.60 |
| hard_1 | ~0.38 | > 0.60 |
 
---
 
## 🧪 Grader
 
The grader evaluates the final environment state after episode completion:
 
```python
score = (
    0.5 × survival_rate       # fraction of initial population still alive
  + 0.2 × efficiency           # tasks_completed / max_steps
  + 0.2 × utilization          # resources used / resources available
  − 0.1 × cost_penalty         # penalty for invalid allocations
)
```
 
All components normalized. Score clamped to `[0.01, 0.99]`. Success threshold: `score > 0.6`.
 
---
 
## 🌐 API Endpoints
 
The environment exposes a FastAPI server on port `7860`:
 
### `POST /reset`
 
Resets the environment and returns the initial observation.
 
**Response:**
```json
{
  "regions": [
    { "population": 150, "severity": 0.82, "delay": 1, "alive": 150.0 }
  ],
  "available_resources": [10.0, 10.0, 10.0],
  "time_step": 0
}
```
 
### `POST /step`
 
Applies an action and returns the next observation, reward, and done flag.
 
**Request:**
```json
{
  "region_id": 2,
  "resource_type": "medical",
  "quantity": 1.5
}
```
 
**Response:**
```json
{
  "observation": {
    "regions": [...],
    "available_resources": [10.0, 8.5, 10.0],
    "time_step": 1
  },
  "reward": 0.14,
  "done": false
}
```
 
---
 
## 🤖 Inference Script
 
The agent uses an OpenAI-compatible client. If no API key is available, it falls back to a deterministic greedy policy (always dispatches medical to highest-severity region).
 
### Stdout Format (Strictly Compliant)
 
```
[START] task=easy_1 env=rescuenet-env model=gpt-4o-mini
 
[STEP] step=1 action=medical(r3,1.50) reward=0.12 done=false error=null
[STEP] step=2 action=rescue(r0,1.00) reward=0.08 done=false error=null
...
 
[END] success=true steps=15 score=0.712 rewards=0.12,0.08,...
```
 
### Running Inference
 
```bash
# With LLM agent
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4o-mini
export HF_TOKEN=your_api_key
python inference.py
 
# Without API key (greedy fallback runs automatically)
python inference.py
```
 
---
 
## 🚀 Local Setup
 
```bash
<<<<<<< HEAD
# Clone the repo
git clone https://huggingface.co/spaces/<your-username>/rescuenet-env
cd rescuenet-env
 
# Install dependencies
pip install -r requirements.txt
 
# Run the server
python app.py
 
# Or run inference directly
python inference.py
=======
API_BASE_URL=https://api.openai.com/v1
MODEL_NAME=gpt-4o-mini
API_KEY=your_api_key
LOCAL_IMAGE_NAME=optional_docker_image
>>>>>>> e0ae444 (Fix: add tasks with graders in openenv.yaml)
```
 
---
 
## 🐳 Docker
 
```bash
# Build
docker build -t rescuenet-env .
 
# Run server
docker run -p 7860:7860 rescuenet-env
 
# Run inference inside container
docker run \
  -e API_BASE_URL=https://api.openai.com/v1 \
  -e MODEL_NAME=gpt-4o-mini \
  -e HF_TOKEN=your_api_key \
  rescuenet-env python inference.py
```
 
---
 
## 🔑 Environment Variables
 
| Variable | Required | Default | Description |
|---|---|---|---|
| `API_BASE_URL` | No | `https://api.openai.com/v1` | LLM API endpoint |
| `MODEL_NAME` | No | `gpt-4o-mini` | Model identifier |
| `HF_TOKEN` | No | — | Hugging Face / API key |
 
If `HF_TOKEN` is not set, the greedy fallback policy runs automatically — the environment and grader still function fully.
 
---
 
## ✅ Pre-Submission Validation
 
```bash
# Install validator
pip install openenv-core
 
# Run OpenEnv spec check
openenv validate
 
# Run submission validator (replace URL with your HF Space URL)
./validate-submission.sh https://your-space.hf.space .
```
 
---
 
## 📁 Project Structure
 
```
rescuenet-env/
├── app.py            # FastAPI server (POST /reset, POST /step)
├── env.py            # RescueNetEnv — core RL environment
├── models.py         # Typed models: Region, Observation, Action
├── tasks.py          # Task factories: easy_1, medium_1, hard_1
├── grader.py         # Scoring logic, normalized [0.01, 0.99]
├── inference.py      # LLM agent + greedy fallback + stdout logging
├── openenv.yaml      # OpenEnv spec metadata
├── requirements.txt  # Python dependencies
├── Dockerfile        # Container definition
└── README.md         # This file
```
 
---
 
## 🏆 Why RescueNet-Env
 
| Property | Status |
|---|---|
| Real-world task (not a toy) | ✅ Disaster response coordination |
| Dense reward signal | ✅ Per-step survival delta |
| True difficulty progression | ✅ Severity × scarcity × region count |
| Greedy agent fails on hard task | ✅ Score ~0.38 vs threshold 0.60 |
| OpenEnv spec compliant | ✅ step / reset / state + openenv.yaml |
| Reproducible with seeds | ✅ reset(seed=N) supported |
| HF Spaces deployable | ✅ Docker + port 7860 |
| LLM fallback safe | ✅ Greedy policy if no API key |
 
---
 
## 👥 Team
 
- **Shweta** — RL Environment Design, Survival Model, Reward Engineering
- **Ranjita** — API Integration, Deployment, Documentation
