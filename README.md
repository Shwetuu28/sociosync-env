---
title: RescueNet Env
emoji: 🚨
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
  - disaster-response
  - resource-allocation
  - reinforcement-learning
  - real-world
---

# 🚨 RescueNet-Env

### Disaster Response Resource Allocation — OpenEnv Hackathon Submission

> **An OpenEnv-compliant RL environment where an AI agent plays the role of an emergency coordinator — dispatching scarce resources across disaster-struck regions to maximize population survival under time pressure.**

🔗 **[Live Demo on Hugging Face Spaces](https://huggingface.co/spaces/Shwetuu28/rescuenet-env)** 

---

## ⚡ Quick Start (No Setup Required)

```bash
# Reset the environment — get your first observation
curl -X POST https://shwetuu28-rescuenetenv.hf.space/reset \
     -H "Content-Type: application/json" -d '{}'

# Take an action — dispatch medical supplies to region 2
curl -X POST https://shwetuu28-rescuenetenv.hf.space/step \
     -H "Content-Type: application/json" \
     -d '{"region_id": 2, "resource_type": "medical", "quantity": 1.5}'

# Check current state
curl https://shwetuu28-rescuenetenv.hf.space/state
```

Or run the full inference loop locally in one command:

```bash
git clone https://github.com/Shwetuu28/RescueNetEnv.git && cd RescueNetEnv
pip install -r requirements.txt
python inference.py   # greedy fallback runs even without an API key
```

---

## 💡 Why This Problem?

When disasters strike — earthquakes, floods, epidemics — emergency responders must allocate scarce resources (food, medical supplies, rescue teams) across multiple affected regions simultaneously. Every delayed decision costs lives.

This is a genuinely hard sequential decision-making problem:

- Resources are **finite and shared** across competing regions
- Severity evolves over time — inaction **compounds damage exponentially**
- No single greedy strategy works across all scenarios
- The hardest scenarios introduce **noisy sensor data** to simulate real-world information uncertainty

Real emergency management agencies face exactly this problem. RescueNet-Env makes it a rigorous RL benchmark.

---

## 🧠 Environment Design

### Survival Model

Population decay at each step follows an exponential model driven by severity and response delay:

```
alive(t+1) = alive(t) × exp(−severity × delay)
```

`delay` increments every step a region receives no aid and resets on dispatch. Inaction is never neutral — every missed step accelerates deaths.

### Agent Loop

```
env.reset()
    │
    ▼
Observation ──► Agent (LLM or policy)
                    │
                    ▼ Action: {region_id, resource_type, quantity}
                env.step(action)
                    │
                    ▼
        reward + next Observation
                    │
              (repeat until done)
                    │
                    ▼
            grader → score ∈ [0.01, 0.99]
```

### Observation Space

```python
class Region(BaseModel):
    population: int            # initial population
    severity: float            # disaster severity in [0.0, 1.0]
    delay: int                 # steps since last dispatch
    resource_need: List[float] # relative need for [food, medical, rescue]
    alive: float               # surviving population estimate
    sensor_note: Optional[str] # non-null on hard task: sensor data may be corrupted
    phantom_note: Optional[str]# non-null on hard task: demand may be spurious

class Observation(BaseModel):
    regions: List[Region]
    available_resources: List[float]  # remaining [food, medical, rescue]
    time_step: int
```

### Action Space

```python
class Action(BaseModel):
    region_id: int       # which region to dispatch to (0-indexed)
    resource_type: str   # "food" | "medical" | "rescue"
    quantity: float      # amount in [0.0, 2.0]
```

### Reward Function

Dense per-step reward — every step counts:

```
reward = (Δalive / total_population) − 0.05 × (unused_resources / total_resources)
```

Clipped to `[−1.0, 1.0]`. Primary signal is the change in total surviving population. A secondary penalty discourages hoarding supplies while people die.

---

## 🎮 Tasks

Three tasks with a clear easy → medium → hard progression, all defined in `tasks.py` and registered in `openenv.yaml`.

### 🟢 `easy_1`

| Regions | Max Steps | Severity | Resources |
|---------|-----------|----------|-----------|
| 5 | 15 | 0.7× | 10 per type |

Full observability, lower severity. A well-tuned greedy agent scores ~0.68. Designed to confirm basic dispatch logic works.

### 🟡 `medium_1`

| Regions | Max Steps | Severity | Resources |
|---------|-----------|----------|-----------|
| 7 | 20 | 1.0× | 10 per type |

More regions compete for the same resource budget. Greedy-by-severity underperforms — the agent must learn to prioritize.

### 🔴 `hard_1`

| Regions | Max Steps | Severity | Resources |
|---------|-----------|----------|-----------|
| 10 | 25 | 1.3× | 6 per type |

Scarce resources across 10 high-severity regions. Some regions **will not receive aid** — optimal play requires multi-step lookahead and triage reasoning. The environment also introduces **corrupted sensor readings** (`sensor_note`) and **phantom demand signals** (`phantom_note`), so agents cannot blindly trust observations. Greedy agents score ~0.38 — well below the 0.60 success threshold.

### 📊 Baseline Scores

| Task | Baseline (Greedy) | Success Threshold |
|------|-------------------|-------------------|
| `easy_1` | ~0.68 | > 0.60 ✅ |
| `medium_1` | ~0.55 | > 0.60 ❌ |
| `hard_1` | ~0.38 | > 0.60 ❌ |

The greedy policy passes easy but fails medium and hard — demonstrating that this is a genuine benchmark requiring learned behavior.

---

## 🧪 Grader

End-of-episode scoring from `grader.py`:

```
score = 0.5 × survival_rate
      + 0.2 × efficiency        # tasks_completed / max_steps
      + 0.2 × utilization       # resources used / resources available
      − 0.1 × cost_penalty      # penalty for invalid allocations

score clamped to [0.01, 0.99]   success = score > 0.6
```

```python
class GradeResult(BaseModel):
    score: float          # final score in [0.01, 0.99]
    success: bool         # True if score > 0.6
    survival_rate: float
    efficiency: float
    utilization: float
    cost_penalty: float
    breakdown: str
```

---

## 🌐 API Reference

The environment runs as a FastAPI server on port `7860`.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset` | POST | Reset environment, returns initial `Observation` |
| `/step` | POST | Apply `Action`, returns `Observation`, reward, done |
| `/state` | GET | Returns raw internal state dict |

**Example `/step` request:**
```json
{ "region_id": 2, "resource_type": "medical", "quantity": 1.5 }
```

**Example `/step` response:**
```json
{
  "observation": {
    "regions": [{ "population": 150, "severity": 0.82, "delay": 1, "alive": 138.4, "resource_need": [1.0, 1.0, 1.0] }],
    "available_resources": [10.0, 8.5, 10.0],
    "time_step": 1
  },
  "reward": 0.14,
  "done": false,
  "info": {}
}
```

---

## 🤖 Inference Script

`inference.py` runs all three tasks in sequence, emitting strict `[START]` / `[STEP]` / `[END]` stdout logs. It uses an LLM if `HF_TOKEN` is set, otherwise falls back to a deterministic greedy policy — **the environment and grader work fully either way**.

```
[START] task=easy_1 env=rescuenet-env model=gpt-4o-mini
[STEP] step=1 action=medical(r3,1.50) reward=0.12 done=false error=null
[STEP] step=2 action=rescue(r0,1.00) reward=0.08 done=false error=null
...
[END] success=true steps=15 score=0.712 rewards=0.12,0.08,...
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `API_BASE_URL` | `https://api.openai.com/v1` | LLM API endpoint |
| `MODEL_NAME` | `gpt-4o-mini` | Model identifier |
| `HF_TOKEN` | — | API key (optional — greedy fallback runs if unset) |

```bash
# With LLM
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4o-mini
export HF_TOKEN=your_key
python inference.py

# Without API key — greedy fallback runs automatically
python inference.py
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
  -e HF_TOKEN=your_key \
  rescuenet-env python inference.py
```

---

## ✅ Validation

```bash
pip install openenv-core
openenv validate

# Full submission check (replace with your HF Space URL)
./validate-submission.sh https://shwetuu28-rescuenetenv.hf.space .
```

---

## 📁 Project Structure

```
RescueNetEnv/
├── server/           # FastAPI app — /reset, /step, /state endpoints
├── env.py            # Core RL environment: survival model, step(), reset(), state()
├── models.py         # Pydantic models: Region, Observation, Action, GradeResult
├── tasks.py          # Task factories: easy_1, medium_1, hard_1
├── grader.py         # End-of-episode scoring, normalized to [0.01, 0.99]
├── inference.py      # LLM agent + greedy fallback + compliant stdout logging
├── openenv.yaml      # OpenEnv spec: observation/action spaces, task registry
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 🏆 What Makes This Stand Out

| Property | Status |
|---|---|
| Real-world task (not a toy) | ✅ Disaster triage — immediate RL/agent eval value |
| Dense reward every step | ✅ Per-step survival delta, not sparse end-reward |
| Genuine difficulty progression | ✅ Severity × scarcity × region count |
| Greedy agent fails on medium/hard | ✅ Proves benchmark requires learned behavior |
| Noisy observations on hard task | ✅ `sensor_note` + `phantom_note` — tests observation trust |
| Seeded reproducibility | ✅ `reset(seed=N)` fully supported |
| OpenEnv spec compliant | ✅ `step` / `reset` / `state` + `openenv.yaml` validated |
| Safe without API key | ✅ Greedy fallback — no inference failure on missing token |
| Docker + HF Spaces ready | ✅ Port 7860, containerized |

---

## 👥 Team

- **Shweta** — RL environment design, survival model, reward engineering
- **Ranjita** — API integration, deployment, documentation

---

## Citation

```bibtex
@software{rescuenetenv2026,
  title   = {RescueNet-Env: Disaster Response Resource Allocation as an OpenEnv RL Benchmark},
  author  = {Shweta and Ranjita},
  year    = {2026},
  url     = {https://huggingface.co/spaces/Shwetuu28/RescueNetEnv},
  note    = {OpenEnv-compliant RL environment for disaster response agent evaluation}
}
```
