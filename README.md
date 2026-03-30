---
title: SaaSAuditEnv
emoji: 💰
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
tags:
  - openenv
pinned: false
---

# SaaSAuditEnv

> An OpenEnv-style AI agent environment for SaaS cost optimization.

An AI agent audits a company's SaaS subscriptions and makes cost-saving decisions —
reducing seats, cancelling unused tools, consolidating overlapping products —
**without breaking business-critical operations**.

---

## Environment Description & Motivation

Companies accumulate dozens of SaaS subscriptions over time: project management tools,
CRMs, design platforms, communication apps, and more. Many of these have unused seats,
overlapping functionality, or are entirely abandoned. Manual auditing is tedious and
error-prone.

**SaaSAuditEnv** simulates this real-world task so AI agents can learn to:
- Identify wasted seats and reduce them
- Spot overlapping/duplicate tools and consolidate
- Safely cancel unused subscriptions
- Hit savings targets without disrupting critical business operations

This has immediate value for the RL/agent community as it models a genuine decision-making
task with partial observability, safety constraints, and multi-objective optimization.

---

## Overview

| Property | Value |
|---|---|
| Framework | Flask + Hypercorn |
| Models | Pydantic v2 |
| LLM Client | OpenAI Python SDK |
| Test Runner | unittest |
| Tasks | 3 (easy / medium / hard) |
| Grading | Deterministic, 0.0 – 1.0 |

---

## Action Space

| Action | Required Fields | Description |
|---|---|---|
| `inspect_tool` | `tool_id` | Retrieve detailed information about a subscription |
| `reduce_seats` | `tool_id`, `new_seat_count` | Reduce seats (must be ≥ active_users) |
| `downgrade_plan` | `tool_id`, `target_plan` | Downgrade plan tier (LOW/MEDIUM criticality only) |
| `cancel_subscription` | `tool_id`, `reason` | Cancel entirely (avoid CRITICAL/HIGH tools) |
| `merge_tools` | `keep_tool_id`, `cancel_tool_id`, `reason` | Consolidate overlapping tools |
| `submit_recommendation` | `recommendations`, `total_estimated_savings`, `executive_summary` | Submit final audit report (ends episode) |

## Observation Space

Each observation includes:
- `task_id`, `goal` — current task identity and objective
- `current_step`, `max_steps` — progress tracking
- `current_monthly_spend`, `original_monthly_spend` — spend before and after actions
- `target_savings`, `savings_achieved` — how much the agent needs to save
- `subscriptions` — full list with: tool_id, name, department, plan, seats_purchased, active_users, cost_per_seat_monthly, monthly_cost, wasted_seats, utilization_rate, criticality, overlap_group, renewal_days, notes
- `action_history` — record of all actions taken with rewards
- `last_action_error` — error message from last failed action (if any)

---

## Tasks

### Task 1: Easy (`task_easy`)
- **Target savings:** $400/mo | **Max steps:** 15 | **Tools:** 5
- **Scenario:** Small company with obvious seat waste on Zoom (28 unused seats) and Canva (18 unused seats with only 2 active users). Simple right-sizing.
- **Expected difficulty:** Low — clear signals, minimal constraints

### Task 2: Medium (`task_medium`)
- **Target savings:** $900/mo | **Max steps:** 18 | **Tools:** 7
- **Scenario:** Overlapping project management (Jira + Asana + Basecamp) and docs (Confluence + Notion). Requires consolidation plus seat right-sizing. Must preserve Jira, Google Workspace, PagerDuty.
- **Expected difficulty:** Medium — requires overlap detection and safe consolidation

### Task 3: Hard (`task_hard`)
- **Target savings:** $2,500/mo | **Max steps:** 25 | **Tools:** 18
- **Scenario:** Large portfolio across Sales, Engineering, HR, Design, Marketing, IT. Four overlap groups (CRM, design, HR, project management). Multiple critical tools. Tight savings target.
- **Expected difficulty:** Hard — large action space, many constraints, requires planning

### Task 4: Medium-Hard (`task_startup`)
- **Target savings:** $1,200/mo | **Max steps:** 20 | **Tools:** 11
- **Scenario:** Fast-growing 25-person startup that signed expensive enterprise plans during a funding round. Most tools only need professional or basic tiers. Tests **plan downgrade** actions heavily, plus overlap consolidation (analytics, docs).
- **Expected difficulty:** Medium-Hard — requires recognizing over-provisioned plans, not just seat waste

### Task 5: Expert (`task_merger`)
- **Target savings:** $4,000/mo | **Max steps:** 30 | **Tools:** 16
- **Scenario:** Post-acquisition merger with duplicate tool stacks in every department. Seven overlap groups (chat, cloud, CRM, VCS, identity, project management, video). Each category has competing tools from acquirer vs acquired company.
- **Expected difficulty:** Expert — 7 consolidation decisions, large scale, ambiguity in tool selection

---

## Setup Instructions

### Local Development

```bash
# Clone and install
pip install -r requirements.txt

# Run server
hypercorn app.main:app --bind 0.0.0.0:7860

# Run tests
python -m unittest discover tests -v

# Run inference (requires LLM API key)
export OPENAI_API_KEY=your-key
export MODEL_NAME=gpt-4o-mini
python inference.py
```

### Docker

```bash
docker build -t saas-audit-env .
docker run -p 7860:7860 saas-audit-env
```

### Hugging Face Spaces

This project is configured for HF Spaces with Docker SDK. Push to a Space tagged with `openenv`:

```bash
git push https://huggingface.co/spaces/YOUR_USERNAME/saas-audit-env main
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness check |
| GET | `/tasks` | List available tasks |
| POST | `/reset` | Start new episode: `{"task_id": "task_easy"}` |
| POST | `/step` | Take action: `{"action": {"action_type": "...", ...}}` |
| GET | `/state` | Full internal state (debugging) |

### Example: Reset + Step

```bash
# Reset
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task_easy"}'

# Inspect a tool
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"action_type": "inspect_tool", "tool_id": "zoom_001"}}'

# Reduce seats
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"action_type": "reduce_seats", "tool_id": "zoom_001", "new_seat_count": 22}}'
```

---

## Baseline Scores

Using GPT-4o-mini with the default system prompt:

| Task | Score | Savings Achieved | Target |
|---|---|---|---|
| task_easy | ~0.90 | ~$876 | $400 |
| task_medium | ~0.75 | ~$900 | $900 |
| task_hard | ~0.55 | ~$2,100 | $2,500 |

---

## Reward Shaping

| Event | Reward |
|---|---|
| First-time tool inspection | +0.02 |
| Seat reduction (proportional to savings) | +0.10 to +0.60 |
| Plan downgrade | +0.10 to +0.50 |
| Safe cancellation | +0.15 to +0.65 |
| Valid merge (same overlap group) | +0.15 to +0.65 |
| Submission (target met) | +0.25 |
| Invalid action | -0.05 |
| Reduce below active users | -0.10 |
| Downgrade critical tool | -0.15 |
| Cancel high-criticality tool | -0.15 to -0.20 |
| Cancel must-preserve tool | -0.30 |

---

## Grading

Each task grader is deterministic and returns a score in [0.0, 1.0]:

- **Savings achieved** (40-50%): fraction of target savings met
- **Critical tools preserved** (20-25%): binary — all must-preserve tools intact
- **Overlap handling** (15-20%): correctly consolidated duplicate tools
- **Seat right-sizing** (10-20%): reduced wasted seats
- **Submission bonus** (5-10%): submitted a recommendation with accurate estimate

Scores are multiplied by an invalid-action penalty (max 50% reduction for excessive errors).

---

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `OPENAI_API_KEY` | LLM API key (required for inference) | — |
| `API_BASE_URL` | LLM API endpoint | — |
| `MODEL_NAME` | LLM model identifier | `gpt-4o-mini` |
| `HF_TOKEN` | Hugging Face / API key | — |
| `ENV_SERVER_URL` | Environment server URL | `http://localhost:7860` |
| `PORT` | Server port | `7860` |
