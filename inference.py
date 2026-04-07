"""
inference.py — Baseline AI agent for SaaSAuditEnv.

Runs all three tasks using the OpenAI API as the agent brain. The agent
observes the environment, reasons about the subscription data, and takes
actions via HTTP calls to the environment server.

Environment variables:
    OPENAI_API_KEY   — required
    API_BASE_URL     — default http://localhost:8000
    MODEL_NAME       — default gpt-4o-mini
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import httpx
from openai import OpenAI, APIError, RateLimitError

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "") or os.environ.get("HF_TOKEN", "")
API_BASE_URL = os.environ.get("API_BASE_URL", "")  # LLM endpoint (hackathon requirement)
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# Environment server URL (where the SaaSAuditEnv Flask app runs)
ENV_SERVER_URL = os.environ.get("ENV_SERVER_URL", "http://localhost:7860").rstrip("/")

MAX_RETRIES = 3
RETRY_DELAY = 2.0

TASKS = ["task_easy", "task_medium", "task_hard", "task_startup", "task_merger"]

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a SaaS cost-optimization agent auditing a company's software subscriptions.

Your goal is to reduce unnecessary monthly spend while keeping business-critical tools running.

At each step you must respond with EXACTLY ONE JSON object — no prose, no markdown, no extra text.
The JSON must have an "action_type" field plus the required fields for that action.

Available actions:

1. Inspect a tool (learn details before acting):
{"action_type": "inspect_tool", "tool_id": "<id>"}

2. Reduce seats (must not go below active_users):
{"action_type": "reduce_seats", "tool_id": "<id>", "new_seat_count": <int>}

3. Downgrade plan (only for LOW/MEDIUM criticality tools):
{"action_type": "downgrade_plan", "tool_id": "<id>", "target_plan": "<plan>"}
Plans (lowest to highest): free, starter, basic, professional, business, enterprise

4. Cancel subscription (avoid CRITICAL/HIGH tools or must-preserve tools):
{"action_type": "cancel_subscription", "tool_id": "<id>", "reason": "<reason at least 10 chars>"}

5. Merge overlapping tools (cancel one, keep the other):
{"action_type": "merge_tools", "keep_tool_id": "<id>", "cancel_tool_id": "<id>", "reason": "<reason>"}

6. Submit final recommendation (ends the episode):
{"action_type": "submit_recommendation",
 "recommendations": [{"tool_id": "<id>", "action": "<action>", "estimated_monthly_savings": <float>, "justification": "<text>"}],
 "total_estimated_savings": <float>,
 "executive_summary": "<summary at least 30 chars>"}

CRITICAL RULES:
- Never cancel or downgrade tools with criticality "critical"
- Never cancel tools with criticality "high" if they have active users
- Never reduce seats below active_users
- Tools with 0 active users are safe to cancel if criticality is "low"
- Overlapping tools in the same category should be consolidated
- Submit your recommendation when you have achieved or are close to the savings target
- Be efficient — you have limited steps
"""


# ---------------------------------------------------------------------------
# HTTP client helpers
# ---------------------------------------------------------------------------

def env_request(method: str, path: str, body: Optional[Dict] = None) -> Dict[str, Any]:
    """Make an HTTP request to the environment server."""
    url = f"{ENV_SERVER_URL}{path}"
    with httpx.Client(timeout=30.0) as client:
        if method.upper() == "GET":
            resp = client.get(url)
        else:
            resp = client.post(url, json=body or {})
    resp.raise_for_status()
    return resp.json()


def reset_task(task_id: str) -> Dict[str, Any]:
    return env_request("POST", "/reset", {"task_id": task_id})


def take_step(action: Dict[str, Any]) -> Dict[str, Any]:
    return env_request("POST", "/step", {"action": action})


def get_state() -> Dict[str, Any]:
    return env_request("GET", "/state")


# ---------------------------------------------------------------------------
# Agent helpers
# ---------------------------------------------------------------------------

def build_user_message(obs: Dict[str, Any]) -> str:
    """Summarise the observation for the LLM."""
    subs = obs.get("subscriptions", [])
    sub_lines = []
    for s in subs:
        wasted = s["seats_purchased"] - s["active_users"]
        sub_lines.append(
            f"  - {s['tool_id']} ({s['name']}): "
            f"{s['seats_purchased']} seats / {s['active_users']} active "
            f"({wasted} wasted), "
            f"${s['monthly_cost']:.2f}/mo, "
            f"plan={s['plan']}, criticality={s['criticality']}"
            + (f", overlap_group={s['overlap_group']}" if s.get("overlap_group") else "")
            + (f", renewal_days={s['renewal_days']}" if s.get("renewal_days") else "")
        )

    history_lines = []
    for record in obs.get("action_history", [])[-5:]:  # last 5 actions
        status = "✓" if record["success"] else "✗"
        history_lines.append(f"  {status} [{record['action_type']}] {record['message']}")

    return f"""STEP {obs['current_step']}/{obs['max_steps']}
GOAL: {obs['goal']}
CURRENT SPEND: ${obs['current_monthly_spend']:.2f}/mo (was ${obs['original_monthly_spend']:.2f}/mo)
SAVINGS ACHIEVED: ${obs['savings_achieved']:.2f} / TARGET: ${obs['target_savings']:.2f}

ACTIVE SUBSCRIPTIONS:
{chr(10).join(sub_lines) if sub_lines else '  (none)'}

RECENT ACTIONS:
{chr(10).join(history_lines) if history_lines else '  (none yet)'}
{f"LAST ERROR: {obs['last_action_error']}" if obs.get('last_action_error') else ""}

Respond with ONE JSON action object."""


def call_llm(client: OpenAI, messages: List[Dict]) -> Optional[str]:
    """Call the LLM with retry logic. Returns the text content or None on failure."""
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.2,
                max_tokens=512,
            )
            return response.choices[0].message.content
        except RateLimitError:
            print(f"  [Rate limit] Waiting {RETRY_DELAY * (attempt + 1)}s...")
            time.sleep(RETRY_DELAY * (attempt + 1))
        except APIError as exc:
            print(f"  [API error] {exc}")
            if attempt == MAX_RETRIES - 1:
                return None
            time.sleep(RETRY_DELAY)
    return None


def parse_action_from_response(text: str) -> Optional[Dict[str, Any]]:
    """Extract a JSON action from the LLM response text."""
    if not text:
        return None
    # Strip markdown fences if present
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(
            line for line in lines
            if not line.strip().startswith("```")
        )
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object in the text
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
    return None


def fallback_action(obs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simple rule-based fallback when the LLM fails.
    Inspects the most wasteful tool or submits if near end.
    """
    step = obs.get("current_step", 0)
    max_steps = obs.get("max_steps", 20)
    subs = obs.get("subscriptions", [])

    # Near end: submit whatever we have
    if step >= max_steps - 2:
        return {
            "action_type": "submit_recommendation",
            "recommendations": [],
            "total_estimated_savings": obs.get("savings_achieved", 0.0),
            "executive_summary": (
                "Automated fallback submission. "
                "Achieved partial savings through seat reductions and cancellations."
            ),
        }

    # Find most wasteful tool
    best = None
    best_waste = 0
    for s in subs:
        waste = s["seats_purchased"] - s["active_users"]
        if waste > best_waste and s["criticality"] not in ("critical", "high"):
            best_waste = waste
            best = s

    if best and best_waste > 0:
        if best["active_users"] == 0:
            return {
                "action_type": "cancel_subscription",
                "tool_id": best["tool_id"],
                "reason": f"Zero active users out of {best['seats_purchased']} purchased seats.",
            }
        return {
            "action_type": "reduce_seats",
            "tool_id": best["tool_id"],
            "new_seat_count": max(1, best["active_users"]),
        }

    # Default: inspect something
    if subs:
        return {"action_type": "inspect_tool", "tool_id": subs[0]["tool_id"]}

    return {
        "action_type": "submit_recommendation",
        "recommendations": [],
        "total_estimated_savings": 0.0,
        "executive_summary": "No subscriptions found; nothing to optimise.",
    }


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(client: OpenAI, task_id: str) -> Dict[str, Any]:
    """Run a complete episode for the given task. Returns the episode result."""
    print(f"\n{'=' * 60}")
    print(f"  TASK: {task_id.upper()}")
    print(f"{'=' * 60}")

    # Reset environment
    reset_resp = reset_task(task_id)
    obs = reset_resp["observation"]

    # Structured output for validator
    print(f"[START] task={task_id}", flush=True)

    print(f"  Goal: {obs['goal'][:80]}...")
    print(f"  Starting spend: ${obs['current_monthly_spend']:.2f}/mo")
    print(f"  Target savings: ${obs['target_savings']:.2f}/mo")
    print(f"  Max steps: {obs['max_steps']}")
    print()

    conversation: List[Dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    episode_result = None
    llm_failures = 0

    while not obs.get("done", False):
        user_msg = build_user_message(obs)
        conversation.append({"role": "user", "content": user_msg})

        # Call LLM
        raw_response = call_llm(client, conversation)
        action = parse_action_from_response(raw_response or "")

        if action is None:
            llm_failures += 1
            print(f"  [step {obs['current_step'] + 1}] LLM parse failed, using fallback")
            action = fallback_action(obs)
        else:
            print(f"  [step {obs['current_step'] + 1}] {action.get('action_type', '?')}", end="")

        # Apply action
        try:
            step_result = take_step(action)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 400:
                # Environment returned an error (e.g. episode done)
                try:
                    error_body = exc.response.json()
                    print(f"\n  [env error] {error_body.get('error', str(exc))}")
                except Exception:
                    print(f"\n  [env error] {exc}")
                break
            print(f"\n  [HTTP error] {exc}")
            break
        except httpx.HTTPError as exc:
            print(f"\n  [HTTP error] {exc}")
            break

        obs = step_result["observation"]
        reward = step_result["reward"]
        last_record = obs["action_history"][-1] if obs["action_history"] else None

        if last_record:
            status = "✓" if last_record["success"] else "✗"
            short_msg = last_record["message"][:60]
            print(f" → {status} reward={reward:+.3f} | {short_msg}")

        # Structured output for validator
        step_num = obs.get("current_step", 0)
        print(f"[STEP] step={step_num} reward={reward} done={step_result.get('done', False)}", flush=True)

        if raw_response:
            conversation.append({"role": "assistant", "content": raw_response})

        if step_result.get("done") and "episode_result" in step_result.get("info", {}):
            episode_result = step_result["info"]["episode_result"]

    if episode_result is None:
        # Episode ended by max_steps or HTTP error without explicit submission
        try:
            state = get_state()
            episode_result = {
                "task_id": task_id,
                "savings_achieved": state.get("savings_achieved", 0),
                "target_savings": state.get("target_savings", 0),
                "score": 0.0,
                "total_reward": sum(
                    r["reward"] for r in state.get("action_history", [])
                ),
            }
        except Exception:
            # Fallback if state is also unreachable
            episode_result = {
                "task_id": task_id,
                "savings_achieved": obs.get("savings_achieved", 0),
                "target_savings": obs.get("target_savings", 0),
                "score": 0.0,
                "total_reward": sum(
                    r["reward"]
                    for r in obs.get("action_history", [])
                ),
            }

    print()
    print(f"  RESULT: score={episode_result.get('score', 0):.4f} | "
          f"savings=${episode_result.get('savings_achieved', 0):.2f} / "
          f"${episode_result.get('target_savings', 0):.2f} target")
    if llm_failures:
        print(f"  LLM failures (used fallback): {llm_failures}")

    # Structured output for validator
    ep_score = episode_result.get('score', 0)
    ep_steps = episode_result.get('steps_taken', obs.get('current_step', 0))
    print(f"[END] task={task_id} score={ep_score} steps={ep_steps}", flush=True)

    return episode_result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY environment variable is not set.")
        sys.exit(1)

    # Verify server is reachable
    try:
        health = env_request("GET", "/health")
        print(f"Environment server: {health['status']} (version {health['version']})")
    except Exception as exc:
        print(f"ERROR: Cannot reach environment server at {ENV_SERVER_URL}: {exc}")
        print("Start the server with:  hypercorn app.main:app --bind 0.0.0.0:7860")
        sys.exit(1)

    # Create OpenAI client with optional base_url for hackathon LLM endpoint
    client_kwargs = {"api_key": OPENAI_API_KEY}
    if API_BASE_URL:
        client_kwargs["base_url"] = API_BASE_URL
    client = OpenAI(**client_kwargs)

    results = []
    for task_id in TASKS:
        result = run_episode(client, task_id)
        results.append(result)

    # Summary table
    print(f"\n{'=' * 60}")
    print("  FINAL SCORES")
    print(f"{'=' * 60}")
    print(f"  {'Task':<15} {'Score':>8} {'Savings':>12} {'Target':>12}")
    print(f"  {'-'*15} {'-'*8} {'-'*12} {'-'*12}")
    total_score = 0.0
    for r in results:
        score = r.get("score", 0.0)
        total_score += score
        print(
            f"  {r['task_id']:<15} {score:>8.4f} "
            f"${r.get('savings_achieved', 0):>10.2f} "
            f"${r.get('target_savings', 0):>10.2f}"
        )
    avg = total_score / len(results) if results else 0.0
    print(f"  {'AVERAGE':<15} {avg:>8.4f}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()