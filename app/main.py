"""Flask application for SaaSAuditEnv.

Routes:
    GET  /health        — liveness check
    POST /reset         — start new episode
    POST /step          — take one action
    GET  /state         — inspect full internal state
    GET  /tasks         — list available tasks
"""

from __future__ import annotations

import traceback

from flask import Flask, jsonify, request

from app.config import config
from app.env import SaaSAuditEnv
from app.models import ActionRequest, HealthResponse, ResetRequest
from app.tasks import TASK_REGISTRY

app = Flask(__name__)
app.config["SECRET_KEY"] = config.SECRET_KEY

# Single environment instance per process (sufficient for hackathon/Spaces use).
env = SaaSAuditEnv()


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    """Liveness probe."""
    resp = HealthResponse(
        status="ok",
        version=SaaSAuditEnv.VERSION,
        active_task=env.current_task.task_id if env.current_task else None,
    )
    return jsonify(resp.model_dump()), 200


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

@app.route("/tasks", methods=["GET"])
def list_tasks():
    """List all available tasks with metadata."""
    tasks = []
    for task in TASK_REGISTRY.values():
        total_spend = sum(
            s.seats_purchased * s.cost_per_seat_monthly
            for s in task.subscriptions
        )
        tasks.append({
            "task_id": task.task_id,
            "difficulty": task.difficulty,
            "goal": task.goal,
            "max_steps": task.max_steps,
            "target_savings": task.target_savings,
            "num_subscriptions": len(task.subscriptions),
            "total_monthly_spend": round(total_spend, 2),
        })
    return jsonify({"tasks": tasks}), 200


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

@app.route("/reset", methods=["POST"])
def reset():
    """Start a new episode.

    Request body::

        {"task_id": "task_easy"}
    """
    body = request.get_json(force=True, silent=True) or {}
    try:
        req = ResetRequest(**body)
    except Exception as exc:
        return jsonify({"error": f"Invalid request: {exc}"}), 400

    try:
        obs = env.reset(req.task_id)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 404
    except Exception as exc:
        return jsonify({"error": f"Unexpected error: {exc}"}), 500

    return jsonify({"observation": obs.model_dump()}), 200


# ---------------------------------------------------------------------------
# Step
# ---------------------------------------------------------------------------

@app.route("/step", methods=["POST"])
def step():
    """Apply one action to the environment.

    Request body::

        {"action": {"action_type": "inspect_tool", "tool_id": "zoom_001"}}
    """
    if env.current_task is None:
        return jsonify({"error": "No active episode. Call /reset first."}), 400

    body = request.get_json(force=True, silent=True) or {}
    try:
        req = ActionRequest(**body)
    except Exception as exc:
        return jsonify({"error": f"Invalid request: {exc}"}), 400

    try:
        result = env.step(req.action)
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": f"Unexpected error: {exc}"}), 500

    return jsonify(result.model_dump()), 200


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@app.route("/state", methods=["GET"])
def state():
    """Return full internal state (for debugging and agent context)."""
    if env.current_task is None:
        return jsonify({"error": "No active episode. Call /reset first."}), 400
    return jsonify(env.state()), 200


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found."}), 404


@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Method not allowed."}), 405


if __name__ == "__main__":
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG)
