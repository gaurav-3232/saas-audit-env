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
# Interactive Demo
# ---------------------------------------------------------------------------

@app.route("/demo", methods=["GET"])
def demo():
    """Interactive browser-based demo for playing the agent role."""
    import os
    template_path = os.path.join(
        os.path.dirname(__file__), "templates", "demo.html"
    )
    with open(template_path, "r") as f:
        return f.read(), 200


# ---------------------------------------------------------------------------
# Landing page
# ---------------------------------------------------------------------------

@app.route("/", methods=["GET"])
def index():
    """Landing page with environment info and API docs."""
    tasks_info = []
    for task in TASK_REGISTRY.values():
        total_spend = sum(
            s.seats_purchased * s.cost_per_seat_monthly
            for s in task.subscriptions
        )
        tasks_info.append({
            "id": task.task_id,
            "difficulty": task.difficulty,
            "tools": len(task.subscriptions),
            "spend": f"${total_spend:,.0f}",
            "target": f"${task.target_savings:,.0f}",
            "steps": task.max_steps,
        })

    html = """<!DOCTYPE html>
<html>
<head>
    <title>SaaSAuditEnv</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI',
               sans-serif; background: #0f172a; color: #e2e8f0; padding: 2rem; }
        .container { max-width: 900px; margin: 0 auto; }
        h1 { font-size: 2.2rem; margin-bottom: 0.3rem; color: #38bdf8; }
        .subtitle { color: #94a3b8; font-size: 1.1rem; margin-bottom: 2rem; }
        .status { display: inline-block; background: #059669; color: white;
                  padding: 0.25rem 0.75rem; border-radius: 1rem;
                  font-size: 0.85rem; margin-bottom: 2rem; }
        h2 { font-size: 1.3rem; margin: 1.5rem 0 0.8rem; color: #7dd3fc; }
        table { width: 100%; border-collapse: collapse; margin-bottom: 1.5rem; }
        th, td { padding: 0.6rem 1rem; text-align: left; border-bottom:
                 1px solid #1e293b; }
        th { color: #94a3b8; font-weight: 600; font-size: 0.85rem;
             text-transform: uppercase; }
        td { color: #e2e8f0; }
        .badge { display: inline-block; padding: 0.15rem 0.5rem;
                 border-radius: 0.25rem; font-size: 0.8rem; font-weight: 600; }
        .easy { background: #065f46; color: #6ee7b7; }
        .medium { background: #713f12; color: #fcd34d; }
        .medium-hard { background: #7c2d12; color: #fdba74; }
        .hard { background: #7f1d1d; color: #fca5a5; }
        .expert { background: #581c87; color: #d8b4fe; }
        code { background: #1e293b; padding: 0.15rem 0.4rem;
               border-radius: 0.25rem; font-size: 0.9rem; color: #38bdf8; }
        .endpoint { margin-bottom: 0.5rem; }
        .method { display: inline-block; width: 55px; font-weight: 700;
                  font-size: 0.8rem; }
        .get { color: #4ade80; }
        .post { color: #facc15; }
        a { color: #38bdf8; }
    </style>
</head>
<body>
    <div class="container">
        <h1>SaaSAuditEnv</h1>
        <p class="subtitle">AI agent environment for SaaS subscription
           cost optimization</p>
        <span class="status">Running — v""" + SaaSAuditEnv.VERSION + """</span>

        <h2>Tasks</h2>
        <table>
            <tr>
                <th>Task</th><th>Difficulty</th><th>Tools</th>
                <th>Monthly Spend</th><th>Savings Target</th><th>Max Steps</th>
            </tr>"""

    for t in tasks_info:
        diff_class = t["difficulty"].replace("-", "-")
        html += f"""
            <tr>
                <td><code>{t["id"]}</code></td>
                <td><span class="badge {diff_class}">{t["difficulty"]}</span></td>
                <td>{t["tools"]}</td>
                <td>{t["spend"]}/mo</td>
                <td>{t["target"]}/mo</td>
                <td>{t["steps"]}</td>
            </tr>"""

    html += """
        </table>

        <h2>API Endpoints</h2>
        <div class="endpoint">
            <span class="method get">GET</span>
            <code>/health</code> — Liveness check
        </div>
        <div class="endpoint">
            <span class="method get">GET</span>
            <code>/tasks</code> — List all tasks
        </div>
        <div class="endpoint">
            <span class="method post">POST</span>
            <code>/reset</code> — Start episode:
            <code>{"task_id": "task_easy"}</code>
        </div>
        <div class="endpoint">
            <span class="method post">POST</span>
            <code>/step</code> — Take action:
            <code>{"action": {"action_type": "..."}}</code>
        </div>
        <div class="endpoint">
            <span class="method get">GET</span>
            <code>/state</code> — Full internal state
        </div>

        <h2>Actions</h2>
        <table>
            <tr><th>Action</th><th>Description</th></tr>
            <tr><td><code>inspect_tool</code></td>
                <td>Get details about a subscription</td></tr>
            <tr><td><code>reduce_seats</code></td>
                <td>Right-size seat count (must be &ge; active users)</td></tr>
            <tr><td><code>downgrade_plan</code></td>
                <td>Downgrade to a lower plan tier</td></tr>
            <tr><td><code>cancel_subscription</code></td>
                <td>Cancel a subscription entirely</td></tr>
            <tr><td><code>merge_tools</code></td>
                <td>Consolidate overlapping tools</td></tr>
            <tr><td><code>submit_recommendation</code></td>
                <td>Submit final audit report (ends episode)</td></tr>
        </table>

        <p style="margin-top:2rem; color:#64748b; font-size:0.85rem;">
            OpenEnv Hackathon Submission &middot;
            <a href="/health">/health</a> &middot;
            <a href="/tasks">/tasks</a> &middot;
            <a href="/demo">Interactive Demo</a>
        </p>
    </div>
</body>
</html>"""
    return html, 200


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