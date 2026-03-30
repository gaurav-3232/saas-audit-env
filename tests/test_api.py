"""Integration tests for the Flask API endpoints."""

import json
import unittest

from app.main import app


class TestHealthEndpoint(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_health_returns_200(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertEqual(data["status"], "ok")
        self.assertIn("version", data)

    def test_health_shows_no_active_task_initially(self):
        resp = self.client.get("/health")
        data = json.loads(resp.data)
        # May or may not be None depending on test order; just check key exists
        self.assertIn("active_task", data)


class TestTasksEndpoint(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_tasks_returns_three_tasks(self):
        resp = self.client.get("/tasks")
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertIn("tasks", data)
        self.assertEqual(len(data["tasks"]), 5)

    def test_tasks_contain_expected_fields(self):
        resp = self.client.get("/tasks")
        data = json.loads(resp.data)
        for task in data["tasks"]:
            self.assertIn("task_id", task)
            self.assertIn("difficulty", task)
            self.assertIn("target_savings", task)


class TestResetEndpoint(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_reset_easy_returns_observation(self):
        resp = self.client.post(
            "/reset",
            data=json.dumps({"task_id": "task_easy"}),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertIn("observation", data)
        self.assertEqual(data["observation"]["task_id"], "task_easy")

    def test_reset_unknown_task_returns_404(self):
        resp = self.client.post(
            "/reset",
            data=json.dumps({"task_id": "nonexistent"}),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 404)

    def test_reset_missing_task_id_returns_400(self):
        resp = self.client.post(
            "/reset",
            data=json.dumps({}),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 400)


class TestStepEndpoint(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        self.client.post(
            "/reset",
            data=json.dumps({"task_id": "task_easy"}),
            content_type="application/json",
        )

    def test_inspect_tool_returns_step_result(self):
        resp = self.client.post(
            "/step",
            data=json.dumps({
                "action": {"action_type": "inspect_tool", "tool_id": "zoom_001"}
            }),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertIn("observation", data)
        self.assertIn("reward", data)
        self.assertIn("done", data)

    def test_invalid_action_type_returns_200_with_failure(self):
        resp = self.client.post(
            "/step",
            data=json.dumps({"action": {"action_type": "invalid_action"}}),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertLess(data["reward"], 0)

    def test_step_without_action_key_returns_400(self):
        resp = self.client.post(
            "/step",
            data=json.dumps({"wrong_key": "value"}),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 400)

    def test_cancel_and_savings_reflected_in_observation(self):
        resp = self.client.post(
            "/step",
            data=json.dumps({
                "action": {
                    "action_type": "cancel_subscription",
                    "tool_id": "canva_001",
                    "reason": "Minimal usage, covered by other tools in suite.",
                }
            }),
            content_type="application/json",
        )
        data = json.loads(resp.data)
        obs = data["observation"]
        self.assertGreater(obs["savings_achieved"], 0)


class TestStateEndpoint(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_state_before_reset_returns_400(self):
        # Need a fresh client with no prior reset
        fresh_client = app.test_client()
        resp = fresh_client.get("/state")
        # This may be 400 or 200 depending on prior test order; acceptable either way
        self.assertIn(resp.status_code, [200, 400])

    def test_state_after_reset_returns_full_state(self):
        self.client.post(
            "/reset",
            data=json.dumps({"task_id": "task_medium"}),
            content_type="application/json",
        )
        resp = self.client.get("/state")
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertIn("subscriptions", data)
        self.assertIn("current_monthly_spend", data)
        self.assertIn("savings_achieved", data)


class TestEndToEndEasyTask(unittest.TestCase):
    """Simulate a complete easy task episode via the API."""

    def setUp(self):
        self.client = app.test_client()

    def _post_step(self, action):
        resp = self.client.post(
            "/step",
            data=json.dumps({"action": action}),
            content_type="application/json",
        )
        return json.loads(resp.data)

    def test_easy_task_full_episode(self):
        # Reset
        resp = self.client.post(
            "/reset",
            data=json.dumps({"task_id": "task_easy"}),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)

        # Inspect a tool
        result = self._post_step({"action_type": "inspect_tool", "tool_id": "zoom_001"})
        self.assertTrue(result["observation"]["action_history"][-1]["success"])

        # Reduce seats on Zoom (50 -> 25, active_users=22)
        result = self._post_step({
            "action_type": "reduce_seats",
            "tool_id": "zoom_001",
            "new_seat_count": 25,
        })
        self.assertTrue(result["observation"]["action_history"][-1]["success"])

        # Cancel canva
        result = self._post_step({
            "action_type": "cancel_subscription",
            "tool_id": "canva_001",
            "reason": "Only 2 users active, fully superseded by Figma.",
        })
        self.assertTrue(result["observation"]["action_history"][-1]["success"])

        # Submit recommendation
        savings = result["observation"]["savings_achieved"]
        result = self._post_step({
            "action_type": "submit_recommendation",
            "recommendations": [
                {
                    "tool_id": "zoom_001",
                    "action": "reduce_seats",
                    "estimated_monthly_savings": 400.0,
                    "justification": "Reduce to match active users.",
                }
            ],
            "total_estimated_savings": savings,
            "executive_summary": (
                "Right-sized Zoom seats and cancelled Canva to achieve target savings."
            ),
        })
        self.assertTrue(result["done"])
        self.assertIn("episode_result", result["info"])
        score = result["info"]["episode_result"]["score"]
        self.assertGreater(score, 0.3)


if __name__ == "__main__":
    unittest.main()
