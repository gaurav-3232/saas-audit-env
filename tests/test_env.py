"""Unit tests for the SaaSAuditEnv environment engine."""

import unittest

from app.env import SaaSAuditEnv


class TestEnvReset(unittest.TestCase):
    def setUp(self):
        self.env = SaaSAuditEnv()

    def test_reset_easy_returns_observation(self):
        obs = self.env.reset("task_easy")
        self.assertEqual(obs.task_id, "task_easy")
        self.assertEqual(obs.current_step, 0)
        self.assertFalse(obs.done)
        self.assertGreater(obs.current_monthly_spend, 0)

    def test_reset_medium_returns_observation(self):
        obs = self.env.reset("task_medium")
        self.assertEqual(obs.task_id, "task_medium")
        self.assertGreater(len(obs.subscriptions), 0)

    def test_reset_hard_returns_observation(self):
        obs = self.env.reset("task_hard")
        self.assertEqual(obs.task_id, "task_hard")

    def test_reset_unknown_task_raises(self):
        with self.assertRaises(ValueError):
            self.env.reset("task_does_not_exist")

    def test_reset_clears_previous_state(self):
        self.env.reset("task_easy")
        self.env.step({"action_type": "inspect_tool", "tool_id": "zoom_001"})
        obs = self.env.reset("task_easy")
        self.assertEqual(obs.current_step, 0)
        self.assertEqual(len(obs.action_history), 0)

    def test_original_spend_is_positive(self):
        obs = self.env.reset("task_easy")
        self.assertGreater(obs.original_monthly_spend, 0)
        self.assertEqual(obs.savings_achieved, 0.0)


class TestEnvInspect(unittest.TestCase):
    def setUp(self):
        self.env = SaaSAuditEnv()
        self.env.reset("task_easy")

    def test_inspect_known_tool_succeeds(self):
        result = self.env.step({"action_type": "inspect_tool", "tool_id": "zoom_001"})
        self.assertTrue(result.observation.action_history[-1].success)
        self.assertGreater(result.reward, 0)

    def test_inspect_unknown_tool_fails(self):
        result = self.env.step({"action_type": "inspect_tool", "tool_id": "nonexistent"})
        self.assertFalse(result.observation.action_history[-1].success)
        self.assertLess(result.reward, 0)

    def test_reinspect_gives_no_extra_reward(self):
        self.env.step({"action_type": "inspect_tool", "tool_id": "zoom_001"})
        result2 = self.env.step({"action_type": "inspect_tool", "tool_id": "zoom_001"})
        self.assertEqual(result2.reward, 0.0)


class TestEnvReduceSeats(unittest.TestCase):
    def setUp(self):
        self.env = SaaSAuditEnv()
        self.env.reset("task_easy")

    def test_reduce_seats_saves_money(self):
        obs_before = self.env._build_observation()
        spend_before = obs_before.current_monthly_spend
        result = self.env.step({
            "action_type": "reduce_seats",
            "tool_id": "zoom_001",
            "new_seat_count": 25,
        })
        self.assertTrue(result.observation.action_history[-1].success)
        self.assertLess(result.observation.current_monthly_spend, spend_before)

    def test_reduce_below_active_users_fails(self):
        # zoom_001 has 22 active users; reducing to 10 should fail
        result = self.env.step({
            "action_type": "reduce_seats",
            "tool_id": "zoom_001",
            "new_seat_count": 10,
        })
        self.assertFalse(result.observation.action_history[-1].success)
        self.assertLess(result.reward, 0)

    def test_reduce_to_same_count_fails(self):
        result = self.env.step({
            "action_type": "reduce_seats",
            "tool_id": "zoom_001",
            "new_seat_count": 50,
        })
        self.assertFalse(result.observation.action_history[-1].success)

    def test_savings_tracked_correctly(self):
        self.env.step({
            "action_type": "reduce_seats",
            "tool_id": "zoom_001",
            "new_seat_count": 22,  # exactly active users
        })
        obs = self.env._build_observation()
        self.assertGreater(obs.savings_achieved, 0)


class TestEnvCancelSubscription(unittest.TestCase):
    def setUp(self):
        self.env = SaaSAuditEnv()
        self.env.reset("task_easy")

    def test_cancel_low_criticality_tool_succeeds(self):
        # canva_001 is LOW criticality and in safe_to_cancel
        result = self.env.step({
            "action_type": "cancel_subscription",
            "tool_id": "canva_001",
            "reason": "Only 2 active users, fully covered by Figma.",
        })
        self.assertTrue(result.observation.action_history[-1].success)
        self.assertGreater(result.reward, 0)

    def test_cancel_critical_tool_fails(self):
        result = self.env.step({
            "action_type": "cancel_subscription",
            "tool_id": "slack_001",
            "reason": "Trying to cancel critical tool.",
        })
        self.assertFalse(result.observation.action_history[-1].success)
        self.assertLess(result.reward, 0)

    def test_cancel_must_preserve_tool_fails(self):
        result = self.env.step({
            "action_type": "cancel_subscription",
            "tool_id": "github_001",
            "reason": "Cost cutting.",
        })
        self.assertFalse(result.observation.action_history[-1].success)
        self.assertLess(result.reward, 0)

    def test_cancelled_tool_no_longer_in_subscriptions(self):
        self.env.step({
            "action_type": "cancel_subscription",
            "tool_id": "canva_001",
            "reason": "No active users, redundant with Figma subscription.",
        })
        tool_ids = [s.tool_id for s in self.env._build_observation().subscriptions]
        self.assertNotIn("canva_001", tool_ids)


class TestEnvDowngradePlan(unittest.TestCase):
    def setUp(self):
        self.env = SaaSAuditEnv()
        self.env.reset("task_easy")

    def test_downgrade_medium_criticality_succeeds(self):
        # trello_001 is MEDIUM criticality, plan=BUSINESS
        result = self.env.step({
            "action_type": "downgrade_plan",
            "tool_id": "trello_001",
            "target_plan": "starter",
        })
        self.assertTrue(result.observation.action_history[-1].success)

    def test_downgrade_critical_tool_fails(self):
        result = self.env.step({
            "action_type": "downgrade_plan",
            "tool_id": "slack_001",
            "target_plan": "starter",
        })
        self.assertFalse(result.observation.action_history[-1].success)

    def test_upgrade_fails(self):
        # trello_001 is BUSINESS; trying to go to ENTERPRISE should fail
        result = self.env.step({
            "action_type": "downgrade_plan",
            "tool_id": "trello_001",
            "target_plan": "enterprise",
        })
        self.assertFalse(result.observation.action_history[-1].success)


class TestEnvMergeTools(unittest.TestCase):
    def setUp(self):
        self.env = SaaSAuditEnv()
        self.env.reset("task_medium")

    def test_merge_overlapping_tools_succeeds(self):
        # Keep jira_001, cancel asana_001 (same overlap group)
        result = self.env.step({
            "action_type": "merge_tools",
            "keep_tool_id": "jira_001",
            "cancel_tool_id": "asana_001",
            "reason": "Asana is redundant with Jira; migrating remaining users.",
        })
        self.assertTrue(result.observation.action_history[-1].success)
        tool_ids = [s.tool_id for s in result.observation.subscriptions]
        self.assertNotIn("asana_001", tool_ids)
        self.assertIn("jira_001", tool_ids)

    def test_merge_cancelling_must_preserve_fails(self):
        result = self.env.step({
            "action_type": "merge_tools",
            "keep_tool_id": "asana_001",
            "cancel_tool_id": "jira_001",
            "reason": "Trying to cancel must-preserve tool.",
        })
        self.assertFalse(result.observation.action_history[-1].success)


class TestEnvSubmitRecommendation(unittest.TestCase):
    def setUp(self):
        self.env = SaaSAuditEnv()
        self.env.reset("task_easy")

    def test_submit_ends_episode(self):
        result = self.env.step({
            "action_type": "submit_recommendation",
            "recommendations": [
                {
                    "tool_id": "zoom_001",
                    "action": "reduce_seats",
                    "estimated_monthly_savings": 100.0,
                    "justification": "28 unused seats.",
                }
            ],
            "total_estimated_savings": 100.0,
            "executive_summary": "Reduce seats on underutilised tools to cut spend.",
        })
        self.assertTrue(result.done)

    def test_double_submit_penalised(self):
        self.env.reset("task_easy")
        self.env.step({
            "action_type": "submit_recommendation",
            "recommendations": [],
            "total_estimated_savings": 0.0,
            "executive_summary": "No changes recommended at this time by agent.",
        })
        # Episode is done; second step should raise
        with self.assertRaises(RuntimeError):
            self.env.step({
                "action_type": "submit_recommendation",
                "recommendations": [],
                "total_estimated_savings": 0.0,
                "executive_summary": "Second submission attempt by the agent.",
            })


class TestEnvMaxSteps(unittest.TestCase):
    def test_episode_ends_at_max_steps(self):
        env = SaaSAuditEnv()
        env.reset("task_easy")
        task_max = env.current_task.max_steps

        for _ in range(task_max - 1):
            result = env.step({"action_type": "inspect_tool", "tool_id": "zoom_001"})
            if result.done:
                break

        result = env.step({"action_type": "inspect_tool", "tool_id": "zoom_001"})
        self.assertTrue(result.done)


class TestEnvInvalidActions(unittest.TestCase):
    def setUp(self):
        self.env = SaaSAuditEnv()
        self.env.reset("task_easy")

    def test_unknown_action_type_penalised(self):
        result = self.env.step({"action_type": "fly_to_moon"})
        self.assertLess(result.reward, 0)
        self.assertFalse(result.observation.action_history[-1].success)

    def test_missing_action_type_penalised(self):
        result = self.env.step({"tool_id": "zoom_001"})
        self.assertLess(result.reward, 0)

    def test_step_before_reset_raises(self):
        env2 = SaaSAuditEnv()
        with self.assertRaises(RuntimeError):
            env2.step({"action_type": "inspect_tool", "tool_id": "zoom_001"})


class TestEnvStartupTask(unittest.TestCase):
    """Tests for task_startup."""

    def setUp(self):
        self.env = SaaSAuditEnv()
        self.env.reset("task_startup")

    def test_reset_startup_returns_observation(self):
        obs = self.env.reset("task_startup")
        self.assertEqual(obs.task_id, "task_startup")
        self.assertEqual(len(obs.subscriptions), 11)
        self.assertGreater(obs.current_monthly_spend, 0)

    def test_downgrade_medium_tool_succeeds(self):
        result = self.env.step({
            "action_type": "downgrade_plan",
            "tool_id": "figma_s01",
            "target_plan": "professional",
        })
        self.assertTrue(result.info.get("success"))
        self.assertGreater(result.reward, 0)

    def test_cancel_airtable_succeeds(self):
        result = self.env.step({
            "action_type": "cancel_subscription",
            "tool_id": "airtable_s01",
            "reason": "Zero active users, team migrated to Notion",
        })
        self.assertTrue(result.info.get("success"))
        self.assertIn("airtable_s01", self.env.cancelled_tools)


class TestEnvMergerTask(unittest.TestCase):
    """Tests for task_merger."""

    def setUp(self):
        self.env = SaaSAuditEnv()
        self.env.reset("task_merger")

    def test_reset_merger_returns_observation(self):
        obs = self.env.reset("task_merger")
        self.assertEqual(obs.task_id, "task_merger")
        self.assertEqual(len(obs.subscriptions), 16)

    def test_merge_chat_tools_succeeds(self):
        result = self.env.step({
            "action_type": "merge_tools",
            "keep_tool_id": "slack_m01",
            "cancel_tool_id": "teams_m01",
            "reason": "Acquired company migrating to Slack, only 45 users remain on Teams",
        })
        self.assertTrue(result.info.get("success"))
        self.assertGreater(result.reward, 0)

    def test_cancel_must_preserve_fails(self):
        result = self.env.step({
            "action_type": "cancel_subscription",
            "tool_id": "salesforce_m01",
            "reason": "Testing must-preserve constraint",
        })
        self.assertFalse(result.info.get("success"))
        self.assertLess(result.reward, 0)

    def test_seven_overlap_groups_defined(self):
        self.assertEqual(len(self.env.current_task.overlap_groups), 7)


if __name__ == "__main__":
    unittest.main()
