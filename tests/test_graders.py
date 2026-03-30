"""Tests for deterministic graders."""

import unittest

from app.env import SaaSAuditEnv
from app.graders import grade_episode, _savings_score, _critical_tool_penalty


class TestSavingsScore(unittest.TestCase):
    def test_full_savings(self):
        self.assertAlmostEqual(_savings_score(400, 400), 1.0)

    def test_half_savings(self):
        self.assertAlmostEqual(_savings_score(200, 400), 0.5)

    def test_over_savings_capped(self):
        self.assertAlmostEqual(_savings_score(600, 400), 1.0)

    def test_zero_savings(self):
        self.assertAlmostEqual(_savings_score(0, 400), 0.0)

    def test_zero_target(self):
        self.assertAlmostEqual(_savings_score(0, 0), 1.0)


class TestCriticalToolPenalty(unittest.TestCase):
    def setUp(self):
        self.env = SaaSAuditEnv()

    def test_no_cancellations_returns_1(self):
        self.env.reset("task_easy")
        self.assertEqual(_critical_tool_penalty(self.env), 1.0)

    def test_cancelling_must_preserve_returns_0(self):
        self.env.reset("task_easy")
        # Directly inject a cancelled must-preserve tool to test grader logic
        self.env.cancelled_tools.add("slack_001")
        self.assertEqual(_critical_tool_penalty(self.env), 0.0)


class TestGradeEasyTask(unittest.TestCase):
    def setUp(self):
        self.env = SaaSAuditEnv()

    def test_zero_actions_gives_low_score(self):
        self.env.reset("task_easy")
        # Immediately submit without doing anything
        self.env.step({
            "action_type": "submit_recommendation",
            "recommendations": [],
            "total_estimated_savings": 0.0,
            "executive_summary": "No changes recommended at this time by the agent.",
        })
        score = grade_episode(self.env)
        self.assertLess(score, 0.4)

    def test_good_actions_give_higher_score(self):
        self.env.reset("task_easy")
        # Reduce zoom seats
        self.env.step({
            "action_type": "reduce_seats",
            "tool_id": "zoom_001",
            "new_seat_count": 22,
        })
        # Cancel canva
        self.env.step({
            "action_type": "cancel_subscription",
            "tool_id": "canva_001",
            "reason": "Only 2 users active; superseded by Figma in marketing stack.",
        })
        # Reduce trello seats
        self.env.step({
            "action_type": "reduce_seats",
            "tool_id": "trello_001",
            "new_seat_count": 10,
        })
        savings = self.env.total_savings_achieved
        self.env.step({
            "action_type": "submit_recommendation",
            "recommendations": [],
            "total_estimated_savings": savings,
            "executive_summary": (
                "Reduced Zoom seats to match active users, cancelled underutilised Canva, "
                "and right-sized Trello to achieve target savings."
            ),
        })
        score = grade_episode(self.env)
        self.assertGreater(score, 0.6)

    def test_cancelling_critical_tool_gives_zero_for_critical_component(self):
        self.env.reset("task_easy")
        # Directly simulate cancellation of a must-preserve tool
        self.env.cancelled_tools.add("slack_001")
        score = grade_episode(self.env)
        # Critical component (20%) should be 0, but savings component may still be 0 too
        self.assertLess(score, 0.5)

    def test_score_is_between_0_and_1(self):
        self.env.reset("task_easy")
        for _ in range(5):
            self.env.step({
                "action_type": "inspect_tool",
                "tool_id": "zoom_001",
            })
        self.env.step({
            "action_type": "submit_recommendation",
            "recommendations": [],
            "total_estimated_savings": 0.0,
            "executive_summary": "Inspected tools but no cost reductions identified.",
        })
        score = grade_episode(self.env)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


class TestGradeMediumTask(unittest.TestCase):
    def setUp(self):
        self.env = SaaSAuditEnv()

    def test_medium_score_reflects_overlap_handling(self):
        self.env.reset("task_medium")
        # Cancel basecamp (zero users, safe)
        self.env.step({
            "action_type": "cancel_subscription",
            "tool_id": "basecamp_001",
            "reason": "Zero active users; team moved to Jira entirely.",
        })
        # Cancel asana (low criticality, safe)
        self.env.step({
            "action_type": "cancel_subscription",
            "tool_id": "asana_001",
            "reason": "Only 5 remaining users who can migrate to Jira.",
        })
        # Reduce confluence seats
        self.env.step({
            "action_type": "reduce_seats",
            "tool_id": "confluence_001",
            "new_seat_count": 30,
        })
        savings = self.env.total_savings_achieved
        self.env.step({
            "action_type": "submit_recommendation",
            "recommendations": [],
            "total_estimated_savings": savings,
            "executive_summary": (
                "Cancelled redundant Asana and Basecamp to consolidate around Jira. "
                "Right-sized Confluence seats to match actual usage."
            ),
        })
        score = grade_episode(self.env)
        self.assertGreater(score, 0.55)

    def test_medium_score_in_range(self):
        self.env.reset("task_medium")
        self.env.step({
            "action_type": "submit_recommendation",
            "recommendations": [],
            "total_estimated_savings": 0.0,
            "executive_summary": "No changes recommended for the medium task.",
        })
        score = grade_episode(self.env)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


class TestGradeHardTask(unittest.TestCase):
    def setUp(self):
        self.env = SaaSAuditEnv()

    def test_hard_score_in_range(self):
        self.env.reset("task_hard")
        self.env.step({
            "action_type": "submit_recommendation",
            "recommendations": [],
            "total_estimated_savings": 0.0,
            "executive_summary": "No changes recommended for the hard task episode.",
        })
        score = grade_episode(self.env)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_hard_good_actions_score_higher(self):
        self.env.reset("task_hard")
        # Cancel zero-user tools
        self.env.step({
            "action_type": "cancel_subscription",
            "tool_id": "miro_legacy_001",
            "reason": "Zero active users, fully covered by main Miro workspace.",
        })
        self.env.step({
            "action_type": "cancel_subscription",
            "tool_id": "monday_001",
            "reason": "Zero active users, team migrated to Linear.",
        })
        self.env.step({
            "action_type": "cancel_subscription",
            "tool_id": "surveymonkey_001",
            "reason": "Only 1 active user; Typeform covers this need.",
        })
        # Reduce wasteful seats
        self.env.step({
            "action_type": "reduce_seats",
            "tool_id": "hubspot_001",
            "new_seat_count": 12,
        })
        self.env.step({
            "action_type": "reduce_seats",
            "tool_id": "zoom_002",
            "new_seat_count": 115,
        })
        self.env.step({
            "action_type": "reduce_seats",
            "tool_id": "miro_001",
            "new_seat_count": 22,
        })
        # Cancel pipedrive (low criticality, overlaps Salesforce)
        self.env.step({
            "action_type": "cancel_subscription",
            "tool_id": "pipedrive_001",
            "reason": "Only 3 users; all functionality covered by Salesforce.",
        })
        savings = self.env.total_savings_achieved
        self.env.step({
            "action_type": "submit_recommendation",
            "recommendations": [
                {
                    "tool_id": "miro_legacy_001",
                    "action": "cancel_subscription",
                    "estimated_monthly_savings": 80.0,
                    "justification": "Zero users.",
                }
            ],
            "total_estimated_savings": savings,
            "executive_summary": (
                "Cancelled four redundant/unused tools and right-sized seats across "
                "HubSpot, Zoom, and Miro to achieve target savings without impacting operations."
            ),
        })
        score = grade_episode(self.env)
        self.assertGreater(score, 0.5)


class TestGradeUnknownTask(unittest.TestCase):
    def test_unknown_task_raises(self):
        env = SaaSAuditEnv()
        env.reset("task_easy")
        env.current_task.task_id = "task_unknown"  # type: ignore
        with self.assertRaises(ValueError):
            grade_episode(env)


class TestGradeStartupTask(unittest.TestCase):
    def setUp(self):
        self.env = SaaSAuditEnv()
        self.env.reset("task_startup")

    def test_startup_score_in_range(self):
        self.env.step({
            "action_type": "submit_recommendation",
            "recommendations": [],
            "total_estimated_savings": 0.0,
            "executive_summary": "Baseline test for startup task grader validation.",
        })
        score = grade_episode(self.env)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_startup_downgrades_boost_score(self):
        # Downgrade Figma from enterprise to professional
        self.env.step({
            "action_type": "downgrade_plan",
            "tool_id": "figma_s01",
            "target_plan": "professional",
        })
        # Cancel Airtable (0 users)
        self.env.step({
            "action_type": "cancel_subscription",
            "tool_id": "airtable_s01",
            "reason": "Zero active users, team uses Notion now.",
        })
        self.env.step({
            "action_type": "submit_recommendation",
            "recommendations": [],
            "total_estimated_savings": self.env.total_savings_achieved,
            "executive_summary": "Downgraded over-provisioned plans and cancelled unused tools.",
        })
        score = grade_episode(self.env)
        self.assertGreater(score, 0.4)


class TestGradeMergerTask(unittest.TestCase):
    def setUp(self):
        self.env = SaaSAuditEnv()
        self.env.reset("task_merger")

    def test_merger_score_in_range(self):
        self.env.step({
            "action_type": "submit_recommendation",
            "recommendations": [],
            "total_estimated_savings": 0.0,
            "executive_summary": "Baseline test for merger task grader validation.",
        })
        score = grade_episode(self.env)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_merger_consolidation_boosts_score(self):
        # Consolidate several overlap groups
        self.env.step({
            "action_type": "merge_tools",
            "keep_tool_id": "slack_m01",
            "cancel_tool_id": "teams_m01",
            "reason": "Acquired company migrating to Slack.",
        })
        self.env.step({
            "action_type": "merge_tools",
            "keep_tool_id": "salesforce_m01",
            "cancel_tool_id": "pipedrive_m01",
            "reason": "Duplicate CRM, data can be migrated.",
        })
        self.env.step({
            "action_type": "cancel_subscription",
            "tool_id": "lastpass_m01",
            "reason": "Being replaced by Okta SSO, 120 unused seats.",
        })
        self.env.step({
            "action_type": "cancel_subscription",
            "tool_id": "basecamp_m01",
            "reason": "Projects migrating to Asana, 65 unused seats.",
        })
        self.env.step({
            "action_type": "cancel_subscription",
            "tool_id": "webex_m01",
            "reason": "All meetings moved to Zoom, 110 unused seats.",
        })
        savings = self.env.total_savings_achieved
        self.env.step({
            "action_type": "submit_recommendation",
            "recommendations": [],
            "total_estimated_savings": savings,
            "executive_summary": "Consolidated duplicate tools across all departments post-merger.",
        })
        score = grade_episode(self.env)
        self.assertGreater(score, 0.6)


if __name__ == "__main__":
    unittest.main()
