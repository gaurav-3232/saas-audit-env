"""Task definitions for SaaSAuditEnv.

Each task represents a realistic SaaS audit scenario at a different difficulty
level. Tasks are fully deterministic and self-contained.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from app.models import CriticalityLevel, PlanTier, Subscription


@dataclass
class Task:
    """Complete specification for one audit episode."""

    task_id: str
    difficulty: str
    goal: str
    max_steps: int
    target_savings: float  # monthly USD the agent must save
    subscriptions: List[Subscription]
    # Constraints used by graders
    must_preserve: List[str] = field(default_factory=list)   # tool_ids that MUST NOT be cancelled
    safe_to_cancel: List[str] = field(default_factory=list)  # tool_ids it is safe to cancel
    overlap_groups: Dict[str, List[str]] = field(default_factory=dict)  # group -> [tool_ids]


# ---------------------------------------------------------------------------
# Task 1 – EASY
# A small company with obvious seat waste and one redundant tool.
# ---------------------------------------------------------------------------

TASK_EASY = Task(
    task_id="task_easy",
    difficulty="easy",
    goal=(
        "Reduce monthly SaaS spend by at least $400. "
        "The company has obvious inactive seats on several tools. "
        "Identify the waste, right-size seat counts, and submit a recommendation."
    ),
    max_steps=15,
    target_savings=400.0,
    must_preserve=["slack_001", "github_001"],
    safe_to_cancel=["canva_001"],
    overlap_groups={},
    subscriptions=[
        Subscription(
            tool_id="slack_001",
            name="Slack",
            department="Engineering",
            plan=PlanTier.PROFESSIONAL,
            seats_purchased=50,
            active_users=48,
            cost_per_seat_monthly=8.75,
            criticality=CriticalityLevel.CRITICAL,
            notes="Primary team communication. Do not touch.",
        ),
        Subscription(
            tool_id="zoom_001",
            name="Zoom",
            department="All",
            plan=PlanTier.BUSINESS,
            seats_purchased=50,
            active_users=22,
            cost_per_seat_monthly=16.66,
            criticality=CriticalityLevel.HIGH,
            renewal_days=14,
            notes="28 seats unused. Consider right-sizing.",
        ),
        Subscription(
            tool_id="github_001",
            name="GitHub",
            department="Engineering",
            plan=PlanTier.BUSINESS,
            seats_purchased=30,
            active_users=29,
            cost_per_seat_monthly=19.00,
            criticality=CriticalityLevel.CRITICAL,
            notes="Core engineering tool. Must be preserved.",
        ),
        Subscription(
            tool_id="canva_001",
            name="Canva",
            department="Marketing",
            plan=PlanTier.PROFESSIONAL,
            seats_purchased=20,
            active_users=2,
            cost_per_seat_monthly=13.00,
            criticality=CriticalityLevel.LOW,
            notes="Only 2 active users. Marketing uses Figma primarily.",
        ),
        Subscription(
            tool_id="trello_001",
            name="Trello",
            department="Operations",
            plan=PlanTier.BUSINESS,
            seats_purchased=25,
            active_users=10,
            cost_per_seat_monthly=10.00,
            criticality=CriticalityLevel.MEDIUM,
            notes="15 inactive seats. Team has partially migrated to Notion.",
        ),
    ],
)


# ---------------------------------------------------------------------------
# Task 2 – MEDIUM
# Duplicate project management tools, seat waste, and one optional cancellation.
# ---------------------------------------------------------------------------

TASK_MEDIUM = Task(
    task_id="task_medium",
    difficulty="medium",
    goal=(
        "Reduce monthly SaaS spend by at least $900. "
        "The company is paying for overlapping project management tools. "
        "Consolidate duplicates, remove inactive seats, and cancel any fully redundant tools. "
        "Do not cancel tools rated CRITICAL or HIGH that have active users."
    ),
    max_steps=18,
    target_savings=900.0,
    must_preserve=["jira_001", "google_workspace_001", "pagerduty_001"],
    safe_to_cancel=["asana_001", "basecamp_001"],
    overlap_groups={
        "project_mgmt": ["jira_001", "asana_001", "basecamp_001"],
        "docs": ["confluence_001", "notion_001"],
    },
    subscriptions=[
        Subscription(
            tool_id="jira_001",
            name="Jira",
            department="Engineering",
            plan=PlanTier.BUSINESS,
            seats_purchased=60,
            active_users=57,
            cost_per_seat_monthly=8.15,
            criticality=CriticalityLevel.CRITICAL,
            overlap_group="project_mgmt",
            notes="Primary engineering tracker. Must be kept.",
        ),
        Subscription(
            tool_id="asana_001",
            name="Asana",
            department="Operations",
            plan=PlanTier.BUSINESS,
            seats_purchased=40,
            active_users=5,
            cost_per_seat_monthly=13.49,
            criticality=CriticalityLevel.LOW,
            overlap_group="project_mgmt",
            renewal_days=7,
            notes="Ops team migrated to Jira. 5 users remain but could move.",
        ),
        Subscription(
            tool_id="basecamp_001",
            name="Basecamp",
            department="Marketing",
            plan=PlanTier.BUSINESS,
            seats_purchased=15,
            active_users=0,
            cost_per_seat_monthly=11.00,
            criticality=CriticalityLevel.LOW,
            overlap_group="project_mgmt",
            notes="Zero active users. Safe to cancel.",
        ),
        Subscription(
            tool_id="confluence_001",
            name="Confluence",
            department="Engineering",
            plan=PlanTier.BUSINESS,
            seats_purchased=60,
            active_users=30,
            cost_per_seat_monthly=5.75,
            criticality=CriticalityLevel.MEDIUM,
            overlap_group="docs",
            notes="30 inactive seats. Consider reducing.",
        ),
        Subscription(
            tool_id="notion_001",
            name="Notion",
            department="All",
            plan=PlanTier.BUSINESS,
            seats_purchased=80,
            active_users=35,
            cost_per_seat_monthly=16.00,
            criticality=CriticalityLevel.MEDIUM,
            overlap_group="docs",
            notes="Overlaps with Confluence. 45 inactive seats.",
        ),
        Subscription(
            tool_id="google_workspace_001",
            name="Google Workspace",
            department="All",
            plan=PlanTier.BUSINESS,
            seats_purchased=100,
            active_users=98,
            cost_per_seat_monthly=12.00,
            criticality=CriticalityLevel.CRITICAL,
            notes="Core productivity suite. Do not touch.",
        ),
        Subscription(
            tool_id="pagerduty_001",
            name="PagerDuty",
            department="Engineering",
            plan=PlanTier.PROFESSIONAL,
            seats_purchased=15,
            active_users=14,
            cost_per_seat_monthly=21.00,
            criticality=CriticalityLevel.HIGH,
            notes="On-call alerting. Preserve.",
        ),
    ],
)


# ---------------------------------------------------------------------------
# Task 3 – HARD
# Large mixed portfolio, critical tools, tight savings target, many overlaps.
# ---------------------------------------------------------------------------

TASK_HARD = Task(
    task_id="task_hard",
    difficulty="hard",
    goal=(
        "Reduce monthly SaaS spend by at least $2,500. "
        "The company has a sprawling SaaS portfolio with duplicate tools, "
        "over-provisioned seats, and low-utilisation subscriptions across multiple departments. "
        "You must achieve the savings target while keeping all CRITICAL tools intact and "
        "not reducing any tool's active_users below its seat count after reduction. "
        "Some tools have imminent renewals — act before they renew."
    ),
    max_steps=25,
    target_savings=2500.0,
    must_preserve=[
        "salesforce_001", "aws_001", "okta_001", "slack_002",
        "github_002", "datadog_001",
    ],
    safe_to_cancel=["surveymonkey_001", "miro_legacy_001", "monday_001"],
    overlap_groups={
        "crm": ["salesforce_001", "hubspot_001", "pipedrive_001"],
        "design": ["figma_001", "miro_001", "miro_legacy_001"],
        "hr": ["bamboo_001", "workday_001"],
        "project_mgmt": ["linear_001", "monday_001", "clickup_001"],
    },
    subscriptions=[
        Subscription(
            tool_id="salesforce_001",
            name="Salesforce",
            department="Sales",
            plan=PlanTier.ENTERPRISE,
            seats_purchased=80,
            active_users=78,
            cost_per_seat_monthly=75.00,
            criticality=CriticalityLevel.CRITICAL,
            overlap_group="crm",
            notes="Primary CRM. Do not modify.",
        ),
        Subscription(
            tool_id="hubspot_001",
            name="HubSpot",
            department="Marketing",
            plan=PlanTier.PROFESSIONAL,
            seats_purchased=40,
            active_users=12,
            cost_per_seat_monthly=45.00,
            criticality=CriticalityLevel.MEDIUM,
            overlap_group="crm",
            renewal_days=10,
            notes="Marketing CRM overlapping Salesforce. 28 unused seats.",
        ),
        Subscription(
            tool_id="pipedrive_001",
            name="Pipedrive",
            department="Sales",
            plan=PlanTier.PROFESSIONAL,
            seats_purchased=20,
            active_users=3,
            cost_per_seat_monthly=32.50,
            criticality=CriticalityLevel.LOW,
            overlap_group="crm",
            notes="Legacy CRM. Only 3 users remain. Fully overlaps Salesforce.",
        ),
        Subscription(
            tool_id="slack_002",
            name="Slack",
            department="All",
            plan=PlanTier.BUSINESS,
            seats_purchased=200,
            active_users=195,
            cost_per_seat_monthly=7.25,
            criticality=CriticalityLevel.CRITICAL,
            notes="Company-wide chat. Do not touch.",
        ),
        Subscription(
            tool_id="aws_001",
            name="AWS",
            department="Engineering",
            plan=PlanTier.BUSINESS,
            seats_purchased=30,
            active_users=28,
            cost_per_seat_monthly=0.0,
            criticality=CriticalityLevel.CRITICAL,
            notes="Infrastructure. Managed separately. No seat cost.",
        ),
        Subscription(
            tool_id="figma_001",
            name="Figma",
            department="Design",
            plan=PlanTier.PROFESSIONAL,
            seats_purchased=30,
            active_users=18,
            cost_per_seat_monthly=15.00,
            criticality=CriticalityLevel.HIGH,
            overlap_group="design",
            notes="Primary design tool. 12 unused seats.",
        ),
        Subscription(
            tool_id="miro_001",
            name="Miro",
            department="Product",
            plan=PlanTier.BUSINESS,
            seats_purchased=50,
            active_users=20,
            cost_per_seat_monthly=16.00,
            criticality=CriticalityLevel.MEDIUM,
            overlap_group="design",
            notes="Used for whiteboarding. 30 inactive seats.",
        ),
        Subscription(
            tool_id="miro_legacy_001",
            name="Miro Legacy Team",
            department="Design",
            plan=PlanTier.STARTER,
            seats_purchased=10,
            active_users=0,
            cost_per_seat_monthly=8.00,
            criticality=CriticalityLevel.LOW,
            overlap_group="design",
            notes="Old Miro workspace. Zero active users. Safe to cancel.",
        ),
        Subscription(
            tool_id="github_002",
            name="GitHub Enterprise",
            department="Engineering",
            plan=PlanTier.ENTERPRISE,
            seats_purchased=100,
            active_users=97,
            cost_per_seat_monthly=21.00,
            criticality=CriticalityLevel.CRITICAL,
            notes="Core VCS. Do not modify.",
        ),
        Subscription(
            tool_id="okta_001",
            name="Okta",
            department="IT",
            plan=PlanTier.ENTERPRISE,
            seats_purchased=250,
            active_users=242,
            cost_per_seat_monthly=8.00,
            criticality=CriticalityLevel.CRITICAL,
            notes="SSO provider. Do not modify.",
        ),
        Subscription(
            tool_id="datadog_001",
            name="Datadog",
            department="Engineering",
            plan=PlanTier.PROFESSIONAL,
            seats_purchased=25,
            active_users=23,
            cost_per_seat_monthly=31.00,
            criticality=CriticalityLevel.CRITICAL,
            notes="Observability platform. Do not touch.",
        ),
        Subscription(
            tool_id="bamboo_001",
            name="BambooHR",
            department="HR",
            plan=PlanTier.PROFESSIONAL,
            seats_purchased=120,
            active_users=60,
            cost_per_seat_monthly=9.00,
            criticality=CriticalityLevel.MEDIUM,
            overlap_group="hr",
            notes="HR platform. 60 inactive seats.",
        ),
        Subscription(
            tool_id="workday_001",
            name="Workday",
            department="HR",
            plan=PlanTier.ENTERPRISE,
            seats_purchased=150,
            active_users=58,
            cost_per_seat_monthly=22.00,
            criticality=CriticalityLevel.HIGH,
            overlap_group="hr",
            notes="Payroll and HRIS. Overlaps BambooHR significantly.",
        ),
        Subscription(
            tool_id="linear_001",
            name="Linear",
            department="Engineering",
            plan=PlanTier.BUSINESS,
            seats_purchased=60,
            active_users=55,
            cost_per_seat_monthly=10.00,
            criticality=CriticalityLevel.HIGH,
            overlap_group="project_mgmt",
            notes="Engineering issue tracker. 5 unused seats.",
        ),
        Subscription(
            tool_id="monday_001",
            name="Monday.com",
            department="Operations",
            plan=PlanTier.BUSINESS,
            seats_purchased=30,
            active_users=0,
            cost_per_seat_monthly=16.00,
            criticality=CriticalityLevel.LOW,
            overlap_group="project_mgmt",
            renewal_days=5,
            notes="Zero active users. Team moved to Linear. Cancel before renewal.",
        ),
        Subscription(
            tool_id="clickup_001",
            name="ClickUp",
            department="Operations",
            plan=PlanTier.BUSINESS,
            seats_purchased=25,
            active_users=8,
            cost_per_seat_monthly=12.00,
            criticality=CriticalityLevel.LOW,
            overlap_group="project_mgmt",
            notes="Partially adopted. Overlaps Linear and Monday.",
        ),
        Subscription(
            tool_id="surveymonkey_001",
            name="SurveyMonkey",
            department="Marketing",
            plan=PlanTier.PROFESSIONAL,
            seats_purchased=10,
            active_users=1,
            cost_per_seat_monthly=25.00,
            criticality=CriticalityLevel.LOW,
            notes="Barely used. 1 active user. Marketing uses Typeform instead.",
        ),
        Subscription(
            tool_id="zoom_002",
            name="Zoom",
            department="All",
            plan=PlanTier.BUSINESS,
            seats_purchased=200,
            active_users=110,
            cost_per_seat_monthly=16.66,
            criticality=CriticalityLevel.HIGH,
            renewal_days=20,
            notes="90 unused seats. Google Meet is also available.",
        ),
    ],
)


# ---------------------------------------------------------------------------
# Task 4 – MEDIUM-HARD (Startup Scaling)
# A fast-growing startup that adopted expensive enterprise plans too early.
# The agent must downgrade plans and right-size seats across the board.
# Tests downgrade_plan action heavily, which other tasks barely exercise.
# ---------------------------------------------------------------------------

TASK_STARTUP = Task(
    task_id="task_startup",
    difficulty="medium-hard",
    goal=(
        "Reduce monthly SaaS spend by at least $1,200. "
        "This 25-person startup signed enterprise-tier plans during a funding round "
        "but most tools only need professional or basic plans. "
        "Downgrade over-provisioned plans, reduce unused seats, and cancel any tools "
        "the team has stopped using. Preserve Slack, GitHub, and Linear."
    ),
    max_steps=20,
    target_savings=1200.0,
    must_preserve=["slack_s01", "github_s01", "linear_s01"],
    safe_to_cancel=["airtable_s01", "loom_s01"],
    overlap_groups={
        "analytics": ["mixpanel_s01", "amplitude_s01"],
        "docs": ["notion_s01", "coda_s01"],
    },
    subscriptions=[
        Subscription(
            tool_id="slack_s01",
            name="Slack",
            department="All",
            plan=PlanTier.ENTERPRISE,
            seats_purchased=50,
            active_users=24,
            cost_per_seat_monthly=12.50,
            criticality=CriticalityLevel.CRITICAL,
            notes=(
                "Company-wide chat. Enterprise plan was negotiated during Series A. "
                "Only 25 employees — enterprise features unused. "
                "Can downgrade to Business but must not cancel."
            ),
        ),
        Subscription(
            tool_id="github_s01",
            name="GitHub",
            department="Engineering",
            plan=PlanTier.ENTERPRISE,
            seats_purchased=40,
            active_users=12,
            cost_per_seat_monthly=21.00,
            criticality=CriticalityLevel.CRITICAL,
            notes=(
                "Core version control. 28 seats unused — many were provisioned "
                "for contractors who left. Enterprise plan needed for SAML SSO."
            ),
        ),
        Subscription(
            tool_id="linear_s01",
            name="Linear",
            department="Engineering",
            plan=PlanTier.BUSINESS,
            seats_purchased=30,
            active_users=12,
            cost_per_seat_monthly=10.00,
            criticality=CriticalityLevel.HIGH,
            notes="Issue tracker. 18 unused seats from over-provisioning.",
        ),
        Subscription(
            tool_id="figma_s01",
            name="Figma",
            department="Design",
            plan=PlanTier.ENTERPRISE,
            seats_purchased=20,
            active_users=3,
            cost_per_seat_monthly=25.00,
            criticality=CriticalityLevel.MEDIUM,
            notes=(
                "Only 3 designers use this. Enterprise plan gives org-wide libraries "
                "but Professional plan is sufficient at this scale."
            ),
        ),
        Subscription(
            tool_id="notion_s01",
            name="Notion",
            department="All",
            plan=PlanTier.ENTERPRISE,
            seats_purchased=50,
            active_users=22,
            cost_per_seat_monthly=15.00,
            criticality=CriticalityLevel.MEDIUM,
            overlap_group="docs",
            notes=(
                "Wiki and docs. Enterprise plan has audit log and SCIM — "
                "not needed at 25 employees. 28 unused seats."
            ),
        ),
        Subscription(
            tool_id="coda_s01",
            name="Coda",
            department="Operations",
            plan=PlanTier.BUSINESS,
            seats_purchased=15,
            active_users=4,
            cost_per_seat_monthly=14.00,
            criticality=CriticalityLevel.LOW,
            overlap_group="docs",
            notes=(
                "Ops team uses Coda for runbooks but Notion covers the same use case. "
                "11 unused seats."
            ),
        ),
        Subscription(
            tool_id="mixpanel_s01",
            name="Mixpanel",
            department="Product",
            plan=PlanTier.ENTERPRISE,
            seats_purchased=20,
            active_users=5,
            cost_per_seat_monthly=28.00,
            criticality=CriticalityLevel.MEDIUM,
            overlap_group="analytics",
            notes=(
                "Product analytics. Enterprise plan signed for 2 years but team "
                "only has 5 analysts. 15 unused seats."
            ),
        ),
        Subscription(
            tool_id="amplitude_s01",
            name="Amplitude",
            department="Product",
            plan=PlanTier.PROFESSIONAL,
            seats_purchased=10,
            active_users=2,
            cost_per_seat_monthly=22.00,
            criticality=CriticalityLevel.LOW,
            overlap_group="analytics",
            notes=(
                "Secondary analytics tool. 2 users who also have Mixpanel access. "
                "Fully redundant."
            ),
        ),
        Subscription(
            tool_id="airtable_s01",
            name="Airtable",
            department="Operations",
            plan=PlanTier.BUSINESS,
            seats_purchased=15,
            active_users=0,
            cost_per_seat_monthly=20.00,
            criticality=CriticalityLevel.LOW,
            notes="Zero active users. Team moved everything to Notion. Safe to cancel.",
        ),
        Subscription(
            tool_id="loom_s01",
            name="Loom",
            department="All",
            plan=PlanTier.BUSINESS,
            seats_purchased=25,
            active_users=1,
            cost_per_seat_monthly=12.50,
            criticality=CriticalityLevel.LOW,
            notes=(
                "Video messaging tool. Only CEO still uses it occasionally. "
                "24 unused seats."
            ),
        ),
        Subscription(
            tool_id="vercel_s01",
            name="Vercel",
            department="Engineering",
            plan=PlanTier.ENTERPRISE,
            seats_purchased=15,
            active_users=8,
            cost_per_seat_monthly=30.00,
            criticality=CriticalityLevel.HIGH,
            notes=(
                "Deployment platform. Enterprise plan gives priority support and SLA "
                "but Pro plan is sufficient. 7 unused seats."
            ),
        ),
    ],
)


# ---------------------------------------------------------------------------
# Task 5 – EXPERT (Post-Acquisition Merger)
# Two companies merged. Every department has duplicate tool stacks.
# The agent must pick the right tool in each pair, consolidate, and achieve
# aggressive savings. Tests large-scale decision-making with ambiguity.
# ---------------------------------------------------------------------------

TASK_MERGER = Task(
    task_id="task_merger",
    difficulty="expert",
    goal=(
        "Reduce monthly SaaS spend by at least $4,000. "
        "After a recent acquisition, the combined company has duplicate tool stacks "
        "in every department. Each category has 2 competing tools — one from each "
        "legacy company. Choose which tool to keep in each pair based on usage data, "
        "criticality, and cost. Consolidate aggressively but preserve all tools marked "
        "CRITICAL. The board expects a clear recommendation report."
    ),
    max_steps=30,
    target_savings=4000.0,
    must_preserve=[
        "slack_m01", "aws_m01", "salesforce_m01", "github_m01", "okta_m01",
    ],
    safe_to_cancel=[
        "teams_m01", "gcp_m01", "pipedrive_m01", "gitlab_m01",
        "lastpass_m01", "basecamp_m01", "webex_m01",
    ],
    overlap_groups={
        "chat": ["slack_m01", "teams_m01"],
        "cloud": ["aws_m01", "gcp_m01"],
        "crm": ["salesforce_m01", "pipedrive_m01"],
        "vcs": ["github_m01", "gitlab_m01"],
        "identity": ["okta_m01", "lastpass_m01"],
        "project_mgmt": ["asana_m01", "basecamp_m01"],
        "video": ["zoom_m01", "webex_m01"],
    },
    subscriptions=[
        # --- Chat ---
        Subscription(
            tool_id="slack_m01",
            name="Slack (Acquirer)",
            department="All",
            plan=PlanTier.BUSINESS,
            seats_purchased=300,
            active_users=290,
            cost_per_seat_monthly=7.25,
            criticality=CriticalityLevel.CRITICAL,
            overlap_group="chat",
            notes="Acquirer's primary chat. 290 daily active users.",
        ),
        Subscription(
            tool_id="teams_m01",
            name="Microsoft Teams (Acquired)",
            department="All",
            plan=PlanTier.BUSINESS,
            seats_purchased=150,
            active_users=45,
            cost_per_seat_monthly=6.00,
            criticality=CriticalityLevel.MEDIUM,
            overlap_group="chat",
            notes=(
                "Acquired company's chat. 45 users still active but migration "
                "to Slack is underway. 105 unused seats."
            ),
        ),
        # --- Cloud ---
        Subscription(
            tool_id="aws_m01",
            name="AWS (Acquirer)",
            department="Engineering",
            plan=PlanTier.ENTERPRISE,
            seats_purchased=50,
            active_users=48,
            cost_per_seat_monthly=0.0,
            criticality=CriticalityLevel.CRITICAL,
            overlap_group="cloud",
            notes="Primary infrastructure. No per-seat cost. Do not touch.",
        ),
        Subscription(
            tool_id="gcp_m01",
            name="Google Cloud (Acquired)",
            department="Engineering",
            plan=PlanTier.BUSINESS,
            seats_purchased=30,
            active_users=5,
            cost_per_seat_monthly=0.0,
            criticality=CriticalityLevel.LOW,
            overlap_group="cloud",
            notes=(
                "Acquired company's cloud. 5 services still running but "
                "migration to AWS is planned. No per-seat cost but "
                "platform fee applies as subscription."
            ),
            # Model the platform fee as a flat subscription
        ),
        # --- CRM ---
        Subscription(
            tool_id="salesforce_m01",
            name="Salesforce (Acquirer)",
            department="Sales",
            plan=PlanTier.ENTERPRISE,
            seats_purchased=100,
            active_users=95,
            cost_per_seat_monthly=75.00,
            criticality=CriticalityLevel.CRITICAL,
            overlap_group="crm",
            notes="Primary CRM with full pipeline data. Must be preserved.",
        ),
        Subscription(
            tool_id="pipedrive_m01",
            name="Pipedrive (Acquired)",
            department="Sales",
            plan=PlanTier.PROFESSIONAL,
            seats_purchased=40,
            active_users=8,
            cost_per_seat_monthly=32.50,
            criticality=CriticalityLevel.LOW,
            overlap_group="crm",
            notes=(
                "Acquired company's CRM. 8 reps still using it but all data "
                "can be migrated to Salesforce. 32 unused seats."
            ),
        ),
        # --- VCS ---
        Subscription(
            tool_id="github_m01",
            name="GitHub Enterprise (Acquirer)",
            department="Engineering",
            plan=PlanTier.ENTERPRISE,
            seats_purchased=120,
            active_users=115,
            cost_per_seat_monthly=21.00,
            criticality=CriticalityLevel.CRITICAL,
            overlap_group="vcs",
            notes="Primary version control. All repos here.",
        ),
        Subscription(
            tool_id="gitlab_m01",
            name="GitLab (Acquired)",
            department="Engineering",
            plan=PlanTier.BUSINESS,
            seats_purchased=50,
            active_users=12,
            cost_per_seat_monthly=19.00,
            criticality=CriticalityLevel.MEDIUM,
            overlap_group="vcs",
            notes=(
                "Acquired company's VCS. 12 devs still committing. "
                "Repos being migrated to GitHub. 38 unused seats."
            ),
        ),
        # --- Identity ---
        Subscription(
            tool_id="okta_m01",
            name="Okta (Acquirer)",
            department="IT",
            plan=PlanTier.ENTERPRISE,
            seats_purchased=350,
            active_users=340,
            cost_per_seat_monthly=8.00,
            criticality=CriticalityLevel.CRITICAL,
            overlap_group="identity",
            notes="SSO provider for acquirer. Company-wide.",
        ),
        Subscription(
            tool_id="lastpass_m01",
            name="LastPass (Acquired)",
            department="IT",
            plan=PlanTier.BUSINESS,
            seats_purchased=150,
            active_users=30,
            cost_per_seat_monthly=6.00,
            criticality=CriticalityLevel.LOW,
            overlap_group="identity",
            notes=(
                "Acquired company's password manager. Being replaced by Okta "
                "SSO integration. 120 unused seats."
            ),
        ),
        # --- Project Management ---
        Subscription(
            tool_id="asana_m01",
            name="Asana (Acquirer)",
            department="All",
            plan=PlanTier.BUSINESS,
            seats_purchased=200,
            active_users=160,
            cost_per_seat_monthly=13.49,
            criticality=CriticalityLevel.HIGH,
            overlap_group="project_mgmt",
            notes=(
                "Primary project management. 40 unused seats from acquired "
                "employees not yet onboarded."
            ),
        ),
        Subscription(
            tool_id="basecamp_m01",
            name="Basecamp (Acquired)",
            department="All",
            plan=PlanTier.BUSINESS,
            seats_purchased=80,
            active_users=15,
            cost_per_seat_monthly=11.00,
            criticality=CriticalityLevel.LOW,
            overlap_group="project_mgmt",
            notes=(
                "Acquired company's PM tool. 15 users remain but "
                "projects are being migrated to Asana. 65 unused seats."
            ),
        ),
        # --- Video ---
        Subscription(
            tool_id="zoom_m01",
            name="Zoom (Acquirer)",
            department="All",
            plan=PlanTier.BUSINESS,
            seats_purchased=300,
            active_users=250,
            cost_per_seat_monthly=16.66,
            criticality=CriticalityLevel.HIGH,
            overlap_group="video",
            notes="Primary video conferencing. 50 unused seats.",
        ),
        Subscription(
            tool_id="webex_m01",
            name="Webex (Acquired)",
            department="All",
            plan=PlanTier.BUSINESS,
            seats_purchased=120,
            active_users=10,
            cost_per_seat_monthly=14.95,
            criticality=CriticalityLevel.LOW,
            overlap_group="video",
            notes=(
                "Acquired company's video tool. 10 users still on Webex "
                "but all meetings moved to Zoom. 110 unused seats."
            ),
        ),
        # --- HR (no overlap, just seat waste) ---
        Subscription(
            tool_id="workday_m01",
            name="Workday",
            department="HR",
            plan=PlanTier.ENTERPRISE,
            seats_purchased=400,
            active_users=200,
            cost_per_seat_monthly=22.00,
            criticality=CriticalityLevel.HIGH,
            notes=(
                "Combined HRIS after merger. Seats were summed from both "
                "companies but only 200 unique employees onboarded. "
                "200 unused seats."
            ),
        ),
        # --- Analytics (no overlap, just wrong plan) ---
        Subscription(
            tool_id="datadog_m01",
            name="Datadog",
            department="Engineering",
            plan=PlanTier.ENTERPRISE,
            seats_purchased=60,
            active_users=25,
            cost_per_seat_monthly=31.00,
            criticality=CriticalityLevel.HIGH,
            notes=(
                "Observability. Enterprise plan was for combined headcount "
                "estimate but actual engineering team is 25. 35 unused seats."
            ),
        ),
    ],
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

TASK_REGISTRY: Dict[str, Task] = {
    t.task_id: t
    for t in [TASK_EASY, TASK_MEDIUM, TASK_HARD, TASK_STARTUP, TASK_MERGER]
}
