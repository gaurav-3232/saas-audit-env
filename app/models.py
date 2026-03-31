"""Pydantic models for SaaSAuditEnv."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, computed_field, field_validator


# ---------------------------------------------------------------------------
# Domain enums
# ---------------------------------------------------------------------------

class CriticalityLevel(str, Enum):
    """How critical a SaaS tool is to business operations."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class PlanTier(str, Enum):
    """Generic plan tiers across SaaS tools."""
    FREE = "free"
    STARTER = "starter"
    BASIC = "basic"
    PROFESSIONAL = "professional"
    BUSINESS = "business"
    ENTERPRISE = "enterprise"


# ---------------------------------------------------------------------------
# Core domain models
# ---------------------------------------------------------------------------

class Subscription(BaseModel):
    """Represents a single SaaS subscription."""

    tool_id: str = Field(..., description="Unique identifier for this subscription")
    name: str = Field(..., description="Human-readable tool name, e.g. 'Slack'")
    department: str = Field(..., description="Primary department using this tool")
    plan: PlanTier = Field(..., description="Current subscription plan tier")
    seats_purchased: int = Field(..., ge=1, description="Number of seats/licenses purchased")
    active_users: int = Field(..., ge=0, description="Number of users active in last 30 days")
    cost_per_seat_monthly: float = Field(..., ge=0.0, description="Monthly cost per seat in USD")
    criticality: CriticalityLevel = Field(..., description="Business criticality level")
    overlap_group: Optional[str] = Field(
        None,
        description="Identifier grouping tools that serve overlapping purposes"
    )
    renewal_days: Optional[int] = Field(
        None,
        ge=0,
        description="Days until subscription renewal; None if rolling monthly"
    )
    notes: str = Field(default="", description="Additional context for the agent")

    @computed_field
    @property
    def monthly_cost(self) -> float:
        """Total monthly cost for this subscription."""
        return round(self.seats_purchased * self.cost_per_seat_monthly, 2)

    @computed_field
    @property
    def wasted_seats(self) -> int:
        """Seats purchased but not actively used."""
        return max(0, self.seats_purchased - self.active_users)

    @computed_field
    @property
    def utilization_rate(self) -> float:
        """Fraction of purchased seats actively used."""
        if self.seats_purchased == 0:
            return 0.0
        return round(self.active_users / self.seats_purchased, 3)


# ---------------------------------------------------------------------------
# Action models
# ---------------------------------------------------------------------------

class ActionType(str, Enum):
    """All supported agent actions."""
    INSPECT_TOOL = "inspect_tool"
    REDUCE_SEATS = "reduce_seats"
    DOWNGRADE_PLAN = "downgrade_plan"
    CANCEL_SUBSCRIPTION = "cancel_subscription"
    MERGE_TOOLS = "merge_tools"
    SUBMIT_RECOMMENDATION = "submit_recommendation"


class InspectToolAction(BaseModel):
    """Request detailed information about a specific tool."""
    action_type: ActionType = Field(ActionType.INSPECT_TOOL, frozen=True)
    tool_id: str


class ReduceSeatsAction(BaseModel):
    """Reduce the number of purchased seats for a tool."""
    action_type: ActionType = Field(ActionType.REDUCE_SEATS, frozen=True)
    tool_id: str
    new_seat_count: int = Field(..., ge=1)


class DowngradePlanAction(BaseModel):
    """Downgrade a tool to a lower plan tier."""
    action_type: ActionType = Field(ActionType.DOWNGRADE_PLAN, frozen=True)
    tool_id: str
    target_plan: PlanTier


class CancelSubscriptionAction(BaseModel):
    """Cancel a tool's subscription entirely."""
    action_type: ActionType = Field(ActionType.CANCEL_SUBSCRIPTION, frozen=True)
    tool_id: str
    reason: str = Field(..., min_length=10)


class MergeToolsAction(BaseModel):
    """Consolidate two overlapping tools, keeping one and canceling the other."""
    action_type: ActionType = Field(ActionType.MERGE_TOOLS, frozen=True)
    keep_tool_id: str
    cancel_tool_id: str
    reason: str = Field(..., min_length=10)


class Recommendation(BaseModel):
    """A single cost-saving recommendation."""
    tool_id: str
    action: str
    estimated_monthly_savings: float = Field(..., ge=0.0)
    justification: str


class SubmitRecommendationAction(BaseModel):
    """Submit the final audit report with recommendations."""
    action_type: ActionType = Field(ActionType.SUBMIT_RECOMMENDATION, frozen=True)
    recommendations: List[Recommendation]
    total_estimated_savings: float = Field(..., ge=0.0)
    executive_summary: str = Field(..., min_length=30)


# Union type for all actions
AnyAction = (
    InspectToolAction
    | ReduceSeatsAction
    | DowngradePlanAction
    | CancelSubscriptionAction
    | MergeToolsAction
    | SubmitRecommendationAction
)


# ---------------------------------------------------------------------------
# Environment state models
# ---------------------------------------------------------------------------

class ActionRecord(BaseModel):
    """Record of a single action taken during an episode."""
    step: int
    action_type: str
    details: Dict[str, Any]
    reward: float
    success: bool
    message: str


class Observation(BaseModel):
    """What the agent sees at each step."""
    task_id: str
    goal: str
    current_step: int
    max_steps: int
    current_monthly_spend: float
    original_monthly_spend: float
    target_savings: float
    savings_achieved: float
    subscriptions: List[Subscription]
    action_history: List[ActionRecord]
    last_action_error: Optional[str]
    done: bool


class StepResult(BaseModel):
    """Returned by env.step()."""
    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any]


class EpisodeResult(BaseModel):
    """Final result after episode completion."""
    task_id: str
    total_reward: float
    score: float
    steps_taken: int
    savings_achieved: float
    target_savings: float
    savings_pct: float
    critical_tools_preserved: bool
    invalid_actions: int
    summary: str


# ---------------------------------------------------------------------------
# API request/response models
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    """Request body for /reset."""
    task_id: str = "task_easy"


class ActionRequest(BaseModel):
    """Request body for /step — raw action dict validated by the env."""
    action: Dict[str, Any]

    @field_validator("action")
    @classmethod
    def action_must_have_type(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        if "action_type" not in v:
            raise ValueError("action must include 'action_type'")
        return v


class HealthResponse(BaseModel):
    """Response for /health."""
    status: str
    version: str
    active_task: Optional[str]