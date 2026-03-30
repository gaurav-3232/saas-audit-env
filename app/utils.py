"""Utility functions for SaaSAuditEnv."""

from __future__ import annotations

from typing import Any, Dict

from app.models import (
    ActionType,
    AnyAction,
    CancelSubscriptionAction,
    DowngradePlanAction,
    InspectToolAction,
    MergeToolsAction,
    PlanTier,
    ReduceSeatsAction,
    SubmitRecommendationAction,
)

# Plan tier ordering for downgrade validation
PLAN_ORDER = [
    PlanTier.FREE,
    PlanTier.STARTER,
    PlanTier.BASIC,
    PlanTier.PROFESSIONAL,
    PlanTier.BUSINESS,
    PlanTier.ENTERPRISE,
]

PLAN_RANK: Dict[PlanTier, int] = {p: i for i, p in enumerate(PLAN_ORDER)}

# Approximate plan downgrade cost multipliers (relative to ENTERPRISE=1.0)
PLAN_COST_MULTIPLIER: Dict[PlanTier, float] = {
    PlanTier.FREE: 0.0,
    PlanTier.STARTER: 0.20,
    PlanTier.BASIC: 0.40,
    PlanTier.PROFESSIONAL: 0.60,
    PlanTier.BUSINESS: 0.80,
    PlanTier.ENTERPRISE: 1.0,
}


def parse_action(raw: Dict[str, Any]) -> AnyAction:
    """
    Parse and validate a raw action dictionary into the appropriate Pydantic model.

    Raises ValueError if the action_type is unknown or data is invalid.
    """
    action_type_str = raw.get("action_type", "")
    try:
        action_type = ActionType(action_type_str)
    except ValueError:
        raise ValueError(
            f"Unknown action_type '{action_type_str}'. "
            f"Valid types: {[a.value for a in ActionType]}"
        )

    parsers = {
        ActionType.INSPECT_TOOL: InspectToolAction,
        ActionType.REDUCE_SEATS: ReduceSeatsAction,
        ActionType.DOWNGRADE_PLAN: DowngradePlanAction,
        ActionType.CANCEL_SUBSCRIPTION: CancelSubscriptionAction,
        ActionType.MERGE_TOOLS: MergeToolsAction,
        ActionType.SUBMIT_RECOMMENDATION: SubmitRecommendationAction,
    }

    model_cls = parsers[action_type]
    try:
        return model_cls(**raw)
    except Exception as exc:
        raise ValueError(f"Invalid action payload for '{action_type_str}': {exc}") from exc


def is_downgrade(current_plan: PlanTier, target_plan: PlanTier) -> bool:
    """Return True if target_plan is strictly lower than current_plan."""
    return PLAN_RANK[target_plan] < PLAN_RANK[current_plan]


def plan_cost_reduction_factor(current_plan: PlanTier, target_plan: PlanTier) -> float:
    """
    Estimate the fractional cost reduction from downgrading current -> target.

    Returns a value in [0.0, 1.0] representing the fraction of current cost saved.
    """
    current_mult = PLAN_COST_MULTIPLIER[current_plan]
    target_mult = PLAN_COST_MULTIPLIER[target_plan]
    if current_mult <= 0:
        return 0.0
    return max(0.0, (current_mult - target_mult) / current_mult)
