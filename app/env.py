"""Core environment engine for SaaSAuditEnv.

SaaSAuditEnv follows the standard (reset, step, state) interface.
All mutations happen on deep copies of task data so the original tasks
remain pristine across episodes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

from app.graders import grade_episode
from app.models import (
    ActionRecord,
    ActionType,
    AnyAction,
    CancelSubscriptionAction,
    CriticalityLevel,
    DowngradePlanAction,
    EpisodeResult,
    InspectToolAction,
    MergeToolsAction,
    Observation,
    ReduceSeatsAction,
    StepResult,
    Subscription,
    SubmitRecommendationAction,
)
from app.tasks import TASK_REGISTRY, Task
from app.utils import is_downgrade, parse_action, plan_cost_reduction_factor


class SaaSAuditEnv:
    """
    SaaS Audit Environment.

    An AI agent interacts with this environment to audit and reduce a company's
    SaaS spending without harming business-critical operations.

    Lifecycle::

        env = SaaSAuditEnv()
        obs = env.reset("task_easy")
        while not done:
            result = env.step({"action_type": "inspect_tool", "tool_id": "zoom_001"})
            done = result.done
    """

    VERSION = "1.0.0"

    def __init__(self) -> None:
        self.current_task: Optional[Task] = None
        self._subscriptions: Dict[str, Subscription] = {}
        self._action_history: List[ActionRecord] = []
        self._current_step: int = 0
        self._done: bool = False
        self._last_error: Optional[str] = None

        # Tracking state for graders
        self.cancelled_tools: Set[str] = set()
        self.invalid_action_count: int = 0
        self.recommendation_submitted: bool = False
        self.submitted_savings_estimate: Optional[float] = None
        self._original_spend: float = 0.0
        self._inspected_tools: Set[str] = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, task_id: str) -> Observation:
        """
        Start a new episode for the given task.

        Returns the initial observation.
        """
        if task_id not in TASK_REGISTRY:
            raise ValueError(
                f"Unknown task_id '{task_id}'. "
                f"Available: {list(TASK_REGISTRY.keys())}"
            )
        task = TASK_REGISTRY[task_id]
        self.current_task = task

        # Deep copy subscriptions so mutations don't affect the registry
        self._subscriptions = {
            s.tool_id: s.model_copy(deep=True)
            for s in task.subscriptions
        }
        self._action_history = []
        self._current_step = 0
        self._done = False
        self._last_error = None
        self.cancelled_tools = set()
        self.invalid_action_count = 0
        self.recommendation_submitted = False
        self.submitted_savings_estimate = None
        self._inspected_tools = set()
        self._original_spend = self._compute_total_spend()

        return self._build_observation()

    def step(self, raw_action: Dict[str, Any]) -> StepResult:
        """
        Apply one action and return the resulting (observation, reward, done, info).
        """
        if self.current_task is None:
            raise RuntimeError("Call reset() before step().")
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        self._current_step += 1
        reward = 0.0
        info: Dict[str, Any] = {}

        # Parse and validate action
        try:
            action = parse_action(raw_action)
        except ValueError as exc:
            self.invalid_action_count += 1
            reward = -0.05
            self._last_error = str(exc)
            record = ActionRecord(
                step=self._current_step,
                action_type=raw_action.get("action_type", "unknown"),
                details=raw_action,
                reward=reward,
                success=False,
                message=str(exc),
            )
            self._action_history.append(record)
            obs = self._build_observation()
            self._check_done()
            return StepResult(observation=obs, reward=reward, done=self._done, info=info)

        self._last_error = None

        # Dispatch action
        reward, message, success = self._dispatch(action)

        record = ActionRecord(
            step=self._current_step,
            action_type=action.action_type.value,
            details=raw_action,
            reward=reward,
            success=success,
            message=message,
        )
        self._action_history.append(record)
        info["message"] = message
        info["success"] = success

        self._check_done()

        if self._done:
            score = grade_episode(self)
            info["episode_result"] = self._build_episode_result(score).model_dump()

        return StepResult(
            observation=self._build_observation(),
            reward=reward,
            done=self._done,
            info=info,
        )

    def state(self) -> Dict[str, Any]:
        """Return the full internal state as a plain dict (for debugging/API)."""
        return {
            "task_id": self.current_task.task_id if self.current_task else None,
            "current_step": self._current_step,
            "max_steps": self.current_task.max_steps if self.current_task else None,
            "done": self._done,
            "original_monthly_spend": self._original_spend,
            "current_monthly_spend": self._compute_total_spend(),
            "savings_achieved": self.total_savings_achieved,
            "target_savings": self.current_task.target_savings if self.current_task else None,
            "subscriptions": [s.model_dump() for s in self._subscriptions.values()],
            "cancelled_tools": list(self.cancelled_tools),
            "invalid_action_count": self.invalid_action_count,
            "recommendation_submitted": self.recommendation_submitted,
            "action_history": [r.model_dump() for r in self._action_history],
        }

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def action_history(self) -> List[ActionRecord]:
        return self._action_history

    @property
    def total_savings_achieved(self) -> float:
        return round(self._original_spend - self._compute_total_spend(), 2)

    # ------------------------------------------------------------------
    # Action dispatch
    # ------------------------------------------------------------------

    def _dispatch(self, action: AnyAction) -> tuple[float, str, bool]:
        """Route action to the correct handler. Returns (reward, message, success)."""
        handlers = {
            ActionType.INSPECT_TOOL: self._handle_inspect,
            ActionType.REDUCE_SEATS: self._handle_reduce_seats,
            ActionType.DOWNGRADE_PLAN: self._handle_downgrade_plan,
            ActionType.CANCEL_SUBSCRIPTION: self._handle_cancel,
            ActionType.MERGE_TOOLS: self._handle_merge,
            ActionType.SUBMIT_RECOMMENDATION: self._handle_submit,
        }
        handler = handlers[action.action_type]
        return handler(action)  # type: ignore[arg-type]

    def _handle_inspect(self, action: InspectToolAction) -> tuple[float, str, bool]:
        sub = self._subscriptions.get(action.tool_id)
        if sub is None:
            if action.tool_id in self.cancelled_tools:
                return -0.02, f"Tool '{action.tool_id}' has been cancelled.", False
            return -0.02, f"Unknown tool_id '{action.tool_id}'.", False

        # Small positive reward for first-time inspection; helps exploration
        if action.tool_id not in self._inspected_tools:
            self._inspected_tools.add(action.tool_id)
            reward = 0.02
        else:
            reward = 0.0  # No reward for re-inspecting

        msg = (
            f"Inspected '{sub.name}': "
            f"{sub.seats_purchased} seats, {sub.active_users} active users, "
            f"${sub.monthly_cost:.2f}/mo, plan={sub.plan.value}, "
            f"criticality={sub.criticality.value}, "
            f"utilization={sub.utilization_rate:.0%}, "
            f"wasted_seats={sub.wasted_seats}"
        )
        return reward, msg, True

    def _handle_reduce_seats(self, action: ReduceSeatsAction) -> tuple[float, str, bool]:
        sub = self._subscriptions.get(action.tool_id)
        if sub is None:
            return -0.05, f"Unknown tool_id '{action.tool_id}'.", False
        if action.tool_id in self.cancelled_tools:
            return -0.05, f"Tool '{action.tool_id}' is already cancelled.", False

        if action.new_seat_count < sub.active_users:
            self.invalid_action_count += 1
            return (
                -0.10,
                f"Cannot reduce seats to {action.new_seat_count}: "
                f"{sub.active_users} users are active. Minimum is {sub.active_users}.",
                False,
            )
        if action.new_seat_count >= sub.seats_purchased:
            return (
                -0.02,
                f"New seat count {action.new_seat_count} is not less than "
                f"current {sub.seats_purchased}.",
                False,
            )

        old_cost = sub.monthly_cost
        # Mutate the subscription
        self._subscriptions[action.tool_id] = sub.model_copy(
            update={"seats_purchased": action.new_seat_count}
        )
        new_cost = self._subscriptions[action.tool_id].monthly_cost
        delta = old_cost - new_cost

        # Partial progress reward proportional to savings
        reward = round(0.1 + (delta / max(1, self.current_task.target_savings)) * 0.5, 4)
        msg = (
            f"Reduced '{sub.name}' seats from {sub.seats_purchased} "
            f"to {action.new_seat_count}. Saved ${delta:.2f}/mo."
        )
        return reward, msg, True

    def _handle_downgrade_plan(self, action: DowngradePlanAction) -> tuple[float, str, bool]:
        sub = self._subscriptions.get(action.tool_id)
        if sub is None:
            return -0.05, f"Unknown tool_id '{action.tool_id}'.", False
        if action.tool_id in self.cancelled_tools:
            return -0.05, f"Tool '{action.tool_id}' is already cancelled.", False

        if not is_downgrade(sub.plan, action.target_plan):
            return (
                -0.02,
                f"Plan '{action.target_plan.value}' is not lower than "
                f"current plan '{sub.plan.value}'.",
                False,
            )

        # Penalty for downgrading critical tools
        if sub.criticality in (CriticalityLevel.CRITICAL, CriticalityLevel.HIGH):
            self.invalid_action_count += 1
            return (
                -0.15,
                f"Cannot downgrade '{sub.name}': it is {sub.criticality.value}. "
                "Downgrading critical tools risks service disruption.",
                False,
            )

        factor = plan_cost_reduction_factor(sub.plan, action.target_plan)
        old_cost = sub.monthly_cost
        new_cost_per_seat = round(sub.cost_per_seat_monthly * (1.0 - factor), 4)

        self._subscriptions[action.tool_id] = sub.model_copy(
            update={
                "plan": action.target_plan,
                "cost_per_seat_monthly": new_cost_per_seat,
            }
        )
        delta = old_cost - self._subscriptions[action.tool_id].monthly_cost
        reward = round(0.1 + (delta / max(1, self.current_task.target_savings)) * 0.4, 4)
        msg = (
            f"Downgraded '{sub.name}' from {sub.plan.value} to "
            f"{action.target_plan.value}. Saved ${delta:.2f}/mo."
        )
        return reward, msg, True

    def _handle_cancel(self, action: CancelSubscriptionAction) -> tuple[float, str, bool]:
        sub = self._subscriptions.get(action.tool_id)
        if sub is None:
            return -0.05, f"Unknown tool_id '{action.tool_id}'.", False
        if action.tool_id in self.cancelled_tools:
            return -0.05, f"Tool '{action.tool_id}' is already cancelled.", False

        # Hard block on must-preserve tools
        if action.tool_id in self.current_task.must_preserve:
            self.invalid_action_count += 1
            return (
                -0.30,
                f"Cannot cancel '{sub.name}': it is marked as must-preserve. "
                "This would break business-critical operations.",
                False,
            )

        # Penalty for cancelling critical/high tools not in must_preserve
        if sub.criticality == CriticalityLevel.CRITICAL:
            self.invalid_action_count += 1
            return (
                -0.20,
                f"Refusing to cancel '{sub.name}': criticality=CRITICAL.",
                False,
            )

        if sub.criticality == CriticalityLevel.HIGH and sub.active_users > 0:
            self.invalid_action_count += 1
            return (
                -0.15,
                f"Refusing to cancel '{sub.name}': criticality=HIGH with "
                f"{sub.active_users} active users.",
                False,
            )

        monthly_savings = sub.monthly_cost
        self.cancelled_tools.add(action.tool_id)
        del self._subscriptions[action.tool_id]

        reward = round(0.15 + (monthly_savings / max(1, self.current_task.target_savings)) * 0.5, 4)
        msg = (
            f"Cancelled '{sub.name}'. "
            f"Saved ${monthly_savings:.2f}/mo. Reason: {action.reason}"
        )
        return reward, msg, True

    def _handle_merge(self, action: MergeToolsAction) -> tuple[float, str, bool]:
        keep = self._subscriptions.get(action.keep_tool_id)
        cancel_sub = self._subscriptions.get(action.cancel_tool_id)

        if keep is None:
            return -0.05, f"Keep tool '{action.keep_tool_id}' not found.", False
        if cancel_sub is None:
            if action.cancel_tool_id in self.cancelled_tools:
                return -0.02, f"Cancel tool '{action.cancel_tool_id}' already cancelled.", False
            return -0.05, f"Cancel tool '{action.cancel_tool_id}' not found.", False

        # Validate the cancel target
        if action.cancel_tool_id in self.current_task.must_preserve:
            self.invalid_action_count += 1
            return (
                -0.30,
                f"Cannot cancel '{cancel_sub.name}' during merge: it is must-preserve.",
                False,
            )
        if cancel_sub.criticality == CriticalityLevel.CRITICAL:
            self.invalid_action_count += 1
            return (
                -0.20,
                f"Cannot cancel '{cancel_sub.name}': criticality=CRITICAL.",
                False,
            )

        # They should be in the same overlap group (soft warning, not hard block)
        in_same_group = False
        for group_tools in self.current_task.overlap_groups.values():
            if action.keep_tool_id in group_tools and action.cancel_tool_id in group_tools:
                in_same_group = True
                break

        savings = cancel_sub.monthly_cost
        self.cancelled_tools.add(action.cancel_tool_id)
        del self._subscriptions[action.cancel_tool_id]

        reward = round(
            (0.15 if in_same_group else 0.08)
            + (savings / max(1, self.current_task.target_savings)) * 0.5,
            4,
        )
        msg = (
            f"Merged: kept '{keep.name}', cancelled '{cancel_sub.name}'. "
            f"Saved ${savings:.2f}/mo."
            + ("" if in_same_group else " Warning: tools may not overlap.")
        )
        return reward, msg, True

    def _handle_submit(self, action: SubmitRecommendationAction) -> tuple[float, str, bool]:
        if self.recommendation_submitted:
            return -0.05, "Recommendation already submitted.", False

        self.recommendation_submitted = True
        self.submitted_savings_estimate = action.total_estimated_savings

        actual_savings = self.total_savings_achieved
        target = self.current_task.target_savings
        target_met = actual_savings >= target

        # Base reward for submitting
        reward = 0.15 if target_met else 0.05

        # Accuracy bonus
        if actual_savings > 0:
            ratio = action.total_estimated_savings / actual_savings
            if 0.85 <= ratio <= 1.15:
                reward += 0.10

        self._done = True

        msg = (
            f"Recommendation submitted. "
            f"Actual savings: ${actual_savings:.2f}/mo. "
            f"Target: ${target:.2f}/mo. "
            f"{'✓ Target met!' if target_met else '✗ Target not met.'}"
        )
        return reward, msg, True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_total_spend(self) -> float:
        return round(
            sum(s.monthly_cost for s in self._subscriptions.values()), 2
        )

    def _check_done(self) -> None:
        """Mark episode done if max_steps reached or recommendation submitted."""
        if self.current_task and self._current_step >= self.current_task.max_steps:
            self._done = True

    def _build_observation(self) -> Observation:
        return Observation(
            task_id=self.current_task.task_id,
            goal=self.current_task.goal,
            current_step=self._current_step,
            max_steps=self.current_task.max_steps,
            current_monthly_spend=self._compute_total_spend(),
            original_monthly_spend=self._original_spend,
            target_savings=self.current_task.target_savings,
            savings_achieved=self.total_savings_achieved,
            subscriptions=list(self._subscriptions.values()),
            action_history=list(self._action_history),
            last_action_error=self._last_error,
            done=self._done,
        )

    def _build_episode_result(self, score: float) -> EpisodeResult:
        actual = self.total_savings_achieved
        target = self.current_task.target_savings
        critical_ok = all(
            tid not in self.cancelled_tools
            for tid in self.current_task.must_preserve
        )
        return EpisodeResult(
            task_id=self.current_task.task_id,
            total_reward=round(sum(r.reward for r in self._action_history), 4),
            score=score,
            steps_taken=self._current_step,
            savings_achieved=actual,
            target_savings=target,
            savings_pct=round(actual / max(1, target) * 100, 1),
            critical_tools_preserved=critical_ok,
            invalid_actions=self.invalid_action_count,
            summary=(
                f"Task {self.current_task.task_id} complete. "
                f"Score: {score:.2%}. "
                f"Saved ${actual:.2f} of ${target:.2f} target "
                f"({'✓' if actual >= target else '✗'}). "
                f"Critical tools {'preserved' if critical_ok else 'VIOLATED'}."
            ),
        )
