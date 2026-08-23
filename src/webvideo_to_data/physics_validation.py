"""Phase-aware MuJoCo contact policy and conservative rollout verdicts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco
import numpy as np

from .config import CollisionConfig
from .scene import PandaSceneIds

if TYPE_CHECKING:
    from .simulation import SimulationResult


class PhysicsRolloutFailure(RuntimeError):
    """A finite rollout failed one or more conservative physical gates."""

    def __init__(self, failed_checks: tuple[str, ...]) -> None:
        self.failed_checks = failed_checks
        super().__init__("physics rollout failed: " + ", ".join(failed_checks))


class InvalidNumericalStateError(RuntimeError):
    """The rollout produced its first non-finite measured state."""

    def __init__(self, timestamp_s: float) -> None:
        self.timestamp_s = timestamp_s
        super().__init__(f"non-finite measured state at {timestamp_s:.9g} s")


@dataclass(frozen=True)
class ContactObservation:
    bilateral_fingertip_can_contact: bool
    has_box_support_contact: bool
    has_forbidden_contact: bool
    forbidden_pairs: tuple[tuple[str, str], ...]
    maximum_forbidden_penetration_m: float
    maximum_penetration_m: float = 0.0
    observed_pairs: tuple[tuple[str, str], ...] = ()
    contact_pair_count: int = 0


@dataclass(frozen=True)
class PhysicsValidationResult:
    passed: bool
    failed_checks: tuple[str, ...]
    forbidden_contact_count: int
    maximum_forbidden_penetration_m: float


def _is_descendant(
    model: mujoco.MjModel, body_id: int, ancestor_body_id: int
) -> bool:
    if ancestor_body_id < 0:
        return False
    current = body_id
    while current > 0:
        if current == ancestor_body_id:
            return True
        current = int(model.body_parentid[current])
    return False


def _contact_body_name(
    model: mujoco.MjModel, geom_id: int, ids: PandaSceneIds
) -> str:
    if geom_id == ids.table_geom_id:
        return "table"
    body_id = int(model.geom_bodyid[geom_id])
    if _is_descendant(model, body_id, ids.can_body_id):
        return "can"
    if _is_descendant(model, body_id, ids.box_body_id):
        return "box"
    for finger_name, finger_body_id in zip(
        ("left_finger", "right_finger"), ids.finger_body_ids
    ):
        if _is_descendant(model, body_id, finger_body_id):
            return finger_name

    current = body_id
    while current > 0:
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, current)
        if name:
            return name
        current = int(model.body_parentid[current])
    return "world"


def _pair(left: str, right: str) -> tuple[str, str]:
    return tuple(sorted((left, right)))  # type: ignore[return-value]


def observe_contacts(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ids: PandaSceneIds,
    phase: str,
    collision_config: CollisionConfig,
) -> ContactObservation:
    """Classify real ``data.contact`` pairs by body ancestry and phase."""

    allowed = {
        _pair(left, right)
        for left, right in collision_config.allowed_contact_pairs.get(phase, ())
    }
    observed: dict[tuple[str, str], float] = {}
    box_floor_support = False
    for index in range(data.ncon):
        contact = data.contact[index]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        name1 = _contact_body_name(model, geom1, ids)
        name2 = _contact_body_name(model, geom2, ids)
        names = _pair(name1, name2)
        box_floor_support = box_floor_support or (
            (name1 == "can" and geom2 == ids.box_floor_geom_id)
            or (name2 == "can" and geom1 == ids.box_floor_geom_id)
        )
        penetration = max(0.0, -float(contact.dist))
        observed[names] = max(observed.get(names, 0.0), penetration)

    forbidden = {
        names: penetration
        for names, penetration in observed.items()
        if names not in allowed
        or penetration > collision_config.maximum_penetration_m
    }
    observed_pairs = tuple(sorted(observed))
    left_can = _pair("left_finger", "can") in observed
    right_can = _pair("right_finger", "can") in observed
    return ContactObservation(
        bilateral_fingertip_can_contact=left_can and right_can,
        has_box_support_contact=box_floor_support,
        has_forbidden_contact=bool(forbidden),
        forbidden_pairs=tuple(sorted(forbidden)),
        maximum_forbidden_penetration_m=max(forbidden.values(), default=0.0),
        maximum_penetration_m=max(observed.values(), default=0.0),
        observed_pairs=observed_pairs,
        contact_pair_count=len(observed_pairs),
    )


def _verdict(
    *,
    execution_tracking_ratio: float,
    bilateral_close_contact_duration_s: float,
    bilateral_lift_contact_duration_s: float,
    maximum_lift_m: float,
    target_error_m: float,
    settle_duration_s: float,
    final_tilt_rad: float,
    final_linear_speed_m_s: float,
    forbidden_contact_count: int,
    maximum_forbidden_penetration_m: float,
    joint_position_violation_count: int,
    joint_velocity_violation_count: int,
    joint_acceleration_violation_count: int,
    invalid_numerical_state: bool,
    collision_config: CollisionConfig,
) -> PhysicsValidationResult:
    failed: list[str] = []
    gates = (
        ("execution_tracking_ratio", execution_tracking_ratio >= 0.95),
        (
            "bilateral_close_contact_duration_s",
            bilateral_close_contact_duration_s
            >= collision_config.minimum_bilateral_contact_duration_s,
        ),
        (
            "bilateral_lift_contact_duration_s",
            bilateral_lift_contact_duration_s
            >= collision_config.minimum_lift_contact_duration_s,
        ),
        ("maximum_lift_m", maximum_lift_m >= collision_config.minimum_lift_m),
        (
            "target_error_m",
            target_error_m <= collision_config.maximum_target_error_m,
        ),
        ("settle_duration_s", settle_duration_s >= collision_config.settle_duration_s),
        ("final_tilt_rad", final_tilt_rad <= collision_config.maximum_final_tilt_rad),
        (
            "final_linear_speed_m_s",
            final_linear_speed_m_s
            <= collision_config.maximum_final_linear_speed_m_s,
        ),
        ("forbidden_contact_count", forbidden_contact_count == 0),
        (
            "maximum_forbidden_penetration_m",
            maximum_forbidden_penetration_m
            <= collision_config.maximum_penetration_m,
        ),
        (
            "joint_position_violation_count",
            joint_position_violation_count == 0,
        ),
        (
            "joint_velocity_violation_count",
            joint_velocity_violation_count == 0,
        ),
        (
            "joint_acceleration_violation_count",
            joint_acceleration_violation_count == 0,
        ),
        ("invalid_numerical_state", not invalid_numerical_state),
    )
    failed.extend(name for name, passed in gates if not passed)
    return PhysicsValidationResult(
        passed=not failed,
        failed_checks=tuple(failed),
        forbidden_contact_count=int(forbidden_contact_count),
        maximum_forbidden_penetration_m=float(maximum_forbidden_penetration_m),
    )


def verdict_from_evidence(
    *,
    execution_tracking_ratio: float,
    bilateral_close_contact_duration_s: float,
    bilateral_lift_contact_duration_s: float,
    maximum_lift_m: float,
    target_error_m: float,
    settle_duration_s: float,
    final_tilt_rad: float,
    final_linear_speed_m_s: float,
    forbidden_contact_count: int,
    maximum_forbidden_penetration_m: float,
    joint_position_violation_count: int,
    joint_velocity_violation_count: int,
    joint_acceleration_violation_count: int,
    invalid_numerical_state: bool,
) -> PhysicsValidationResult:
    """Apply the locked EXP-001 physical gates to explicit evidence."""

    config = CollisionConfig(
        maximum_penetration_m=0.002,
        minimum_lift_m=0.05,
        maximum_target_error_m=0.04,
        settle_duration_s=1.0,
        maximum_final_tilt_rad=float(np.deg2rad(15.0)),
        maximum_final_linear_speed_m_s=0.02,
        minimum_bilateral_contact_duration_s=0.2,
        minimum_lift_contact_duration_s=0.1,
        allowed_contact_pairs={},
    )
    return _verdict(
        execution_tracking_ratio=execution_tracking_ratio,
        bilateral_close_contact_duration_s=bilateral_close_contact_duration_s,
        bilateral_lift_contact_duration_s=bilateral_lift_contact_duration_s,
        maximum_lift_m=maximum_lift_m,
        target_error_m=target_error_m,
        settle_duration_s=settle_duration_s,
        final_tilt_rad=final_tilt_rad,
        final_linear_speed_m_s=final_linear_speed_m_s,
        forbidden_contact_count=forbidden_contact_count,
        maximum_forbidden_penetration_m=maximum_forbidden_penetration_m,
        joint_position_violation_count=joint_position_violation_count,
        joint_velocity_violation_count=joint_velocity_violation_count,
        joint_acceleration_violation_count=joint_acceleration_violation_count,
        invalid_numerical_state=invalid_numerical_state,
        collision_config=config,
    )


def _longest_duration(
    timestamps_s: np.ndarray, selected: np.ndarray
) -> float:
    if not len(timestamps_s):
        return 0.0
    timestep = (
        float(np.median(np.diff(timestamps_s))) if len(timestamps_s) > 1 else 0.0
    )
    longest = current = 0
    for value in selected:
        current = current + 1 if bool(value) else 0
        longest = max(longest, current)
    return round(longest * timestep, 12)


def validate_rollout(
    simulation: SimulationResult, collision_config: CollisionConfig
) -> PhysicsValidationResult:
    """Derive the conservative physical verdict from measured rollout arrays."""

    timestamps = np.asarray(simulation.timestamps_s)
    phases = np.asarray(simulation.phase)
    bilateral = np.asarray(simulation.bilateral_contact, dtype=bool)
    close_duration = _longest_duration(
        timestamps, bilateral & (phases == "close")
    )
    lift_duration = _longest_duration(
        timestamps, bilateral & (phases == "lift")
    )
    settle_duration = _longest_duration(
        timestamps,
        np.asarray(simulation.box_support_contact, dtype=bool)
        & (phases == "settle"),
    )
    forbidden = np.asarray(simulation.forbidden_contact, dtype=bool)
    penetration = np.asarray(simulation.maximum_penetration_m, dtype=float)
    tracking = np.asarray(simulation.tcp_position_within_tolerance, dtype=bool) & np.asarray(
        simulation.tcp_orientation_within_tolerance, dtype=bool
    )
    return _verdict(
        execution_tracking_ratio=float(np.mean(tracking)) if len(tracking) else 0.0,
        bilateral_close_contact_duration_s=close_duration,
        bilateral_lift_contact_duration_s=lift_duration,
        maximum_lift_m=float(simulation.maximum_lift_m),
        target_error_m=float(simulation.target_error_m),
        settle_duration_s=settle_duration,
        final_tilt_rad=float(simulation.can_tilt_rad[-1]),
        final_linear_speed_m_s=float(simulation.can_linear_speed_m_s[-1]),
        forbidden_contact_count=int(np.count_nonzero(forbidden)),
        maximum_forbidden_penetration_m=float(
            np.max(penetration[forbidden]) if np.any(forbidden) else 0.0
        ),
        joint_position_violation_count=int(
            np.count_nonzero(simulation.joint_position_violation)
        ),
        joint_velocity_violation_count=int(
            np.count_nonzero(simulation.joint_velocity_violation)
        ),
        joint_acceleration_violation_count=int(
            np.count_nonzero(simulation.joint_acceleration_violation)
        ),
        invalid_numerical_state=not bool(
            np.all(simulation.valid_numerical_state)
        ),
        collision_config=collision_config,
    )
