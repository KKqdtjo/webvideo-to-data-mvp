from __future__ import annotations

from dataclasses import replace

import mujoco
import numpy as np
import pytest

from webvideo_to_data.config import CollisionConfig
from webvideo_to_data.physics_validation import (
    _longest_duration,
    observe_contacts,
    verdict_from_evidence,
)
from webvideo_to_data.scene import PandaSceneIds


def collision_config() -> CollisionConfig:
    return CollisionConfig(
        maximum_penetration_m=0.002,
        minimum_lift_m=0.05,
        maximum_target_error_m=0.04,
        settle_duration_s=1.0,
        maximum_final_tilt_rad=np.deg2rad(15.0),
        maximum_final_linear_speed_m_s=0.02,
        minimum_bilateral_contact_duration_s=0.2,
        minimum_lift_contact_duration_s=0.1,
        allowed_contact_pairs={
            "home": (("can", "table"),),
            "pregrasp": (("can", "table"),),
            "approach": (("can", "table"),),
            "close": (
                ("can", "table"),
                ("left_finger", "can"),
                ("right_finger", "can"),
            ),
            "lift": (("left_finger", "can"), ("right_finger", "can")),
            "transport": (("left_finger", "can"), ("right_finger", "can")),
            "lower": (("left_finger", "can"), ("right_finger", "can")),
            "open": (("can", "box"),),
            "retreat": (("can", "box"),),
            "settle": (("can", "box"),),
        },
    )


def contact_fixture(
    pair: tuple[str, str], distance: float
) -> tuple[mujoco.MjModel, mujoco.MjData, PandaSceneIds]:
    body_names = sorted(set(pair) | {"left_finger", "right_finger", "can", "box"})
    bodies = []
    for index, name in enumerate(body_names):
        position = "0 0 0" if name == pair[0] else f"{0.04 + distance} 0 0"
        if name not in pair:
            position = f"{1.0 + index} 0 0"
        joint = '<freejoint name="moving_pair_joint"/>' if name == pair[0] else ""
        bodies.append(
            f'<body name="{name}" pos="{position}">{joint}<geom name="geom_{index}" '
            'type="sphere" size="0.02"/></body>'
        )
    xml = (
        '<mujoco model="contact_fixture"><option gravity="0 0 0"/>'
        '<worldbody>' + "".join(bodies) + '</worldbody></mujoco>'
    )
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    body_id = lambda name: mujoco.mj_name2id(  # noqa: E731
        model, mujoco.mjtObj.mjOBJ_BODY, name
    )
    geom_id = lambda name: int(  # noqa: E731
        next(
            geom
            for geom in range(model.ngeom)
            if int(model.geom_bodyid[geom]) == body_id(name)
        )
    )
    robot_names = {
        "left_finger",
        "right_finger",
        "hand",
        "link4",
        "link5",
        "link6",
    }
    ids = PandaSceneIds(
        arm_joint_ids=(),
        arm_actuator_ids=(),
        finger_joint_ids=(-1, -1),
        finger_body_ids=(body_id("left_finger"), body_id("right_finger")),
        robot_body_ids=frozenset(
            body_id(name) for name in body_names if name in robot_names
        ),
        robot_geom_ids=frozenset(
            geom_id(name) for name in body_names if name in robot_names
        ),
        tcp_site_id=-1,
        gripper_actuator_id=-1,
        can_body_id=body_id("can"),
        can_joint_id=-1,
        can_geom_id=geom_id("can"),
        box_body_id=body_id("box"),
        box_floor_geom_id=geom_id("box"),
        box_geom_ids=frozenset({geom_id("box")}),
        table_geom_id=geom_id("table") if "table" in body_names else -1,
    )
    return model, data, ids


@pytest.mark.parametrize(
    ("phase", "pair", "allowed"),
    [
        ("close", ("left_finger", "can"), True),
        ("transport", ("right_finger", "can"), True),
        ("settle", ("can", "box"), True),
        ("home", ("can", "table"), True),
        ("transport", ("can", "table"), False),
        ("transport", ("link4", "table"), False),
        ("transport", ("hand", "box"), False),
        ("transport", ("link5", "can"), False),
        ("transport", ("link4", "link6"), False),
    ],
)
def test_contact_policy_is_phase_and_body_specific(
    phase: str, pair: tuple[str, str], allowed: bool
) -> None:
    model, data, ids = contact_fixture(pair, distance=-0.001)
    observation = observe_contacts(model, data, ids, phase, collision_config())
    assert (not observation.has_forbidden_contact) is allowed


def test_penetration_beyond_two_millimeters_is_forbidden_even_for_allowed_pair() -> None:
    model, data, ids = contact_fixture(("left_finger", "can"), distance=-0.0021)
    observation = observe_contacts(model, data, ids, "close", collision_config())
    assert observation.has_forbidden_contact
    assert observation.maximum_forbidden_penetration_m == pytest.approx(0.0021)


def test_penetration_policy_is_strict_immediately_above_exact_boundary() -> None:
    exact_model, exact_data, exact_ids = contact_fixture(
        ("left_finger", "can"), distance=-0.002
    )
    measured_penetration = max(
        -float(exact_data.contact[index].dist) for index in range(exact_data.ncon)
    )
    exact_config = replace(
        collision_config(), maximum_penetration_m=measured_penetration
    )
    exact = observe_contacts(
        exact_model, exact_data, exact_ids, "close", exact_config
    )
    assert not exact.has_forbidden_contact

    threshold = np.nextafter(measured_penetration, -np.inf)
    assert measured_penetration == np.nextafter(threshold, np.inf)
    strict_config = replace(
        collision_config(), maximum_penetration_m=threshold
    )
    observation = observe_contacts(
        exact_model, exact_data, exact_ids, "close", strict_config
    )
    assert observation.has_forbidden_contact


@pytest.mark.parametrize(("touching", "supported"), [("floor", True), ("wall", False)])
def test_box_support_requires_the_floor_geom(
    touching: str, supported: bool
) -> None:
    floor_x = 0.039 if touching == "floor" else 1.0
    wall_x = 0.039 if touching == "wall" else 1.0
    xml = f"""
    <mujoco model="box_support_fixture">
      <option gravity="0 0 0"/>
      <worldbody>
        <body name="can"><freejoint/><geom name="can_geom" type="sphere" size="0.02"/></body>
        <body name="box">
          <geom name="box_floor_geom" pos="{floor_x} 0 0" type="sphere" size="0.02"/>
          <geom name="box_wall_geom" pos="{wall_x} 0 0" type="sphere" size="0.02"/>
        </body>
        <body name="left_finger" pos="2 0 0"><geom type="sphere" size="0.02"/></body>
        <body name="right_finger" pos="3 0 0"><geom type="sphere" size="0.02"/></body>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    body = lambda name: mujoco.mj_name2id(  # noqa: E731
        model, mujoco.mjtObj.mjOBJ_BODY, name
    )
    geom = lambda name: mujoco.mj_name2id(  # noqa: E731
        model, mujoco.mjtObj.mjOBJ_GEOM, name
    )
    ids = PandaSceneIds(
        arm_joint_ids=(),
        arm_actuator_ids=(),
        finger_joint_ids=(-1, -1),
        finger_body_ids=(body("left_finger"), body("right_finger")),
        robot_body_ids=frozenset(),
        robot_geom_ids=frozenset(),
        tcp_site_id=-1,
        gripper_actuator_id=-1,
        can_body_id=body("can"),
        can_joint_id=0,
        can_geom_id=geom("can_geom"),
        box_body_id=body("box"),
        box_floor_geom_id=geom("box_floor_geom"),
        box_geom_ids=frozenset({geom("box_floor_geom"), geom("box_wall_geom")}),
        table_geom_id=-1,
    )

    observation = observe_contacts(model, data, ids, "settle", collision_config())

    assert observation.has_box_support_contact is supported


def test_multiple_contact_manifolds_are_one_observed_body_pair() -> None:
    xml = """
    <mujoco model="multi_manifold_fixture">
      <option gravity="0 0 0"/>
      <worldbody>
        <body name="can"><freejoint/>
          <geom name="can_a" pos="0 -0.01 0" type="sphere" size="0.02"/>
          <geom name="can_b" pos="0 0.01 0" type="sphere" size="0.02"/>
        </body>
        <body name="table">
          <geom name="table_a" pos="0.039 -0.01 0" type="sphere" size="0.02"/>
          <geom name="table_b" pos="0.039 0.01 0" type="sphere" size="0.02"/>
        </body>
        <body name="box" pos="2 0 0"><geom name="box_floor" type="sphere" size="0.02"/></body>
        <body name="left_finger" pos="3 0 0"><geom type="sphere" size="0.02"/></body>
        <body name="right_finger" pos="4 0 0"><geom type="sphere" size="0.02"/></body>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    body = lambda name: mujoco.mj_name2id(  # noqa: E731
        model, mujoco.mjtObj.mjOBJ_BODY, name
    )
    geom = lambda name: mujoco.mj_name2id(  # noqa: E731
        model, mujoco.mjtObj.mjOBJ_GEOM, name
    )
    ids = PandaSceneIds(
        arm_joint_ids=(),
        arm_actuator_ids=(),
        finger_joint_ids=(-1, -1),
        finger_body_ids=(body("left_finger"), body("right_finger")),
        robot_body_ids=frozenset(),
        robot_geom_ids=frozenset(),
        tcp_site_id=-1,
        gripper_actuator_id=-1,
        can_body_id=body("can"),
        can_joint_id=0,
        can_geom_id=geom("can_a"),
        box_body_id=body("box"),
        box_floor_geom_id=geom("box_floor"),
        box_geom_ids=frozenset({geom("box_floor")}),
        table_geom_id=geom("table_a"),
    )

    observation = observe_contacts(model, data, ids, "home", collision_config())

    assert data.ncon > 1
    assert observation.observed_pairs == (("can", "table"),)
    assert observation.contact_pair_count == 1


def valid_evidence() -> dict[str, object]:
    return {
        "execution_tracking_ratio": 0.95,
        "bilateral_close_contact_duration_s": 0.2,
        "bilateral_lift_contact_duration_s": 0.1,
        "maximum_lift_m": 0.05,
        "target_error_m": 0.04,
        "settle_duration_s": 1.0,
        "final_tilt_rad": np.deg2rad(15.0),
        "final_linear_speed_m_s": 0.02,
        "forbidden_contact_count": 0,
        "maximum_forbidden_penetration_m": 0.0,
        "joint_position_violation_count": 0,
        "joint_velocity_violation_count": 0,
        "joint_acceleration_violation_count": 0,
        "invalid_numerical_state": False,
    }


def test_exact_boundary_evidence_passes() -> None:
    assert verdict_from_evidence(**valid_evidence()).passed


def test_contiguous_duration_preserves_exact_configured_boundary() -> None:
    timestamps = np.cumsum(np.full(500, 0.002))
    assert _longest_duration(timestamps, np.ones(500, dtype=bool)) == 1.0


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("execution_tracking_ratio", 0.949),
        ("bilateral_close_contact_duration_s", 0.199),
        ("bilateral_lift_contact_duration_s", 0.099),
        ("maximum_lift_m", 0.0499),
        ("target_error_m", 0.0401),
        ("settle_duration_s", 0.999),
        ("final_tilt_rad", np.deg2rad(15.01)),
        ("final_linear_speed_m_s", 0.0201),
        ("forbidden_contact_count", 1),
        ("joint_velocity_violation_count", 1),
        ("joint_position_violation_count", 1),
        ("joint_acceleration_violation_count", 1),
        ("maximum_forbidden_penetration_m", 0.0021),
        ("invalid_numerical_state", True),
    ],
)
def test_each_failed_gate_has_a_specific_reason(field: str, bad_value: object) -> None:
    evidence = valid_evidence()
    evidence[field] = bad_value
    verdict = verdict_from_evidence(**evidence)
    assert not verdict.passed
    assert field in verdict.failed_checks
