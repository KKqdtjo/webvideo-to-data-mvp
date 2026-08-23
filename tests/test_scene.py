"""Public model contract for the pinned EXP-001 Panda scene."""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from webvideo_to_data.scene import (
    ScenePerturbation,
    apply_scene_perturbation,
    load_panda_scene,
)


ASSET_ROOT = (
    Path(__file__).parents[1]
    / "src"
    / "webvideo_to_data"
    / "assets"
    / "mujoco_menagerie"
    / "franka_emika_panda"
)
UPSTREAM_PATH = ASSET_ROOT / "UPSTREAM.json"
LICENSE_PATH = ASSET_ROOT / "LICENSE"
TCP_SITE_LINE = (
    b'                      <site name="panda_tcp" pos="0 0 0.1034" '
    b'size="0.005" rgba="0 1 0 1"/>\n'
)


def test_pinned_panda_scene_has_required_dynamics_and_collision_geometry() -> None:
    """Would catch replacing the official collision-capable Panda model."""
    model, data, ids = load_panda_scene()

    assert len(ids.arm_joint_ids) == 7
    assert len(ids.arm_actuator_ids) == 7
    assert len(ids.finger_body_ids) == 2
    assert model.site(ids.tcp_site_id).name == "panda_tcp"
    robot_geoms = [geom for geom in ids.robot_geom_ids if model.geom_group[geom] == 3]
    assert robot_geoms
    assert all(model.geom_contype[geom] != 0 for geom in robot_geoms)
    assert model.body("can").id == ids.can_body_id
    assert model.body("box").id == ids.box_body_id
    assert model.geom("table_geom").id == ids.table_geom_id
    assert np.isfinite(data.qpos).all()


def test_upstream_metadata_pins_commit_and_license() -> None:
    """Would catch an unpinned source snapshot or omitted upstream license."""
    metadata = json.loads(UPSTREAM_PATH.read_text(encoding="utf-8"))

    assert metadata["commit"] == "da76818e269b82289eba39808e2fb91d679d6994"
    assert metadata["source"] == (
        "https://github.com/google-deepmind/mujoco_menagerie/tree/"
        "da76818e269b82289eba39808e2fb91d679d6994/franka_emika_panda"
    )
    assert metadata["license"] == "Apache-2.0"
    assert LICENSE_PATH.read_text(encoding="utf-8").lstrip().startswith("Apache License")


def test_scene_perturbation_updates_free_can_and_contact_properties() -> None:
    """Would catch B0 perturbations being ignored by the physical scene."""
    model, data, ids = load_panda_scene()
    qpos_before = data.qpos.copy()
    mass_before = model.body_mass[ids.can_body_id]
    friction_before = model.geom_friction[ids.can_geom_id].copy()

    apply_scene_perturbation(
        model,
        data,
        ids,
        ScenePerturbation(
            can_dx_m=0.01,
            can_dy_m=-0.02,
            can_yaw_rad=np.pi / 2,
            mass_scale=1.1,
            friction_scale=0.9,
        ),
    )

    qpos_address = model.jnt_qposadr[ids.can_joint_id]
    assert data.qpos[qpos_address : qpos_address + 3] == pytest.approx(
        qpos_before[qpos_address : qpos_address + 3] + [0.01, -0.02, 0.0]
    )
    assert data.qpos[qpos_address + 3 : qpos_address + 7] == pytest.approx(
        [np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)]
    )
    assert model.body_mass[ids.can_body_id] == pytest.approx(mass_before * 1.1)
    assert model.geom_friction[ids.can_geom_id] == pytest.approx(friction_before * 0.9)


def test_perturbation_preserves_non_can_state_when_model_constants_are_rebuilt() -> None:
    """Would catch mj_setConst resetting caller state while applying a perturbation."""
    model, data, ids = load_panda_scene()
    data.qpos[:] = np.linspace(-0.2, 0.2, model.nq)
    data.qvel[:] = np.linspace(-0.3, 0.3, model.nv)
    data.ctrl[:] = np.linspace(-0.4, 0.4, model.nu)
    qpos_before = data.qpos.copy()
    qvel_before = data.qvel.copy()
    ctrl_before = data.ctrl.copy()

    apply_scene_perturbation(
        model, data, ids, ScenePerturbation(0.01, -0.02, 0.1, 1.1, 1.1)
    )

    qpos_address = int(model.jnt_qposadr[ids.can_joint_id])
    non_can_qpos = np.ones(model.nq, dtype=bool)
    non_can_qpos[qpos_address : qpos_address + 7] = False
    assert data.qpos[non_can_qpos] == pytest.approx(qpos_before[non_can_qpos])
    assert data.qvel == pytest.approx(qvel_before)
    assert data.ctrl == pytest.approx(ctrl_before)


def _can_table_contact_friction(scale: float) -> float:
    model, data, ids = load_panda_scene()
    apply_scene_perturbation(model, data, ids, ScenePerturbation(0.0, 0.0, 0.0, 1.0, scale))
    qpos_address = int(model.jnt_qposadr[ids.can_joint_id])
    data.qpos[qpos_address + 2] -= 0.001
    mujoco.mj_forward(model, data)
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        if {contact.geom1, contact.geom2} == {ids.can_geom_id, ids.table_geom_id}:
            return float(contact.friction[0])
    raise AssertionError("expected a can-table contact")


def test_perturbation_scales_real_can_table_contact_friction_in_both_directions() -> None:
    """Would catch lower friction scales being masked by the table's max rule."""
    assert _can_table_contact_friction(0.9) == pytest.approx(0.72)
    assert _can_table_contact_friction(1.1) == pytest.approx(0.88)


def test_perturbation_scales_can_mass_and_inertia_together() -> None:
    """Would catch a mass-only change that no longer represents a fixed rigid shape."""
    model, data, ids = load_panda_scene()
    mass_before = model.body_mass[ids.can_body_id]
    inertia_before = model.body_inertia[ids.can_body_id].copy()

    apply_scene_perturbation(
        model, data, ids, ScenePerturbation(0.0, 0.0, 0.0, 1.1, 1.0)
    )

    assert model.body_mass[ids.can_body_id] == pytest.approx(mass_before * 1.1)
    assert model.body_inertia[ids.can_body_id] == pytest.approx(inertia_before * 1.1)


def test_adapted_panda_xml_is_a_single_raw_tcp_site_insertion() -> None:
    """Would catch a normalization or unrecorded change to the upstream model."""
    upstream = (ASSET_ROOT / "panda.xml").read_bytes()
    adapted = (ASSET_ROOT / "panda_exp001.xml").read_bytes()

    assert adapted.count(TCP_SITE_LINE) == 1
    assert adapted.replace(TCP_SITE_LINE, b"") == upstream
