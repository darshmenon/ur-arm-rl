"""UR5e peg-in-hole env: builds, inserts when aligned, jams at 4 mm offset, spiral search recovers it."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
envs = pytest.importorskip("envs.ur_peg_in_hole_env")

if not os.path.exists(envs._UR5E_XML):
    pytest.skip("mujoco_menagerie UR5e not found (set MUJOCO_MENAGERIE_PATH)", allow_module_level=True)


@pytest.fixture(scope="module")
def env():
    return envs.PegInHole()


def test_builds_with_peg_and_fixture(env):
    assert env.m.nv == 6
    assert env.tip()[2] > env.hole_pos[2]          # hover pose is above the hole


def test_inserts_when_aligned(env):
    r = env.run(k_lin=800.0, lateral_offset=(0.0004, 0.0), t_end=7.0)
    assert r["inserted"]
    assert r["wall"].max() < 20.0


def test_jams_at_4mm_without_search(env):
    r = env.run(k_lin=800.0, lateral_offset=(0.004, 0.0), t_end=7.0)
    assert not r["inserted"]
    assert r["depth"].max() < 0.002                 # stopped on the top face


@pytest.mark.parametrize("angle", [0.0, 1.5708, 3.1416])
def test_spiral_search_recovers_4mm(env, angle):
    import numpy as np
    r = env.run(k_lin=800.0, lateral_offset=(0.004 * np.cos(angle), 0.004 * np.sin(angle)), t_end=20.0, search=True)
    assert r["inserted"]
