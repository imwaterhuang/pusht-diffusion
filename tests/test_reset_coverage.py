import numpy as np

from pusht_diffusion.environment import make_env


def test_reset_reports_geometry_coverage_without_stepping():
    env = make_env()
    try:
        # A partly overlapping block must not be recorded as zero coverage.
        _, info = env.reset(seed=0, options={
            'reset_to_state': np.array([50, 50, 256, 256, np.pi / 4], dtype=np.float32)
        })
        assert info['coverage'] == env.unwrapped._get_coverage()
        assert 0 < info['coverage'] < 1
        assert env.env._elapsed_steps == 0
    finally:
        env.close()


def test_coverage_wrapper_preserves_observations_and_physics():
    wrapped = make_env()
    raw_holder = make_env()
    raw = raw_holder.env
    try:
        observed, _ = wrapped.reset(seed=71)
        expected, _ = raw.reset(seed=71)
        for key in observed:
            np.testing.assert_array_equal(observed[key], expected[key])
        left = wrapped.step(np.array([200, 200], dtype=np.float32))
        right = raw.step(np.array([200, 200], dtype=np.float32))
        for key in left[0]:
            np.testing.assert_array_equal(left[0][key], right[0][key])
        assert left[1:4] == right[1:4]
        assert left[4]['coverage'] == right[4]['coverage']
    finally:
        wrapped.close()
        raw_holder.close()
