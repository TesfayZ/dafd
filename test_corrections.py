import unittest
from unittest.mock import patch
import numpy as np
import fingerprint as F


class GuardTests(unittest.TestCase):
    def test_detection_floor_requires_all_larger_faults(self):
        from benchmark_matched_compute_noise_fingerprint import thresholds
        table = {0.: {"test": 1.}, .01: {"test": 11.},
                 .02: {"test": 9.}, .03: {"test": 12.}}
        self.assertEqual(thresholds(table, ["test"])["test"], (.03, 1.))
        table[.04] = {"test": 9.}
        self.assertEqual(thresholds(table, ["test"])["test"], (None, 1.))

    def test_replica_rotated_inertial_frame(self):
        import mujoco
        from benchmark_quaternion_mechanism import Top, build, XMLS, step_mujoco_style, mj_advance, quat_err
        body = XMLS["heavy top, ball joint"].replace('mass="1"',
                    'mass="1" quat="0.9238795325 0.3826834324 0 0"')
        model = build(body); data = mujoco.MjData(model); top = Top(model)
        self.assertFalse(np.allclose(model.body_iquat[1], [1, 0, 0, 0]))
        qp = model.qpos0.copy(); qv = np.array([.4, -.3, .2])
        data.qpos[:] = qp; data.qvel[:] = qv; mujoco.mj_forward(model, data)
        self.assertAlmostEqual(top.energy(qp, qv), float(data.energy.sum()), places=12)
        q1, w1 = mj_advance(model, data, qp, qv, .01, 1)
        q2, w2 = step_mujoco_style(top, qp, qv, .01)
        self.assertLess(quat_err(q1, q2), 1e-12)
        np.testing.assert_allclose(w1, w2, rtol=0, atol=1e-12)

    def test_root_height_indices(self):
        import gymnasium as gym
        from benchmark_regime_gated_fingerprint import root_height_indices
        for name, expected in (("Hopper-v5", (1, 1)), ("Ant-v5", (2, 2))):
            with gym.make(name) as env:
                self.assertEqual(root_height_indices(env.unwrapped.model), expected)

    def test_self_convergence_same_calls_and_definitions(self):
        import benchmark_self_convergence as B
        # A consistent step family with a known m^-4 endpoint error.
        # Its limiting drift can break E even while state differences show order 4.
        def step(x, h):
            return np.array(x) + np.array([.1*h + h**5, 0, 0, 0])
        with patch.object(B.I, "ship_step", side_effect=step) as advance, \
             patch.object(B.A, "energy", side_effect=lambda x: x[0]):
            values, records, status = B.trial(np.zeros(4), .1, 3)
        self.assertEqual(advance.call_count, 3 * (1+2+4+8))
        self.assertEqual(status, "smooth")
        self.assertEqual(values["cls"], "model")
        self.assertAlmostEqual(values["p12"], 4, places=6)
        self.assertAlmostEqual(values["p24"], 4, places=6)
        self.assertAlmostEqual(values["r8"], np.mean([r["r8"] for r in records]))
        self.assertAlmostEqual(B.state_distance([np.pi-.01, 0, 0, 0],
                                               [-np.pi+.01, 0, 0, 0]), .02)

    def test_self_convergence_excludes_clipped_paths(self):
        import benchmark_self_convergence as B
        clipped = np.array([0, 0, B.I.u.MAX_VEL_1, 0])
        with patch.object(B.I, "ship_step", return_value=clipped):
            values, records, status = B.trial(np.zeros(4), .1, 3)
        self.assertIsNone(values)
        self.assertEqual(status, "velocity-clipped")
        self.assertEqual(records, [])

    def test_power_laws_and_model_defects(self):
        for q in range(1, 5):
            r = [0.01/m**q for m in (1, 2, 4, 8)]
            self.assertEqual(F.guarded_classify(*r, q, 1, .1), "consistent")
        for sign in (-1, 1):
            r = [abs(.01 + sign*.0001/m**4) for m in (1, 2, 4, 8)]
            self.assertEqual(F.guarded_classify(*r, 4, 1, .1), "model")

    def test_floors_and_rate_guard(self):
        self.assertEqual(F.guarded_classify(1e-16, 1e-16, 1e-16, 1e-16, 4, 1, .1), "undetermined")
        self.assertEqual(F.guarded_classify(.01, .01, .01, .01, 4, 1, .1, noise_floor=.01), "undetermined")
        self.assertEqual(F.guarded_classify(2, 2, 2, 2, 4, 1, .1), "unstable")
        self.assertEqual(F.guarded_classify(0, 0, 0, 0, 4, 1, .1), "undetermined")
        self.assertEqual(F.guarded_classify(np.nan, 1, 1, 1, 4, 1, .1), "blowup")
        with self.assertRaises(ValueError):
            F.guarded_classify(1, 1, 1, 1, 4, 1, 0)

    def test_runner_failure_is_not_success(self):
        import contextlib, io, sys, tempfile, json
        from pathlib import Path
        import run_all_experiments as runner
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(runner, "ROOT", Path(directory)), \
             patch.object(runner, "SCRIPTS", [("failure.py", 0, False, False)]), \
             patch.object(runner, "run_one", return_value=("failure.py", 1, 0)), \
             patch.object(sys, "argv", ["runner"]), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as caught:
                runner.main()
            self.assertEqual(caught.exception.code, 1)
            report = json.loads((Path(directory)/"Results/runtime_full.json").read_text())
            self.assertEqual(report["failures"], ["failure.py"])
            self.assertFalse(report["includes_native_source_build"])

    def test_runner_rejects_missing_comparison_repository(self):
        import tempfile
        from pathlib import Path
        import sys
        import run_all_experiments as runner
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(runner, "ROOT", Path(directory)), \
             patch.object(runner, "SCRIPTS", [("comparison.py", 0, True, False)]), \
             patch.dict(runner.os.environ, {}, clear=True), \
             patch.object(sys, "argv", ["runner"]):
            with self.assertRaisesRegex(SystemExit, "Comparison repository required"):
                runner.main()

    def test_native_step_and_same_starts(self):
        import benchmark_mujoco_integrator_order as B
        trials = []
        _, _, dt, fs, out = B.audit("Ant-v5", ntr=2, nsteps=2, Ks=(1,), trial_rows=trials)
        self.assertGreater(fs, 1)
        self.assertEqual(out[1][0], dt)
        enlarged = []
        B.audit("Ant-v5", ntr=2, nsteps=2, Ks=(4,), trial_rows=enlarged)
        self.assertEqual([x["initial_velocity"] for x in trials], [x["initial_velocity"] for x in enlarged])
        self.assertTrue(all(x["macro_h"] == 4*dt for x in enlarged))

    def test_repeated_start_endpoint(self):
        import pandas as pd
        from benchmark_repeated_start_analysis import summarize
        rows = [dict(cluster_id=str(i), true_order=4, h=h, cls="model" if i == 0 else "consistent")
                for i in range(4) for h in (.1, .2)]
        r = summarize(pd.DataFrame(rows), "test", "test", "any_flag")
        self.assertEqual((r["successes"], r["n_starts"], r["n_evaluations"]), (1, 4, 8))
        with self.assertRaises(ValueError):
            summarize(pd.DataFrame(rows[:-1]), "test", "test", "any_flag")


if __name__ == "__main__":
    unittest.main()
