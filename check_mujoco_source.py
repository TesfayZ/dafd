"""Record a pinned source comparison and audit a separately built native engine.

Build instructions are in README. Uses a C++ worker to avoid loading a new
engine through ABI-incompatible installed Python bindings. The installed
Gymnasium package supplies XML assets and initial positions only. All dynamics
and energy evaluations use the source-built shared library.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import platform
import subprocess
import tempfile

import gymnasium as gym
import mujoco
import numpy as np
import pandas as pd
import fingerprint as F

PIN = "8892bf44b07c20f0e4dcd8b8c34ca46683b97b76"
ROOT = Path(__file__).resolve().parent


def function(data):
    start = data.index(b"void mj_RungeKutta(")
    pos = data.index(b"{", start)
    depth = 1
    end = pos+1
    while depth:
        depth += (data[end] == 123) - (data[end] == 125)
        end += 1
    return data[start:end] + b"\n"


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--build", type=Path, required=True)
    ap.add_argument("--release-source", type=Path, required=True,
                    help="engine_forward.c from release tag 3.16.0")
    a = ap.parse_args()
    commit = subprocess.check_output(["git", "-C", str(a.source), "rev-parse", "HEAD"], text=True).strip()
    if commit != PIN:
        raise SystemExit(f"Expected {PIN}, got {commit}")
    if subprocess.check_output(["git", "-C", str(a.source), "status", "--porcelain"]):
        raise SystemExit("Source checkout must be clean")
    current = (a.source / "src/engine/engine_forward.c").read_bytes()
    old = a.release_source.read_bytes()
    results = ROOT / "Results"
    rows = []
    trajectory = []
    validation = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        f0, f1 = tmp/"release.c", tmp/"main.c"
        f0.write_bytes(function(old)); f1.write_bytes(function(current))
        cmp_result = subprocess.run(["cmp", "-s", str(f0), str(f1)]).returncode
        if cmp_result not in (0, 1):
            raise SystemExit("cmp failed")
        worker = tmp/"audit"
        lib = (a.build/"lib").resolve()
        build_cmd = ["g++", "-O2", "-std=c++17", str(ROOT/"audit_mujoco_source.cc"),
                     "-I"+str(a.source.resolve()/"include"), "-L"+str(lib),
                     "-Wl,-rpath,"+str(lib), "-lmujoco", "-o", str(worker)]
        subprocess.run(build_cmd, check=True)
        installed = Path(mujoco.__file__).parent
        release_worker = tmp/"audit-release"
        subprocess.run(["g++","-O2","-std=c++17",str(ROOT/"audit_mujoco_source.cc"),
                        "-I"+str(installed/"include"),str(installed/"libmujoco.so.3.16.0"),
                        "-Wl,-rpath,"+str(installed),"-o",str(release_worker)],check=True)
        archived = pd.read_csv(results/"mujoco_audit_shipped_step_trials.csv")
        for env_id in ("Ant-v5", "Humanoid-v5"):
            env = gym.make(env_id)
            u = env.unwrapped
            for seed in (0, 1, 2):
                vel = np.random.default_rng(seed).uniform(-1, 1, (24, u.model.nv))
                for K in (1, 4, 16, 64, 256):
                    h = float(u.model.opt.timestep)*K
                    inputs = f"24 20 {h:.17g}\n" + " ".join(format(x,".17g") for x in u.init_qpos) + "\n"
                    inputs += " ".join(format(x,".17g") for x in vel.ravel()) + "\n"
                    output = subprocess.check_output([str(worker),u.fullpath], input=inputs, text=True)
                    if seed == 0 and K == 1:
                        baseline = subprocess.check_output([str(release_worker),u.fullpath],input=inputs,text=True)
                        baseline = pd.read_csv(io.StringIO(baseline),names=["start","E_abs_mean","r1","r2","r4","r8"])
                        expected = archived[archived.environment == env_id].sort_values("start")
                        cols = ["E_abs_mean","r1","r2","r4","r8"]
                        difference = np.max(np.abs(baseline[cols].values-expected[cols].values))
                        if not np.allclose(baseline[cols].values,expected[cols].values,rtol=1e-10,atol=1e-13):
                            raise RuntimeError(f"Native worker disagrees with released-engine Python protocol: {env_id}")
                        validation.append(dict(environment=env_id,n_starts=24,max_abs_difference=float(difference)))
                    trial = pd.read_csv(io.StringIO(output), names=["start","E_abs_mean","r1","r2","r4","r8"])
                    trial["cls"] = [F.guarded_classify(x.r1,x.r2,x.r4,x.r8,4,x.E_abs_mean,h) for x in trial.itertuples()]
                    resolved = trial[~trial.cls.isin(["undetermined","unstable","blowup"])]
                    selected = len(resolved) >= 12 or K == 256
                    for x in trial.to_dict("records"):
                        start = int(x["start"])
                        rows.append(dict(environment=env_id,seed=seed,K=K,macro_h=h,selected=selected,
                                         commit=commit,initial_velocity=repr(vel[start].tolist()),**x))
                    print(env_id,seed,K,trial.cls.value_counts().to_dict(),flush=True)
                    if selected: break
            vel = np.random.default_rng(0).uniform(-1,1,(8,u.model.nv))
            base_h = .04 if env_id == "Ant-v5" else .048
            inputs = f"8 {round(.48/base_h)} {base_h:.17g}\n"
            inputs += " ".join(format(x,".17g") for x in u.init_qpos)+"\n"
            inputs += " ".join(format(x,".17g") for x in vel.ravel())+"\n"
            output = subprocess.check_output([str(worker),u.fullpath,"trajectory"],input=inputs,text=True)
            tr = pd.read_csv(io.StringIO(output),names=["start","level","qpos_error"])
            tr["environment"] = env_id
            tr["base_h"] = base_h
            tr["horizon_s"] = .48
            tr["reference_substeps"] = 256
            trajectory.extend(tr.to_dict("records"))
            env.close()
    pd.DataFrame(rows).to_csv(results/"mujoco_main_trials.csv", index=False)
    summary=[]
    df=pd.DataFrame(rows)
    for (env,seed,K),g in df.groupby(["environment","seed","K"],sort=False):
        ex=np.log2(g[["r1","r2","r4"]].to_numpy()/g[["r2","r4","r8"]].to_numpy())
        summary.append(dict(environment=env,seed=int(seed),K=int(K),macro_h=float(g.macro_h.iloc[0]),
                            selected=bool(g.selected.iloc[0]),classes=str(g.cls.value_counts().to_dict()),
                            median_a=np.median(ex[:,0]),median_b=np.median(ex[:,1]),median_c=np.median(ex[:,2])))
    pd.DataFrame(summary).to_csv(results/"mujoco_main_audit.csv",index=False)
    pd.DataFrame(trajectory).to_csv(results/"mujoco_main_trajectory.csv",index=False)
    sha=lambda b:hashlib.sha256(b).hexdigest()
    metadata=dict(commit=commit,release="3.16.0",cmp_command="cmp -s release-mj_RungeKutta.c main-mj_RungeKutta.c",
                  cmp_exit_status=cmp_result,byte_identical=cmp_result==0,
                  release_function_sha256=sha(function(old)),main_function_sha256=sha(function(current)),
                  release_file_sha256=sha(old),main_file_sha256=sha(current),
                  rkmk_present=b"rkCorrectVel(m, X[i]+nq, dX, h)" in current,
                  library_sha256=sha((lib/"libmujoco.so").read_bytes()),
                  worker_sha256=sha((ROOT/"audit_mujoco_source.cc").read_bytes()),
                  python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,gymnasium=gym.__version__,
                  compiler=subprocess.check_output(["g++","--version"],text=True).splitlines()[0],
                  engine="4.0.0 (source)",steps=20,starts_per_seed=24,seeds=[0,1,2])
    metadata["release_worker_protocol_validation"] = validation
    (results/"mujoco_source_check.json").write_text(json.dumps(metadata,indent=2)+"\n")


if __name__ == "__main__": main()
