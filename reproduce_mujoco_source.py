"""Build and audit the pinned upstream MuJoCo revision (requires network/CMake).

This is a version comparison, not an isolated correction-patch experiment.
The released-engine benchmarks must have been generated first.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

from check_mujoco_source import PIN, ROOT


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--work-dir", type=Path, default=ROOT / "external/mujoco-source")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    cmake = shutil.which("cmake")
    if cmake is None:
        bundled = Path(sys.executable).parent / "cmake"
        if bundled.is_file():
            cmake = str(bundled)
    if cmake is None or any(shutil.which(tool) is None for tool in ("git", "g++", "make")):
        parser.error("Install CMake (included in requirements-lock.txt), Git, G++ and Make first")
    if not (ROOT / "Results/mujoco_audit_shipped_step_trials.csv").exists():
        parser.error("Run the released-engine benchmarks first (make all)")
    started = datetime.now(timezone.utc).isoformat()
    t0 = time.perf_counter()
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    source, build = work / "source", work / "build"
    if not source.exists():
        subprocess.run(["git", "clone", "--no-checkout",
                        "https://github.com/google-deepmind/mujoco.git", str(source)], check=True)
        subprocess.run(["git", "-C", str(source), "checkout", "--detach", PIN], check=True)
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if commit != PIN or subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"]):
        raise SystemExit("Use a clean checkout at the pinned revision or choose a new --work-dir")
    release = work / "engine_forward-3.16.0.c"
    release.write_bytes(subprocess.check_output(
        ["git", "-C", str(source), "show", "3.16.0:src/engine/engine_forward.c"]))
    subprocess.run([cmake, "-S", str(source), "-B", str(build),
                    "-DCMAKE_BUILD_TYPE=Release", "-DMUJOCO_BUILD_EXAMPLES=OFF",
                    "-DMUJOCO_BUILD_SIMULATE=OFF", "-DMUJOCO_BUILD_TESTS=OFF"], check=True)
    subprocess.run([cmake, "--build", str(build), "--target", "mujoco", "-j", str(args.jobs)], check=True)
    build_seconds = time.perf_counter()-t0
    subprocess.run([sys.executable, str(ROOT / "check_mujoco_source.py"),
                    "--source", str(source), "--build", str(build),
                    "--release-source", str(release)], cwd=ROOT, check=True)
    report = dict(started_utc=started, completed_utc=datetime.now(timezone.utc).isoformat(),
                  commit=PIN, platform=platform.platform(), python=platform.python_version(),
                  cmake=subprocess.check_output([cmake, "--version"], text=True).splitlines()[0],
                  compiler=subprocess.check_output(["g++", "--version"], text=True).splitlines()[0],
                  jobs=args.jobs, fetch_configure_build_seconds=build_seconds,
                  total_wall_seconds=time.perf_counter()-t0,
                  note="Build may reuse existing sources and compiled objects; not an isolated patch test")
    (ROOT / "Results/runtime_source_build.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
