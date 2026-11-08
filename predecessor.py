"""Locate the implementation of Gebrekidan and Gebrerufael (2026), arXiv:2609.13194v2.

Only the three comparison benchmarks need it. Obtain the source distributed
with that paper, at commit 687c868c204b8ec537451b5e046532de80197d9d, and run:
    make predecessor PREDECESSOR_REPO=/path/to/checkout
Its code is imported, not copied. See README for the required files.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PATH = Path(os.environ.get("PREDECESSOR_REPO", ROOT / "external" / "algebraicRLtest"))


def require():
    if not (PATH / "ideal_diff.py").exists():
        raise SystemExit(
            f"Comparison implementation not found at {PATH}.\n"
            "Supply the source accompanying arXiv:2609.13194v2; see README.\n"
            "Then run make predecessor PREDECESSOR_REPO=/path/to/checkout")
    if str(PATH) not in sys.path:
        sys.path.insert(0, str(PATH))
    import bugs, ideal_diff  # noqa: E402
    return bugs, ideal_diff


if __name__ == "__main__":
    require()
    print("Comparison implementation imports verified.")
