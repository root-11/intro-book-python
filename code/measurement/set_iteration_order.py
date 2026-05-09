# /// script
# requires-python = ">=3.11"
# ///
"""
§16 exhibit — set iteration order is process-dependent.

CPython sets randomise their iteration order across processes via
PYTHONHASHSEED. The same set, with the same insertions, in the same
program, iterates in a different order in a different process. This is
*by design* — randomised hashing protects servers from hash-flooding
attacks — but it is also a source of non-determinism that this chapter
forbids inside the simulator.

Dicts are insertion-ordered since CPython 3.7. They survive the
process-to-process test below.

We run the same tiny script in three fresh subprocesses with
PYTHONHASHSEED=random and compare what each prints.

Run:
    uv run code/measurement/set_iteration_order.py
"""

import os
import subprocess
import sys


CHILD_SCRIPT = """
items = {"alpha", "bravo", "charlie", "delta", "echo", "foxtrot"}
print(",".join(items), flush=True)
d = {"alpha": 1, "bravo": 2, "charlie": 3, "delta": 4, "echo": 5, "foxtrot": 6}
print(",".join(d.keys()), flush=True)
"""


def run_once() -> tuple[str, str]:
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "random"
    out = subprocess.run(
        [sys.executable, "-c", CHILD_SCRIPT],
        capture_output=True, text=True, env=env,
    ).stdout.strip().splitlines()
    return out[0], out[1]


def main() -> None:
    print("Running the same script in three fresh subprocesses with "
          "PYTHONHASHSEED=random.\n")
    runs = [run_once() for _ in range(3)]

    print("Set iteration order across runs:")
    for i, (set_out, _) in enumerate(runs, 1):
        print(f"  run {i}: {set_out}")
    set_orders = {set_out for set_out, _ in runs}
    if len(set_orders) > 1:
        print(f"  → {len(set_orders)} distinct orders — sets are non-deterministic.")
    else:
        print("  → orders match this time. Rerun; sets are random across processes.")

    print("\nDict iteration order across runs:")
    for i, (_, dict_out) in enumerate(runs, 1):
        print(f"  run {i}: {dict_out}")
    dict_orders = {dict_out for _, dict_out in runs}
    if len(dict_orders) == 1:
        print("  → orders match — dicts are insertion-ordered since CPython 3.7.")
    else:
        print("  → orders differ — unexpected. Check your CPython version.")

    print("\nLesson: any system in the simulator that iterates a set is a "
          "non-determinism source.")
    print("Use sorted(set), or a numpy array, or a list — never raw set "
          "iteration order.")


if __name__ == "__main__":
    main()
