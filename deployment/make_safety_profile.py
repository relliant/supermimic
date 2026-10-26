"""Generate an explicitly UNVALIDATED hardware review template from a bundle.

This does not make the policy safe to run. All validation flags remain false.
Physical limits, IMU calibration, timing and the loss-of-command exit behavior
must be checked by the robot operator before authorizing any actuation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .contract import sha256_file, verify_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    manifest = verify_bundle(arguments.bundle)
    profile = json.loads((Path(__file__).parent / "config/simulation.json").read_text())
    profile.update({
        "review_status": "UNVALIDATED TEMPLATE; model limits are not hardware-approved limits",
        "manifest_sha256": sha256_file(arguments.bundle / "manifest.json"),
        "position_lower": manifest["joint_lower"],
        "position_upper": manifest["joint_upper"],
        **{name: manifest[name] for name in ("kp", "kd", "velocity_limit")},
        "torque_limit": manifest["effort_limit"],
        "target_rate_limit": [3.0] * 29,
        "damping_kd": [2.0] * 29,
    })
    # Exclusive creation also rejects existing symlinks rather than following them.
    with arguments.output.open("x", encoding="utf-8") as output:
        json.dump(profile, output, indent=2, allow_nan=False)
        output.write("\n")
    print(f"Wrote unvalidated review template: {arguments.output}; hardware remains disabled")


if __name__ == "__main__":
    main()
