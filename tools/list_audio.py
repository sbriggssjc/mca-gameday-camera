"""List available audio capture devices."""

import subprocess


print("Pulse sources:")
subprocess.run(
    "pactl list short sources | awk '{print \"  - \"$2}'",
    shell=True,
    check=False,
)

print("\nALSA devices:")
subprocess.run(
    r"arecord -l | sed -n 's/^card \([0-9]\+\): .*device \([0-9]\+\).*/  - hw:\1,\2/p'",
    shell=True,
    check=False,
)

