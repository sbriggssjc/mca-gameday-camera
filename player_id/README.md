# Player Identification Calibration

This package contains a lightweight placeholder for the player appearance
calibration flow.  In a full deployment coaches would open a small UI that
presents uncertain tracklets and allows them to tag players by visual
attributes (cleat color, socks, etc.).  The resulting labelled data is saved to
`data/players.json` and reused on subsequent runs.

The current implementation does not provide a full UI but exposes a
`launch_ui` function so the analysis pipeline can request calibration when
needed.
