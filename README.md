# CV Autopilot for sim-1

This adds a fully autonomous, computer-vision-based route planner (no hardcoded obstacle positions).

## What it does
- Uses `/capture` frames → HSV segmentation to detect **green obstacles** and **red robot**.
- Inflates obstacles and plans on an occupancy grid with **A***.
- Re-plans every step (robust to moving obstacles).
- Drives via `/move_rel`.
- Saves a debug **video** per run and prints **collision count**.

## Quick start
1. Run the sim as per the main README (Flask & WebSocket running).
2. Install deps:
   ```bash
   pip install -r requirements.txt
# Terafac.ai
