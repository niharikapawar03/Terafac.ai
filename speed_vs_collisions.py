
import csv, json, statistics, subprocess, time
import requests
import matplotlib.pyplot as plt
import pandas as pd

BASE = "http://127.0.0.1:5000"   
SPEED_ENDPOINT = f"{BASE}/obstacles/speed"
MOVE_ENDPOINT  = f"{BASE}/obstacles/move"

CORNERS = ["tl","tr","bl","br"]
SPEEDS = [0.2, 0.4, 0.6, 0.8, 1.0]  

def set_moving(enabled=True):
    try:
        requests.post(MOVE_ENDPOINT, json={"enabled": enabled}, timeout=2)
    except Exception:
        pass

def set_speed(v: float):
    requests.post(SPEED_ENDPOINT, json={"speed": float(v)}, timeout=2)

if __name__ == "__main__":
    set_moving(True)
    rows = [("speed","avg_collisions")]

    for spd in SPEEDS:
        print(f"== Speed {spd} ==")
        set_speed(spd)
        res = []
        for c in CORNERS:
            out = f"lvl3_s{spd}_{c}.mp4"
            p = subprocess.run(
                ["python","autopilot.py","--corner",c,"--out",out],
                capture_output=True, text=True
            )
            stats = json.loads([ln for ln in p.stdout.splitlines() if ln.strip().startswith("{")][-1])
            res.append(stats["collisions"])
            time.sleep(0.5)
        rows.append((spd, statistics.mean(res)))

    with open("speed_vs_collisions.csv","w",newline="") as f:
        csv.writer(f).writerows(rows)
    print("Wrote speed_vs_collisions.csv")
    df = pd.read_csv("speed_vs_collisions.csv")
    ax = df.plot(x="speed", y="avg_collisions", marker="o", legend=False, grid=True,
                 title="Obstacle Speed vs Avg Collisions (4 corners)")
    ax.set_xlabel("Obstacle speed")
    ax.set_ylabel("Average collisions")
    plt.tight_layout()
    plt.savefig("speed_vs_collisions.png")
    print("Saved speed_vs_collisions.png")
