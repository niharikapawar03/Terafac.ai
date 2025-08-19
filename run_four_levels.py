
import json, subprocess, statistics, time

CORNERS = ["tl","tr","bl","br"]

def run_series(prefix="level1"):
    results = []
    for c in CORNERS:
        out = f"{prefix}_{c}.mp4"
        print(f"== {prefix.upper()} :: corner {c} ==")
        p = subprocess.run(
            ["python", "autopilot.py", "--corner", c, "--out", out],
            capture_output=True, text=True
        )
        print(p.stdout)
        stats = json.loads([ln for ln in p.stdout.splitlines() if ln.strip().startswith("{")][-1])
        results.append(stats["collisions"])
        time.sleep(1)
    avg = statistics.mean(results)
    print(f"{prefix.upper()} average collisions over 4 runs: {avg:.2f}")

if __name__ == "__main__":
    run_series("level1")

