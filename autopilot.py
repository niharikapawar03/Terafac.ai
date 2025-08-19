
import io, time, math, json, argparse
from collections import deque
from typing import Tuple, List, Optional, Deque

import numpy as np
import cv2
import requests

BASE = "http://127.0.0.1:5000"   
CAM_ENDPOINT = f"{BASE}/capture"
MOVE_REL_ENDPOINT = f"{BASE}/move_rel"       
MOVE_OBS_ENDPOINT = f"{BASE}/obstacles/move"
SPEED_OBS_ENDPOINT = f"{BASE}/obstacles/speed"

HSV_GREEN = ((35, 60, 60), (85, 255, 255))   
HSV_RED_1 = ((0, 90, 60), (10, 255, 255))     
HSV_RED_2 = ((170, 90, 60), (180, 255, 255))  

INFLATE_PX = 10          
GRID_DOWNSCALE = 2       
STEP_WORLD = 0.12        
WAYPOINT_PIX_STEP = 20   
REPLAN_HZ = 6            
MAX_SECONDS = 180        
GOAL_MARGIN = 30         


def get_frame() -> np.ndarray:
    r = requests.get(CAM_ENDPOINT, timeout=5)
    r.raise_for_status()
    img = np.frombuffer(r.content, np.uint8)
    frame = cv2.imdecode(img, cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError("Failed to decode /capture image")
    return frame

def robot_center_bgr(frame: np.ndarray) -> Optional[Tuple[int, int]]:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, np.array(HSV_RED_1[0]), np.array(HSV_RED_1[1]))
    m2 = cv2.inRange(hsv, np.array(HSV_RED_2[0]), np.array(HSV_RED_2[1]))
    mask = cv2.morphologyEx(cv2.bitwise_or(m1, m2), cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: 
        return None
    c = max(cnts, key=cv2.contourArea)
    (x,y), r = cv2.minEnclosingCircle(c)
    return (int(x), int(y))

def obstacles_mask(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(HSV_GREEN[0]), np.array(HSV_GREEN[1]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (7,7)), iterations=2)
    if INFLATE_PX > 0:
        mask = cv2.dilate(mask,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*INFLATE_PX+1, 2*INFLATE_PX+1)))
    return mask

def to_grid(mask: np.ndarray, scale:int) -> np.ndarray:
    if scale > 1:
        h, w = mask.shape[:2]
        mask = cv2.resize(mask, (w//scale, h//scale), interpolation=cv2.INTER_NEAREST)
    grid = (mask > 0).astype(np.uint8)  
    return grid

def a_star(grid: np.ndarray, start: Tuple[int,int], goal: Tuple[int,int]) -> Optional[List[Tuple[int,int]]]:
    H, W = grid.shape
    sx, sy = start
    gx, gy = goal
    if not (0<=sx<W and 0<=sy<H and 0<=gx<W and 0<=gy<H): 
        return None
    if grid[sy, sx] or grid[gy, gx]: 
        return None

    def h(p): return abs(p[0]-gx) + abs(p[1]-gy)
    open_set = {(sx, sy)}
    came = {}
    g = { (sx,sy): 0.0 }
    f = { (sx,sy): h((sx,sy)) }

    nbrs = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
    while open_set:
        current = min(open_set, key=lambda p: f.get(p, 1e18))
        if current == (gx, gy):
            path = [current]
            while current in came:
                current = came[current]
                path.append(current)
            return list(reversed(path))
        open_set.remove(current)
        cx, cy = current
        for dx,dy in nbrs:
            nx, ny = cx+dx, cy+dy
            if not (0<=nx<W and 0<=ny<H): 
                continue
            if grid[ny, nx]: 
                continue
            step = 1.4142 if dx and dy else 1.0
            tentative = g[current] + step
            if tentative < g.get((nx,ny), 1e18):
                came[(nx,ny)] = current
                g[(nx,ny)] = tentative
                f[(nx,ny)] = tentative + h((nx,ny))
                open_set.add((nx,ny))
    return None

def choose_goal(frame_shape, corner: str, margin:int=GOAL_MARGIN) -> Tuple[int,int]:
    h, w = frame_shape[:2]
    corner = corner.lower()
    if corner == "tl": return (margin, margin)
    if corner == "tr": return (w - margin, margin)
    if corner == "bl": return (margin, h - margin)
    return (w - margin, h - margin)  

def pix_to_step_vec(p_from: Tuple[int,int], p_to: Tuple[int,int]) -> Tuple[float,float]:
    dx = float(p_to[0] - p_from[0])
    dy = float(p_to[1] - p_from[1])
    n = math.hypot(dx, dy) + 1e-6
    ux, uy = dx / n, dy / n
    
    return (STEP_WORLD * ux, STEP_WORLD * uy)

def move_rel(dx: float, dy: float) -> bool:
    r = requests.post(MOVE_REL_ENDPOINT, json={"dx": dx, "dy": dy}, timeout=3)
    try:
        j = r.json()
    except Exception:
        j = {}
    return bool(j.get("collided", False))

def overlay_debug(frame, start, goal, mask, path_pts):
    dbg = frame.copy()
    if mask is not None:
        m3 = cv2.cvtColor((mask>0).astype(np.uint8)*255, cv2.COLOR_GRAY2BGR)
        dbg = cv2.addWeighted(dbg, 0.75, m3, 0.25, 0)
    if start is not None:
        cv2.circle(dbg, start, 6, (0,0,255), -1)   
    if goal is not None:
        cv2.circle(dbg, goal, 6, (0,255,255), -1)  
    if path_pts:
        for i in range(1, len(path_pts)):
            cv2.line(dbg, path_pts[i-1], path_pts[i], (255,0,0), 2)
    return dbg

def run_once(corner: str, out_video: str, max_seconds:int=MAX_SECONDS) -> dict:
    t0 = time.time()
    fps = REPLAN_HZ
    collisions = 0

    frame = get_frame()
    goal_pix = choose_goal(frame.shape, corner)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_video, fourcc, fps, (frame.shape[1], frame.shape[0]))

    consecutive_stuck = 0
    last_pos: Deque[Tuple[int,int]] = deque(maxlen=20)

    while time.time() - t0 < max_seconds:
        frame = get_frame()
        start_pix = robot_center_bgr(frame)
        if start_pix is None:
            time.sleep(1.0/fps)
            continue

        last_pos.append(start_pix)
        if len(last_pos) == last_pos.maxlen and \
           max(np.hypot(last_pos[-1][0]-p[0], last_pos[-1][1]-p[1]) for p in last_pos) < 8:
            consecutive_stuck += 1
        else:
            consecutive_stuck = 0

        mask = obstacles_mask(frame)
        grid = to_grid(mask, GRID_DOWNSCALE)
        s = (start_pix[0]//GRID_DOWNSCALE, start_pix[1]//GRID_DOWNSCALE)
        g = (goal_pix[0]//GRID_DOWNSCALE, goal_pix[1]//GRID_DOWNSCALE)
        path = a_star(grid, s, g)

        path_pix: List[Tuple[int,int]] = []
        if path:
            path_pix = [(x*GRID_DOWNSCALE, y*GRID_DOWNSCALE) for (x,y) in path]
            idx = min(len(path_pix)-1, max(1, WAYPOINT_PIX_STEP))
            waypoint = path_pix[idx]
        else:
            waypoint = goal_pix  

        dx, dy = pix_to_step_vec(start_pix, waypoint)
        if move_rel(dx, dy):
            collisions += 1

        
        if math.hypot(goal_pix[0]-start_pix[0], goal_pix[1]-start_pix[1]) < 24:
            writer.write(overlay_debug(frame, start_pix, goal_pix, mask, path_pix))
            break

        writer.write(overlay_debug(frame, start_pix, goal_pix, mask, path_pix))

       
        if consecutive_stuck >= 10:
            move_rel(-dy*0.6, dx*0.6)
            consecutive_stuck = 0

        time.sleep(max(0.0, 1.0/fps))

    writer.release()
    return {"collisions": collisions, "video": out_video}

def set_obstacles_moving(enabled: bool=True):
    try:
        requests.post(MOVE_OBS_ENDPOINT, json={"enabled": enabled}, timeout=2)
    except Exception:
        pass

def set_obstacle_speed(speed: float):
    try:
        requests.post(SPEED_OBS_ENDPOINT, json={"speed": speed}, timeout=2)
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corner", default="br", choices=["tl","tr","bl","br"],
                    help="goal corner (tl,tr,bl,br)")
    ap.add_argument("--out", default="run.mp4", help="output video filename")
    ap.add_argument("--max-seconds", type=int, default=MAX_SECONDS)
    args = ap.parse_args()

    stats = run_once(args.corner, args.out, args.max_seconds)
    print(json.dumps(stats, indent=2))

if __name__ == "__main__":
    main()
