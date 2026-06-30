import sys
from pathlib import Path
sys.path.append(str(Path("src").resolve()))
from ocr.reader import HUDReader
from utils import get_frame_at, clean_name
import config
from collections import deque

video_path = Path("dataset/possesso_palla_e_parata.mov")
reader = HUDReader(langs=config.OCR_LANGUAGES)

prev_home = ""
prev_away = ""
change_log = deque()
prev_possession = None
POSS_WINDOW_SEC = 2.0

for t in [2.52, 3.02, 3.53, 4.03]:
    frame = get_frame_at(video_path, t)
    home_txt = reader.read_region(frame, config.HUD_REGIONS["active_player_home"], 0.15)
    away_txt = reader.read_region(frame, config.HUD_REGIONS["active_player_away"], 0.15)
    snap_h = clean_name(home_txt)
    snap_a = clean_name(away_txt)
    
    home_changed = bool(snap_h["name"]) and snap_h["name"] != prev_home
    away_changed = bool(snap_a["name"]) and snap_a["name"] != prev_away
    
    if home_changed: change_log.append((t, "home"))
    if away_changed: change_log.append((t, "away"))
    
    while change_log and change_log[0][0] < t - POSS_WINDOW_SEC:
        change_log.popleft()
        
    n_home = sum(1 for _, s in change_log if s == "home")
    n_away = sum(1 for _, s in change_log if s == "away")
    
    if n_home > n_away: current_possession = "home"
    elif n_away > n_home: current_possession = "away"
    else: current_possession = prev_possession
    
    active_team = current_possession if current_possession else config.HUD_ACTIVE_SIDE
    
    print(f"t={t:.2f} | H: {snap_h['name']} A: {snap_a['name']} | changed H:{home_changed} A:{away_changed} | n_H:{n_home} n_A:{n_away} | POSS: {active_team}")
    
    prev_home = snap_h["name"] or prev_home
    prev_away = snap_a["name"] or prev_away
    prev_possession = current_possession
