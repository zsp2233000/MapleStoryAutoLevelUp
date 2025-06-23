'''
Execute this script:
python mapleStoryAutoLevelUp.py --map cloud_balcony --monster brown_windup_bear,pink_windup_bear
'''
# Standard import
import time
import random
import argparse
import glob
import sys


# Library import
import numpy as np
import cv2

# Local import
from logger import logger
from util import find_pattern_sqdiff, draw_rectangle, screenshot, nms, \
                load_image, get_mask, get_minimap_loc_size, get_player_location_on_minimap, \
                is_mac, nms_matches, override_cfg, load_yaml, get_all_other_player_locations_on_minimap, \
                click_in_game_window
from KeyBoardController import KeyBoardController
if is_mac():
    from GameWindowCapturorForMac import GameWindowCapturor
else:
    from GameWindowCapturor import GameWindowCapturor
from HealthMonitor import HealthMonitor

class MapleStoryBot:
    '''
    MapleStoryBot
    '''
    def __init__(self, args):
        '''
        Init MapleStoryBot
        '''
        # self.cfg = Config # Configuration
        self.args = args # User arguments
        self.status = "hunting" # 'resting', 'finding_rune', 'near_rune', 'attacking'
        self.idx_routes = 0 # Index of route map
        self.monster_info = [] # monster information
        self.fps = 0 # Frame per second
        self.is_first_frame = True # first frame flag
        self.red_dot_center_prev = None # previous other player location in minimap
        # Coordinate (top-left coordinate)
        self.loc_nametag = (0, 0) # nametag location on game screen
        self.loc_minimap = (0, 0) # minimap location on game screen
        self.loc_player = (0, 0) # player location on game screen
        self.loc_player_minimap = (0, 0) # player location on minimap
        self.loc_minimap_global = (0, 0) # minimap location on global map
        self.loc_player_global = (0, 0) # player location on global map
        self.loc_watch_dog = (0, 0) # watch dog location on global map
        self.loc_rune = (0, 0) # rune location on game screen
        # Images
        self.frame = None # raw image
        self.img_frame = None # game window frame
        self.img_frame_gray = None # game window frame graysale
        self.img_frame_debug = None # game window frame for visualization
        self.img_route = None # route map
        self.img_route_debug = None # route map for visualization
        self.img_minimap = np.zeros((10, 10, 3), dtype=np.uint8) # minimap on game screen
        # Timers
        self.t_last_frame = time.time() # Last frame timer, for fps calculation
        self.t_last_switch_status = time.time() # Last status switches timer
        self.t_watch_dog = time.time() # Last movement timer
        self.t_last_teleport = time.time() # Last teleport timer
        self.t_patrol_last_attack = time.time() # Last patrol attack timer
        self.t_last_attack = time.time() # Last attack timer for cooldown
        self.t_last_rune_trigger = time.time() # Last time trigger rune
        # Patrol mode
        self.is_patrol_to_left = True # Patrol direction flag
        self.patrol_turn_point_cnt = 0 # Patrol tuning back counter

        # Simplified combat management
        self.t_last_monster_seen = 0  # Last time we saw any monster
        self.combat_delay = 1.0  # Seconds to wait after last monster before moving

        # Load defautl yaml config
        cfg = load_yaml("config/config_default.yaml")
        # Override with platform config
        if is_mac():
            cfg = override_cfg(cfg, load_yaml("config/config_macOS.yaml"))
        # Override with user customized config
        self.cfg = override_cfg(cfg, load_yaml(f"config/config_{args.cfg}.yaml"))

        # Parse color code
        self.color_code = {
            tuple(map(int, k.split(','))): v
            for k, v in cfg["route"]["color_code"].items()
        }

        # Set status to hunting for startup
        self.switch_status("hunting")

        if args.patrol:
            # Patrol mode doesn't need map or route
            self.img_map = None
            self.img_routes = []
            self.img_route_rest = None
        else:
            # Load map.png from minimaps/
            self.img_map = load_image(f"minimaps/{args.map}/map.png",
                                      cv2.IMREAD_COLOR)
            # Load route*.png from minimaps/
            route_files = sorted(glob.glob(f"minimaps/{args.map}/route*.png"))
            route_files = [p for p in route_files if not p.endswith("route_rest.png")]
            self.img_routes = [
                cv2.cvtColor(load_image(p), cv2.COLOR_BGR2RGB) for p in route_files
            ]
            # Load route_rest.png from minimaps/
            self.img_route_rest = cv2.cvtColor(
                load_image(f"minimaps/{args.map}/route_rest.png"), cv2.COLOR_BGR2RGB)

        # Load player's name tag
        self.img_nametag = load_image(f"nametag/{args.nametag}.png")
        self.img_nametag_gray = load_image(f"nametag/{args.nametag}.png", cv2.IMREAD_GRAYSCALE)

        # Load rune images from rune/
        rune_ver = self.cfg["rune_warning"]["language"]
        if rune_ver == "chinese":
            self.img_rune_warning = load_image("rune/rune_warning.png", cv2.IMREAD_GRAYSCALE)
        elif rune_ver == "english":
            self.img_rune_warning = load_image("rune/rune_warning_eng.png", cv2.IMREAD_GRAYSCALE)
        else:
            logger.error(f"Unsupported rune warning version: {rune_ver}")

        self.img_runes = [load_image("rune/rune_1.png"),
                          load_image("rune/rune_2.png"),
                          load_image("rune/rune_3.png"),]
        if rune_ver == "english":
            self.img_runes[1] = load_image("rune/rune_2_eng.png")

        self.img_arrows = {
            "left":
                [load_image("rune/arrow_left_1.png"),
                load_image("rune/arrow_left_2.png"),
                load_image("rune/arrow_left_3.png"),],
            "right":
                [load_image("rune/arrow_right_1.png"),
                load_image("rune/arrow_right_2.png"),
                load_image("rune/arrow_right_3.png"),],
            "up":
                [load_image("rune/arrow_up_1.png"),
                load_image("rune/arrow_up_2.png"),
                load_image("rune/arrow_up_3.png")],
            "down":
                [load_image("rune/arrow_down_1.png"),
                load_image("rune/arrow_down_2.png"),
                load_image("rune/arrow_down_3.png"),],
        }

        # Load monsters images from monster/{monster_name}
        self.monsters = {}
        for monster_name in args.monsters.split(","):
            imgs = []
            for file in glob.glob(f"monster/{monster_name}/{monster_name}*.png"):
                # Add original image
                img = load_image(file)
                imgs.append((img, get_mask(img, (0, 255, 0))))
                # Add flipped image
                img_flip = cv2.flip(img, 1)
                imgs.append((img_flip, get_mask(img_flip, (0, 255, 0))))
            if imgs:
                self.monsters[monster_name] = imgs
            else:
                logger.error(f"No images found in monster/{monster_name}/{monster_name}*")
                raise RuntimeError(f"No images found in monster/{monster_name}/{monster_name}*")
        logger.info(f"Loaded monsters: {list(self.monsters.keys())}")

        # Start keyboard controller thread
        self.kb = KeyBoardController(self.cfg, args)
        if args.disable_control:
            self.kb.disable()

        # Start game window capturing thread
        logger.info("Waiting for game window to activate, please click on game window")
        self.capture = GameWindowCapturor(self.cfg)

        # Start health monitoring thread
        self.health_monitor = HealthMonitor(self.cfg, args, self.kb)
        if self.cfg["health_monitor"]["enable"]:
            self.health_monitor.start()

    def get_player_location_by_nametag(self):
        '''
        Detects the player's location based on the nametag position in the game window.

        This function works by:
        - Extracting a vertical region of interest (ROI) where the nametag is expected.
        - Padding the ROI to avoid template matching edge issues.
        - Using template matching to locate the nametag, split into left and right halves
        to improve robustness against partial occlusion.
        - Selecting the best match (left or right) based on score and cache status.
        - Computing the player's center position by applying a fixed offset to the nametag.

        Returns:
            loc_player (tuple): The (x, y) coordinates of the player's estimated location.
        '''
        # Get camera region in the game window
        img_camera = self.img_frame_gray[
            self.cfg["camera"]["y_start"]:self.cfg["camera"]["y_end"], :]

        # Get nametag image and search image
        if self.cfg["nametag"]["mode"] == "white_mask":
            # Apply Gaussian blur for smoother white detection
            img_camera = cv2.GaussianBlur(img_camera, (3, 3), 0)
            img_nametag = cv2.GaussianBlur(self.img_nametag_gray, (3, 3), 0)
            lower_white, upper_white = (150, 255)
            img_roi = cv2.inRange(img_camera, lower_white, upper_white)
            img_nametag  = cv2.inRange(img_nametag, lower_white, upper_white)
        elif self.cfg["nametag"]["mode"] == "grayscale":
            img_roi = img_camera
            img_nametag = self.img_nametag_gray
        else:
            logger.error(f"Unsupported nametag detection mode: {self.cfg['nametag']['mode']}")
            return
        # cv2.imshow("img_roi", img_roi)
        # cv2.imshow("img_nametag", img_nametag)

        # Pad search region to deal with fail detection when player is at map edge
        (pad_y, pad_x) = self.img_nametag.shape[:2]
        img_roi = cv2.copyMakeBorder(
            img_roi,
            pad_y, pad_y, pad_x, pad_x,
            borderType=cv2.BORDER_REPLICATE  # replicate border for safe matching
        )

        # Get last frame name tag location
        if self.is_first_frame:
            last_result = None
        else:
            last_result = (
                self.loc_nametag[0] + pad_x,
                self.loc_nametag[1] + pad_y - self.cfg["camera"]["y_start"]
            )

        # Get number of splits
        h, w = img_nametag.shape
        num_splits = max(1, w // self.cfg["nametag"]["split_width"])
        w_split = w // num_splits

        # Get nametag's background mask
        mask = get_mask(self.img_nametag, (0, 255, 0))

        # Vertically split the nametag image
        nametag_splits = {}
        for i in range(num_splits):
            x_s = i * w_split
            x_e = (i + 1) * w_split if i < num_splits - 1 else w
            nametag_splits[f"{i+1}/{num_splits}"] = {
                "img": img_nametag[:, x_s:x_e],
                "mask": mask[:, x_s:x_e],
                "last_result": (
                    (last_result[0] + x_s, last_result[1]) if last_result else None
                ),
                "score_penalty": 0.0,
                "offset_x": x_s
            }

        # Match tempalte
        matches = []
        for tag_type, split in nametag_splits.items():
            loc, score, is_cached = find_pattern_sqdiff(
                img_roi,
                split["img"],
                last_result=split["last_result"],
                mask=split["mask"],
                global_threshold=self.cfg["nametag"]["global_diff_thres"]
            )
            w_match = split["img"].shape[1]
            h_match = split["img"].shape[0]
            score += split["score_penalty"]
            matches.append((tag_type, loc, score, w_match, h_match, is_cached, split["offset_x"]))

        # Select best match and fix offset:
        matches.sort(key=lambda x: (not x[5], x[2]))  # prefer cached, then low score
        tag_type, loc_nametag, score, w_match, h_match, is_cached, offset_x = matches[0]

        # Adjust match location back to full nametag coordinates
        loc_nametag = (loc_nametag[0] - offset_x, loc_nametag[1])
        loc_nametag = (
            loc_nametag[0] - pad_x,
            loc_nametag[1] - pad_y + self.cfg["camera"]["y_start"]
        )

        # Only update nametag location when score is good enough
        if score < self.cfg["nametag"]["diff_thres"]:
            self.loc_nametag = loc_nametag

        loc_player = (
            self.loc_nametag[0] + w // 2,
            self.loc_nametag[1] - self.cfg["nametag"]["offset"][1]
        )

        # Draw name tag detection box for debugging
        draw_rectangle(self.img_frame_debug, self.loc_nametag,
                       self.img_nametag.shape, (0, 255, 0), "")
        text = f"NameTag,{round(score, 2)}," + \
                f"{'cached' if is_cached else 'missed'}," + \
                f"{tag_type}"
        cv2.putText(self.img_frame_debug, text,
                    (self.loc_nametag[0],
                     self.loc_nametag[1] + self.img_nametag.shape[0] + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Draw player center for debugging
        cv2.circle(self.img_frame_debug,
                loc_player, radius=3,
                color=(0, 0, 255), thickness=-1)

        return loc_player

    def get_player_location_on_global_map(self):
        '''
        get_player_location_on_global_map
        '''
        
        map_h, map_w = self.img_map.shape[:2]
        minimap_h, minimap_w = self.img_minimap.shape[:2]

        # 調整 img_map 大小以匹配 img_minimap
        if map_h != minimap_h or map_w != minimap_w:
            self.img_map = cv2.resize(self.img_map, (minimap_w, minimap_h), interpolation=cv2.INTER_AREA)

        # 檢查小地圖尺寸是否有變化
        if hasattr(self, 'last_minimap_size'):
            last_w, last_h = self.last_minimap_size
            if minimap_w != last_w or minimap_h != last_h:
                print(f"注意: 小地圖尺寸已變化 {last_w}x{last_h} -> {minimap_w}x{minimap_h}")
                # 當小地圖尺寸變化時，重新調整 img_map
                self.img_map = cv2.resize(self.img_map, (minimap_w, minimap_h), interpolation=cv2.INTER_AREA)

        self.loc_minimap_global, score, _ = find_pattern_sqdiff(
                                        self.img_map,
                                        self.img_minimap)
        loc_player_global = (
            self.loc_minimap_global[0] + self.loc_player_minimap[0],
            self.loc_minimap_global[1] + self.loc_player_minimap[1]
        )

        # Draw local minimap rectangle
        camera_bottom_right = (
            self.loc_minimap_global[0] + self.img_minimap.shape[1],
            self.loc_minimap_global[1] + self.img_minimap.shape[0]
        )
        cv2.rectangle(self.img_route_debug, self.loc_minimap_global,
                      camera_bottom_right, (0, 255, 255), 1)
        cv2.putText(
            self.img_route_debug,
            f"Minimap,score({round(score, 2)})",
            (self.loc_minimap_global[0], self.loc_minimap_global[1]+15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4,
            (0, 255, 255), 1
        )

        # Draw player center
        cv2.circle(self.img_route_debug,
                   loc_player_global, radius=2,
                   color=(0, 255, 255), thickness=-1)

        return loc_player_global

    def get_nearest_color_code(self):
        '''
        Searches for the nearest color-coded action marker
        around the player on the route map.

        This function:
        - Scans each pixel in the search box to find nearest color code
        - Tracks the closest matching pixel using Manhattan distance (|dx| + |dy|).
        - Returns a dictionary containing the nearest matching
          pixel's position, color, action label, and distance.

        Returns:
            dict or None: Dictionary containing:
                - "pixel": (x, y) coordinate of the matched pixel
                - "color": matched RGB color tuple
                - "action": corresponding action string from config
                - "distance": Manhattan distance from player
            Returns None if no matching color is found within the region.
        '''
        x0, y0 = self.loc_player_global
        h, w = self.img_route.shape[:2]
        x_min = max(0, x0 - self.cfg["route"]["search_range"])
        x_max = min(w, x0 + self.cfg["route"]["search_range"])
        y_min = max(0, y0 - self.cfg["route"]["search_range"])
        y_max = min(h, y0 + self.cfg["route"]["search_range"])

        nearest = None
        min_dist = float('inf')
        for y in range(y_min, y_max):
            for x in range(x_min, x_max):
                pixel = tuple(self.img_route[y, x])  # (R, G, B)
                if pixel in self.color_code:
                    dist = abs(x - x0) + abs(y - y0)
                    if dist < min_dist:
                        min_dist = dist
                        nearest = {
                            "pixel": (x, y),
                            "color": pixel,
                            "action": self.color_code[pixel],
                            "distance": dist
                        }

        # Debug
        draw_rectangle(
            self.img_route_debug,
            (x_min, y_min),
            (self.cfg["route"]["search_range"]*2,
             self.cfg["route"]["search_range"]*2),
            (0, 0, 255), "", text_height=0.4, thickness=1,
        )
        # Draw a straigt line from map_loc_player to color_code["pixel"]
        if nearest is not None:
            cv2.line(
                self.img_route_debug,
                self.loc_player_global, # start point
                nearest["pixel"],       # end point
                (0, 255, 0),            # green line
                1                       # thickness
            )
            # Print color code on debug image
            cv2.putText(
                self.img_frame_debug, f"Route Action: {nearest['action']}",
                (720, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),
                2, cv2.LINE_AA
            )
            cv2.putText(
                self.img_frame_debug, f"Route Index: {self.idx_routes}",
                (720, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),
                2, cv2.LINE_AA
            )

        return nearest  # if not found return none

    def get_nearest_monster(self, is_left = True, overlap_threshold=0.5):
        '''
        Finds the nearest monster within the player's attack range.

        This function:
        - Defines an attack box relative to the player position,
            depending on the facing direction (`is_left`).
        - Iterates through all detected monsters and checks which ones overlap
          with the attack box.
        - Returns the closest valid monster that meets the overlap criteria.

        Args:
            is_left (bool): If True, assume the player is facing left;
                            adjusts attack box accordingly.
            overlap_threshold (float): Minimum IoU area ratio required to consider a hit.

        Returns:
            dict or None: The nearest monster's info dict, or None if no valid match.
        '''
        # Get attack box
        if self.args.attack == "aoe_skill":
            dx = self.cfg["aoe_skill"]["range_x"] // 2
            dy = self.cfg["aoe_skill"]["range_y"] // 2
            x0 = max(0, self.loc_player[0] - dx)
            x1 = min(self.img_frame.shape[1], self.loc_player[0] + dx)
            y0 = max(0, self.loc_player[1] - dy)
            y1 = min(self.img_frame.shape[0], self.loc_player[1] + dy)
        elif self.args.attack == "directional":
            if is_left:
                x0 = self.loc_player[0] - self.cfg["directional_attack"]["range_x"]
                x1 = self.loc_player[0]
            else:
                x0 = self.loc_player[0]
                x1 = x0 + self.cfg["directional_attack"]["range_x"]
            y0 = self.loc_player[1] - self.cfg["directional_attack"]["range_y"] // 2
            y1 = y0 + self.cfg["directional_attack"]["range_y"]
        else:
            logger.error(f"Unsupported attack mode: {self.args.attack}")
            return None

        # Draw attack box on debug window
        draw_rectangle(
            self.img_frame_debug, (x0, y0),
            (y1-y0, x1-x0),
            (0, 0, 255), "Attack Range"
        )

        nearest_monster = None
        min_distance = float('inf')
        for monster in self.monster_info:
            mx1, my1 = monster["position"]
            mw, mh = monster["size"]
            mx2 = mx1 + mw
            my2 = my1 + mh

            # Calculate intersection
            ix1 = max(x0, mx1)
            iy1 = max(y0, my1)
            ix2 = min(x1, mx2)
            iy2 = min(y1, my2)

            iw = max(0, ix2 - ix1)
            ih = max(0, iy2 - iy1)
            inter_area = iw * ih

            monster_area = mw * mh
            if monster_area == 0:
                continue  # skip degenerate box

            if inter_area/monster_area >= overlap_threshold:
                # Compute distance to player center
                monster_center = (mx1 + mw // 2, my1 + mh // 2)
                dx = monster_center[0] - self.loc_player[0]
                dy = monster_center[1] - self.loc_player[1]
                distance = abs(dx) + abs(dy)  # Manhattan distance

                if distance < min_distance:
                    min_distance = distance
                    nearest_monster = monster

        return nearest_monster

    def get_monsters_in_range(self, top_left, bottom_right):
        '''
        get_monsters_in_range
        '''
        x0, y0 = top_left
        x1, y1 = bottom_right

        img_roi = self.img_frame[y0:y1, x0:x1]

        monster_info = []
        for monster_name, monster_imgs in self.monsters.items():
            for img_monster, mask_monster in monster_imgs:
                if self.args.patrol:
                    pass # Don't detect monster using template in patrol mode
                elif self.cfg["monster_detect"]["mode"] == "template_free":
                    # Generate mask where pixel is exactly (0,0,0)
                    black_mask = np.all(img_roi == [0, 0, 0], axis=2).astype(np.uint8) * 255
                    cv2.imshow("Black Pixel Mask", black_mask)

                    # Shift player's location into ROI coordinate system
                    px, py = self.loc_player
                    px_in_roi = px - x0
                    py_in_roi = py - y0

                    # Define rectangle range around player (in ROI coordinate)
                    char_x_min = max(0, px_in_roi - self.cfg["character"]["width"] // 2)
                    char_x_max = min(img_roi.shape[1], px_in_roi + self.cfg["character"]["width"] // 2)
                    char_y_min = max(0, py_in_roi - self.cfg["character"]["height"] // 2)
                    char_y_max = min(img_roi.shape[0], py_in_roi + self.cfg["character"]["height"] // 2)

                    # Zero out mask inside this region (ignore player's own character)
                    black_mask[char_y_min:char_y_max, char_x_min:char_x_max] = 0

                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
                    closed_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel)
                    # cv2.imshow("Black Mask", closed_mask)

                    # draw player character bounding box

                    draw_rectangle(
                        self.img_frame_debug, (char_x_min+x0, char_y_min+y0),
                        (self.cfg["character"]["height"], self.cfg["character"]["width"]),
                        (255, 0, 0), "Character Box"
                    )

                    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed_mask, connectivity=8)

                    monster_info = []
                    min_area = 1000
                    for i in range(1, num_labels):
                        x, y, w, h, area = stats[i]
                        if area > min_area:
                            monster_info.append({
                                "name": "",
                                "position": (x0+x, y0+y),
                                "size": (h, w),
                                "score": 1.0,
                            })
                elif self.cfg["monster_detect"]["mode"] == "contour_only":
                    # Use only black lines contour to detect monsters
                    # Create masks (already grayscale)
                    mask_pattern = np.all(img_monster == [0, 0, 0], axis=2).astype(np.uint8) * 255
                    mask_roi = np.all(img_roi == [0, 0, 0], axis=2).astype(np.uint8) * 255

                    # Apply Gaussian blur (soften the masks)
                    blur = self.cfg["monster_detect"]["contour_blur"]
                    img_monster_blur = cv2.GaussianBlur(mask_pattern, (blur, blur), 0)
                    img_roi_blur = cv2.GaussianBlur(mask_roi, (blur, blur), 0)

                    # Check template vs ROI size before matching
                    h_roi, w_roi = img_roi_blur.shape[:2]
                    h_temp, w_temp = img_monster_blur.shape[:2]

                    if h_temp > h_roi or w_temp > w_roi:
                        return []  # template bigger than roi, skip this matching

                    # Perform template matching
                    res = cv2.matchTemplate(img_roi_blur, img_monster_blur, cv2.TM_SQDIFF_NORMED)

                    # Apply soft threshold
                    match_locations = np.where(res <= self.cfg["monster_detect"]["diff_thres"])

                    h, w = img_monster.shape[:2]
                    for pt in zip(*match_locations[::-1]):
                        monster_info.append({
                            "name": monster_name,
                            "position": (pt[0] + x0, pt[1] + y0),
                            "size": (h, w),
                            "score": res[pt[1], pt[0]],
                        })
                elif self.cfg["monster_detect"]["mode"] == "grayscale":
                    img_monster_gray = cv2.cvtColor(img_monster, cv2.COLOR_BGR2GRAY)
                    img_roi_gray = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)
                    res = cv2.matchTemplate(
                            img_roi_gray,
                            img_monster_gray,
                            cv2.TM_SQDIFF_NORMED,
                            mask=mask_monster)
                    match_locations = np.where(res <= self.cfg["monster_detect"]["diff_thres"])
                    h, w = img_monster.shape[:2]
                    for pt in zip(*match_locations[::-1]):
                        monster_info.append({
                            "name": monster_name,
                            "position": (pt[0] + x0, pt[1] + y0),
                            "size": (h, w),
                            "score": res[pt[1], pt[0]],
                    })
                elif self.cfg["monster_detect"]["mode"] == "color":
                    res = cv2.matchTemplate(
                            img_roi,
                            img_monster,
                            cv2.TM_SQDIFF_NORMED,
                            mask=mask_monster)
                    match_locations = np.where(res <= self.cfg["monster_detect"]["diff_thres"])
                    h, w = img_monster.shape[:2]
                    for pt in zip(*match_locations[::-1]):
                        monster_info.append({
                            "name": monster_name,
                            "position": (pt[0] + x0, pt[1] + y0),
                            "size": (h, w),
                            "score": res[pt[1], pt[0]],
                    })
                else:
                    logger.error(f"Unexpected camera localization mode: {self.cfg["monster_detect"]["mode"]}")
                    return []

        # Apply Non-Maximum Suppression to monster detection
        monster_info = nms(monster_info, iou_threshold=0.4)

        # Detect monster via health bar
        if self.cfg["monster_detect"]["with_enemy_hp_bar"]:
            # Create color mask for Monsters' HP bar
            mask = cv2.inRange(img_roi,
                               np.array(self.cfg["monster_detect"]["hp_bar_color"]),
                               np.array(self.cfg["monster_detect"]["hp_bar_color"]))

            # Find connected components (each cluster of green pixels)
            num_labels, labels, stats, centroids = \
                cv2.connectedComponentsWithStats(mask, connectivity=8)

            for i in range(1, num_labels):  # skip background (label 0)
                x, y, w, h, area = stats[i]
                if area < 3:  # small noise filter
                    continue

                # Guess a monster bounding box
                y += 10
                x = max(0, x)
                y = max(0, y)
                w = 70
                h = 70

                monster_info.append({
                    "name": "Health Bar",
                    "position": (x0 + x, y0 + y),
                    "size": (h, w),
                    "score": 1.0,
                })

        # Debug
        # Draw attack detection range
        # draw_rectangle(
        #     self.img_frame_debug, (x0, y0), (y1-y0, x1-x0),
        #     (255, 0, 0), "Monster Detection Box"
        # )

        # Draw monsters bounding box
        for monster in monster_info:
            if monster["name"] == "Health Bar":
                color = (0, 255, 255)
            else:
                color = (0, 255, 0)

            draw_rectangle(
                self.img_frame_debug, monster["position"], monster["size"],
                color, str(round(monster['score'], 2))
            )

        return monster_info

    def switch_status(self, new_status):
        '''
        Switch to new status and log the transition.

        Parameters:
        - new_status: string, the new status to switch to.
        '''
        # Ignore dummy transition
        if self.status == new_status:
            return

        t_elapsed = round(time.time() - self.t_last_switch_status)
        logger.info(f"[switch_status] From {self.status}({t_elapsed} sec) to {new_status}.")
        self.status = new_status
        self.t_last_switch_status = time.time()

    def solve_rune(self):
        '''
        Automatically solves the rune puzzle mini-game by recognizing directional arrows
        on the screen and simulating the correct key presses.

        This function:
        - Continuously checks whether the rune game is active.
        - For each of the 4 arrows:
            - Compares the cropped image with all known arrow templates.
            - Selects the best-matching arrow direction.
            - Simulates a key press corresponding to the detected arrow direction.
        - Repeats until the rune game ends.

        Note:
            - Skips key press simulation if `--disable_control` flag is set.

        Returns:
            None
        '''
        while self.is_in_rune_game():
            for arrow_idx in [0,1,2,3]:
                # Get lastest game screen frame buffer
                self.frame = self.capture.get_frame()
                # Resize game screen to 1296x759
                self.img_frame = cv2.resize(self.frame, (1296, 759),
                                            interpolation=cv2.INTER_NEAREST)

                # Crop arrow detection box
                x0, y0 = self.cfg["rune_solver"]["arrow_box_coord"]
                x = x0 + self.cfg["rune_solver"]["arrow_box_interval"] * arrow_idx
                y = y0
                size = self.cfg["rune_solver"]["arrow_box_size"]
                img_roi = self.img_frame[y:y+size, x:x+size]

                # Loop through all possible arrows template and choose the most possible one
                best_score = float('inf')
                best_direction = ""
                for direction, arrow_list in self.img_arrows.items():
                    for img_arrow in arrow_list:
                        _, score, _ = find_pattern_sqdiff(
                                        img_roi, img_arrow,
                                        mask=get_mask(img_arrow, (0, 255, 0)))
                        if score < best_score:
                            best_score = score
                            best_direction = direction
                logger.info(f"[solve_rune] Arrow({arrow_idx}) is {best_direction} with score({best_score})")

                # Update img_frame_debug
                self.img_frame_debug = self.img_frame.copy()
                draw_rectangle(
                    self.img_frame_debug, (x, y), (size, size),
                    (0, 0, 255), str(round(best_score, 2))
                )
                # Update debug window
                self.update_img_frame_debug()
                cv2.waitKey(1)

                # For logging
                screenshot(self.img_frame_debug, "solve_rune")

                # Press the key for 0.5 second
                if not self.args.disable_control:
                    self.kb.press_key(best_direction, 0.5)
                time.sleep(1)

        logger.info(f"[solve_rune] Solved all arrows")

    def is_player_stuck(self):
        """
        Checks whether the player is stuck (not moving)
        based on their global position on map.

        This function:
        - Compares the player's current position with their last known position
          tracked by the watchdog.
        - If the player has moved beyond a threshold (`watch_dog_range`),
          it resets the watchdog timer.
        - If the player hasn't moved and the elapsed time exceeds (`watch_dog_timeout`),
          it flags the player as stuck and resets the watchdog.

        Returns:
            bool: True if the player is stuck, False otherwise.
        """
        dx = abs(self.loc_player_global[0] - self.loc_watch_dog[0])
        dy = abs(self.loc_player_global[1] - self.loc_watch_dog[1])

        current_time = time.time()
        if dx + dy > self.cfg["watchdog"]["range"]:
            # Player moved, reset watchdog timer
            self.loc_watch_dog = self.loc_player_global
            self.t_watch_dog = current_time
            return False

        dt = current_time - self.t_watch_dog
        if dt > self.cfg["watchdog"]["timeout"]:
            # watch dog idle for too long, player stuck
            self.loc_watch_dog = self.loc_player_global
            self.t_watch_dog = current_time
            logger.warning(f"[is_player_stuck] Player stuck for {dt} seconds.")
            return True
        return False

    def is_rune_warning(self):
        '''
        Checks whether the rune warning icon is appear on the game frame.

        This function:
        - Crops a specific region where the rune warning icon is expected.
        - Compare the cropped region against the known rune warning image template.
        - Returns True if rune warning template matched

        Returns:
            bool: True if the rune warning is detected, False otherwise.
        '''
        x0, y0 = self.cfg["rune_warning"]["top_left"]
        x1, y1 = self.cfg["rune_warning"]["bottom_right"]

        # Debug
        draw_rectangle(
            self.img_frame_debug, (x0, y0), (y1-y0, x1-x0),
            (0, 0, 255), "")
        _, score, _ = find_pattern_sqdiff(
                        self.img_frame_gray[y0:y1, x0:x1],
                        self.img_rune_warning)
        if self.status == "hunting" and score < self.cfg["rune_warning"]["diff_thres"]:
            logger.info(f"[is_rune_warning] Detect rune warning on screen with score({score})")
            return True
        else:
            return False

    def update_rune_location(self):
        '''
        Checks if a rune icon is visible around the player's position.

        This function:
        - Uses template matching to detect the rune icon within this predefine box.

        Returns:
            nearest rune
        '''
        # Calculate bounding box
        h, w = self.img_frame.shape[:2]
        h_rune_box = self.cfg["rune_detect"]["box_height"]
        w_rune_box = self.cfg["rune_detect"]["box_width"]
        x0 = max(0, self.loc_player[0] - w_rune_box // 2)
        y0 = max(0, self.loc_player[1] - h_rune_box)
        x1 = min(w, self.loc_player[0] + w_rune_box // 2)
        y1 = min(h, self.loc_player[1])

        # Debug
        draw_rectangle(
            self.img_frame_debug, (x0, y0), (y1-y0, x1-x0),
            (255, 0, 0), "Rune Detection Range"
        )

        # Make sure ROI is large enough to hold a full rune
        max_rune_height = max(r.shape[0] for r in self.img_runes)
        max_rune_width  = max(r.shape[1] for r in self.img_runes)
        if (x1 - x0) < max_rune_width or (y1 - y0) < max_rune_height:
            return  # Skip check if box is out of range

        # Extract ROI near player
        img_roi = self.img_frame[y0:y1, x0:x1]

        # Match each rune part separately
        matches = []
        for i, img_rune in enumerate(self.img_runes):
            mask = get_mask(img_rune, (0, 255, 0))
            loc, score, _ = find_pattern_sqdiff(img_roi, img_rune, mask=mask)
            matches.append((i, loc, score, img_rune.shape))

        # # Matches box debug
        # for i, (part_idx, loc, score, shape) in enumerate(matches):
        #     draw_rectangle(
        #         self.img_frame_debug,
        #         (x0 + loc[0], y0 + loc[1]),
        #         shape,
        #         (255, 255, 0),
        #         f"{i},{round(score, 2)}",
        #         thickness=1,
        #         text_height=0.4
        #     )

        # Filter out good matches
        good_matches = [
            (i, loc, score, shape)
            for (i, loc, score, shape) in matches
            if score < self.cfg["rune_detect"]["diff_thres"]
        ]

        # Remove overlapping matches
        good_matches = nms_matches(good_matches)

        # Require at least 2 rune parts
        if len(good_matches) < 2:
            return

        # Horizontal: max distance between part's centers are small
        x_centers = [x0 + loc[0] + shape[1] // 2 for (_, loc, _, shape) in good_matches]
        if max(x_centers) - min(x_centers) > 10:
            return

        # Vertical: check if all Y's are strictly increasing
        ys = [y0 + loc[1] for (_, loc, _, _) in good_matches]
        if not all(ys[i] < ys[i + 1] for i in range(len(ys) - 1)):
            return

        logger.info(f"[Rune Detect] Found rune parts near player with scores:"
                    f" {[round(s, 2) for (_, _, s, _) in matches]}")

        # Update rune location
        self.loc_rune = (int(sum(x_centers) / len(x_centers)),
                         int(sum(ys) / len(ys)))

        # Draw all parts on debug window
        for (i, loc, score, shape) in matches:
            draw_rectangle(
                self.img_frame_debug,
                (x0 + loc[0], y0 + loc[1]),
                shape,
                (255, 0, 255),
                f"{i},{round(score, 2)}",
                text_height=0.5,
                thickness=1
            )

        # Draw rune location on debug window
        cv2.circle(self.img_frame_debug, self.loc_rune,
                   radius=5, color=(0, 255, 255), thickness=-1)

        screenshot(self.img_frame_debug, "rune_detected")

    def is_in_rune_game(self):
        '''
        Determines whether the rune puzzle game screen is currently active.

        This function:
        - Extracts the RoI where the first arrow of the rune puzzle appears.
        - Performs masked template matching with all known arrow templates.
        - Computes the best matching score across all arrow directions.
        - If the best match score is below a threshold,
          the rune puzzle is considered active.

        Returns:
            bool: True if the rune game is detected on screen, False otherwise.
        '''
        # Get lastest game screen frame buffer
        self.frame = self.capture.get_frame()
        # Resize game screen to 1296x759
        self.img_frame = cv2.resize(self.frame, (1296, 759), interpolation=cv2.INTER_NEAREST)

        # Crop arrow detection box
        x, y = self.cfg["rune_solver"]["arrow_box_coord"]
        size = self.cfg["rune_solver"]["arrow_box_size"]
        img_roi = self.img_frame[y:y+size, x:x+size]

        # Check if arrow exist on screen
        best_score = float('inf')
        for direc, arrow_list in self.img_arrows.items():
            for img_arrow in arrow_list:
                _, score, _ = find_pattern_sqdiff(
                                img_roi, img_arrow,
                                mask=get_mask(img_arrow, (0, 255, 0)))
                if score < best_score:
                    best_score = score

        draw_rectangle(
            self.img_frame_debug, (x, y), (size, size),
            (0, 0, 255), str(round(best_score, 2))
        )

        if best_score < self.cfg["rune_solver"]["arrow_box_diff_thres"]:
            logger.info(f"Arrow screen detected with score({score})")
            return True
        return False

    def is_near_edge(self):
        '''
        Detects whether the player is near a teleport edge region

        This function:
        - Defines a rectangular search region around the player's current global location.
        - Scans for pixels matching a specific edge teleport color code within the region.
        - If matching pixels are found, it computes the average X position of those pixels.
        - Compares that average to the player's X position to determine whether the edge is on the left or right.

        Returns:
            str: One of:
                - "edge on left"
                - "edge on right"
                - "" (empty string if no edge is detected nearby)
        '''
        x0, y0 = self.loc_player_global
        h, w = self.img_route.shape[:2]
        h_trigger_box = self.cfg["edge_teleport"]["trigger_box_height"]
        w_trigger_box = self.cfg["edge_teleport"]["trigger_box_width"]
        x_min = max(0, x0 - w_trigger_box//2)
        x_max = min(w, x0 + w_trigger_box//2)
        y_min = max(0, y0 - h_trigger_box//2)
        y_max = min(h, y0 + h_trigger_box//2)

        # Debug: draw search box
        # draw_rectangle(
        #     self.img_route_debug,
        #     (x_min, y_min),
        #     (y_max - y_min, x_max - x_min),
        #     (0, 0, 255), "Edge Check", thickness=1, text_height=0.4
        # )

        # Find mask of matching pixels
        roi = self.img_route[y_min:y_max, x_min:x_max]
        mask = np.all(roi == self.cfg["edge_teleport"]["color_code"], axis=2)
        coords = np.column_stack(np.where(mask))

        # No edge pixel
        if coords.size == 0:
            return ""

        # Calculate mean position of matching pixels
        mean_x = np.mean(coords[:, 1])

        # Compare to roi center
        if mean_x < x0:
            return "edge on left"
        else:
            return "edge on right"

    def get_random_action(self):
        '''
        get_random_action
        '''
        action = random.choice(list(self.color_code.values()))
        logger.warning(f"Perform random action: {action}")
        return action

    def update_info_on_img_frame_debug(self):
        '''
        update_info_on_img_frame_debug
        '''
        # Print text at bottom left corner
        self.fps = round(1.0 / (time.time() - self.t_last_frame))
        text_y_interval = 23
        text_y_start = 550
        dt_screenshot = time.time() - self.kb.t_last_screenshot
        text_list = [
            f"FPS: {self.fps}",
            f"Status: {self.status}",
            f"Resolution: {self.frame.shape[0]}x{self.frame.shape[1]}",
            f"Press 'F1' to {'pause' if self.kb.is_enable else 'start'} Bot",
            f"Press 'F2' to save screenshot{' : Saved' if dt_screenshot < 0.7 else ''}"]
        for idx, text in enumerate(text_list):
            cv2.putText(
                self.img_frame_debug, text,
                (10, text_y_start + text_y_interval*idx),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),
                2, cv2.LINE_AA
            )

        # Draw minimap rectangle on img debug
        draw_rectangle(
            self.img_frame_debug,
            self.loc_minimap,
            self.img_minimap.shape[:2],
            (0, 0, 255), "minimap",thickness=2
        )

        # Don't draw minimap in patrol mode
        if self.args.patrol:
            return

        # Compute crop region with boundary check
        crop_w, crop_h = 80, 80
        x0 = max(0, self.loc_player_global[0] - crop_w // 2)
        y0 = max(0, self.loc_player_global[1] - crop_h // 2)
        x1 = min(self.img_route_debug.shape[1], x0 + crop_w)
        y1 = min(self.img_route_debug.shape[0], y0 + crop_h)

        # Check if valid crop region
        if x1 <= x0 or y1 <= y0:
            return

        # Crop region
        mini_map_crop = self.img_route_debug[y0:y1, x0:x1]
        mini_map_crop = cv2.resize(mini_map_crop,
                                (int(mini_map_crop.shape[1] * 3),
                                 int(mini_map_crop.shape[0] * 3)),
                                interpolation=cv2.INTER_NEAREST)
        # Paste into top-right corner of self.img_frame_debug
        h_crop, w_crop = mini_map_crop.shape[:2]
        h_frame, w_frame = self.img_frame_debug.shape[:2]
        x_paste = w_frame - w_crop - 10  # 10px margin from right
        y_paste = 70
        self.img_frame_debug[y_paste:y_paste + h_crop, x_paste:x_paste + w_crop] = mini_map_crop

        # Draw border around minimap
        cv2.rectangle(
            self.img_frame_debug,
            (x_paste, y_paste),
            (x_paste + w_crop, y_paste + h_crop),
            color=(255, 255, 255),   # White border
            thickness=2
        )

    def update_img_frame_debug(self):
        '''
        update_img_frame_debug
        '''
        cv2.imshow("Game Window Debug",
            self.img_frame_debug[self.cfg["camera"]["y_start"]:
                                 self.cfg["camera"]["y_end"], :])
        # Update FPS timer
        self.t_last_frame = time.time()


        
    def channel_change(self):
        '''
        channel_change
        '''
        logger.info("[channel_change] Start")
        coords = [
            (1140, 730),  # 伺服器選單按鈕
            (1140, 666),  # 頻道按鈕
            (877, 161),  # 任意頻道
            (585, 420),  # 確定
            (877, 395),  # 登入後選角色
            (888, 275),  # 確定角色
        ]
        window_title = self.cfg["game_window"]["title"]
        for i, coord in enumerate(coords[:4]):
            click_in_game_window(window_title, coord)
            time.sleep(1)
        time.sleep(25)
        click_in_game_window(window_title, coords[4])
        time.sleep(2)
        click_in_game_window(window_title, coords[5])
        self.kb.enable()
        self.kb.set_command("stop")
        time.sleep(10) #ensure if there's no lagging during log in
        click_in_game_window(window_title, coords[4])
        time.sleep(2)
        click_in_game_window(window_title, coords[5])

    def run_once(self):
        '''
        Process one game window frame
        '''
        command = ""
        # Get window game raw frame
        self.frame = self.capture.get_frame()
        if self.frame is None:
            logger.warning("Failed to capture game frame.")
            return

        # Make sure resolution is as expected
        if self.cfg["game_window"]["size"] != self.frame.shape[:2]:
            text = f"Unexpeted window size: {self.frame.shape[:2]} (expect {self.cfg["game_window"]["size"]})"
            logger.error(text)
            return

        # Resize raw frame to (1296, 759)
        self.img_frame = cv2.resize(self.frame, (1296, 759),
                                    interpolation=cv2.INTER_NEAREST)

        # Get minimap coordinate and size on game window
        minimap_result = get_minimap_loc_size(self.img_frame)
        if minimap_result is None:
            pass
            # logger.warning("Failed to get minimap location and size.") # too verbose
        else:
            x, y, w, h = minimap_result
            self.loc_minimap = (x, y)
            self.img_minimap = self.img_frame[y:y+h, x:x+w]

        # Grayscale game window
        self.img_frame_gray = cv2.cvtColor(self.img_frame, cv2.COLOR_BGR2GRAY)

        # Image for debug use
        self.img_frame_debug = self.img_frame.copy()

        # Get current route image
        if not self.args.patrol:
            self.img_route = self.img_routes[self.idx_routes]
            self.img_route_debug = cv2.cvtColor(self.img_route, cv2.COLOR_RGB2BGR)

        # Update health monitor with current frame
        self.health_monitor.update_frame(self.img_frame[self.cfg["camera"]["y_end"]:, :])

        # Draw HP/MP/EXP bar on debug window
        ratio_bars = [self.health_monitor.hp_ratio,
                      self.health_monitor.mp_ratio,
                      self.health_monitor.exp_ratio]
        for i, bar_name in enumerate(["HP", "MP", "EXP"]):
            x_s, y_s = (250, 90)
            # Print bar ratio on debug window
            cv2.putText(self.img_frame_debug,
                        f"{bar_name}: {ratio_bars[i]*100:.1f}%",
                        (x_s, y_s + 30*i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            # Draw bar on debug window
            x_s, y_s = (410, 73)
            # print(self.health_monitor.loc_size_bars)
            x, y, w, h = self.health_monitor.loc_size_bars[i]
            self.img_frame_debug[y_s+30*i:y_s+h+30*i, x_s:x_s+w] = \
                self.img_frame[self.cfg["camera"]["y_end"]:, :][y:y+h, x:x+w]

        # Check whether "Please remove runes" warning appears on screen
        if self.is_rune_warning():
            self.loc_rune = None
            self.switch_status("finding_rune") # Stop hunting and start find runes

        # Get player location in game window
        self.loc_player = self.get_player_location_by_nametag()

        # Get player location on minimap
        loc_player_minimap = get_player_location_on_minimap(
                                self.img_minimap,
                                minimap_player_color=self.cfg["minimap"]["player_color"])
        if loc_player_minimap:
            self.loc_player_minimap = loc_player_minimap

        # Get other player location on minimap & change channel
        '''
        Detect red dot (0,0,255) and calculate the center to define as other player position.

        Needs improvement - current implementation calculates a single center value for all detected red dots. 
        When multiple players appear at once, this reduces the perceived displacement. 
        Future enhancement should cluster red dots into separate groups when multiple players are detected simultaneously.
        '''
        loc_other_players = get_all_other_player_locations_on_minimap(self.img_minimap)
        if loc_other_players:
            # Calculate center value
            xs = [x for (x, y) in loc_other_players]
            ys = [y for (x, y) in loc_other_players]
            if len(xs) == 0 or len(ys) == 0:
                return
            center_x = np.mean(xs)
            center_y = np.mean(ys)
            if np.isnan(center_x) or np.isnan(center_y):
                return
            center = (int(np.mean(xs)), int(np.mean(ys)))
            #logger.warning(f"[RedDot] Center of mass = {center}")

            # Change channel
            if self.cfg["auto_change_channel"] == "true":
                logger.warning("Player detected, immediately change channel.")
                self.kb.set_command("stop")
                self.kb.disable()
                time.sleep(1)
                self.channel_change()
                self.red_dot_center_prev = None
                return
            elif self.cfg["auto_change_channel"] == "pixel":
                if self.red_dot_center_prev is not None:
                    dx = abs(center[0] - self.red_dot_center_prev[0])
                    dy = abs(center[1] - self.red_dot_center_prev[1])
                    total = dx + dy
                    logger.debug(f"[RedDot] Movement dx={dx}, dy={dy}, total={total}")
                    if total > self.cfg["other_player_move_pixel"]:
                        logger.warning(f"Other player movement > {self.cfg["other_player_move_pixel"]}px detected, triggering channel change.")
                        self.kb.set_command("stop")
                        self.kb.disable()
                        time.sleep(1)
                        self.channel_change()
                        self.red_dot_center_prev = None
                        return
                else:
                    self.red_dot_center_prev = center
        else:
            self.red_dot_center_prev = None

        # Get player location on global map
        if self.args.patrol:
            self.loc_player_global = self.loc_player_minimap
        else:
            self.loc_player_global = self.get_player_location_on_global_map()

        # Check whether a rune icon is near player
        if self.status == "finding_rune":
            self.update_rune_location()
            if self.loc_rune is not None:
                self.switch_status("near_rune")
                logger.info(abs(self.loc_player[0] - self.loc_rune[0]))

        # Check whether we entered the rune mini-game
        if self.status == "near_rune" and (not self.args.disable_control):
            # Update rune location
            self.update_rune_location()

            dt = time.time() - self.t_last_rune_trigger
            dx = abs(self.loc_player[0] - self.loc_rune[0])
            logger.info(f"[Near Rune] Distance to rune: {dx}")

            # Check if close enough to trigger the rune
            if dt > self.cfg["rune_find"]["rune_trigger_cooldown"] and \
                dx < self.cfg["rune_find"]["rune_trigger_distance"]:

                self.kb.set_command("stop") # stop character
                time.sleep(0.1) # Wait for character to stop
                self.kb.disable() # Disable kb thread during rune solving

                # Attempt to trigger rune
                self.kb.press_key("up", 0.02)
                time.sleep(1) # Wait for rune game to pop up

                # If entered the game, start solving rune
                if self.is_in_rune_game():
                    self.solve_rune() # Blocking until runes solved
                    self.switch_status("hunting")

                # Restore kb thread
                self.kb.enable()

                self.t_last_rune_trigger = time.time()

        # Get monster search box
        margin = self.cfg["monster_detect"]["search_box_margin"]
        if self.args.attack == "aoe_skill":
            dx = self.cfg["aoe_skill"]["range_x"] // 2 + margin
            dy = self.cfg["aoe_skill"]["range_y"] // 2 + margin
        elif self.args.attack == "directional":
            dx = self.cfg["directional_attack"]["range_x"] + margin
            dy = self.cfg["directional_attack"]["range_y"] + margin
        else:
            logger.error(f"Unsupported attack mode: {self.args.attack}")
            return
        x0 = max(0, self.loc_player[0] - dx)
        x1 = min(self.img_frame.shape[1], self.loc_player[0] + dx)
        y0 = max(0, self.loc_player[1] - dy)
        y1 = min(self.img_frame.shape[0], self.loc_player[1] + dy)

        # Get monsters in the search box
        if self.status in ["hunting", "attacking"]:
            self.monster_info = self.get_monsters_in_range((x0, y0), (x1, y1))
        else:
            self.monster_info = []

        # 檢查攻擊範圍內是否有怪物，並切換狀態
        has_monsters = self.has_monsters_in_attack_range()
        
        # 狀態切換邏輯
        if has_monsters and self.status == "hunting" and command not in ["up", "down"]:
            self.switch_status("attacking")
        elif not has_monsters and self.status == "attacking":
            self.switch_status("hunting")

        # Simple combat state: if we see monsters, update the timer
        if len(self.monster_info) > 0:
            self.t_last_monster_seen = time.time()

        # Check if we should stay in place (recently saw monsters)
        should_stay_for_combat = (time.time() - self.t_last_monster_seen) < self.combat_delay

        # Get attack direction
        if self.args.attack == "aoe_skill":
            if len(self.monster_info) == 0:
                attack_direction = None
            else:
                attack_direction = "I don't care"
            nearest_monster = self.get_nearest_monster()

        elif self.args.attack == "directional":
            # Get nearest monster to player
            monster_left  = self.get_nearest_monster(is_left = True)
            monster_right = self.get_nearest_monster(is_left = False)
            # Compute distance for left
            distance_left = float('inf')
            if monster_left is not None:
                mx, my = monster_left["position"]
                mw, mh = monster_left["size"]
                center_left = (mx + mw // 2, my + mh // 2)
                distance_left = abs(center_left[0] - self.loc_player[0]) + \
                                abs(center_left[1] - self.loc_player[1])
            # Compute distance for right
            distance_right = float('inf')
            if monster_right is not None:
                mx, my = monster_right["position"]
                mw, mh = monster_right["size"]
                center_right = (mx + mw // 2, my + mh // 2)
                distance_right = abs(center_right[0] - self.loc_player[0]) + \
                                abs(center_right[1] - self.loc_player[1])
            # Choose attack direction and nearest monster
            attack_direction = None
            nearest_monster = None
            
            # Additional validation: check if monster is actually on the correct side
            def is_monster_on_correct_side(monster, direction):
                if monster is None:
                    return False
                mx, my = monster["position"]
                mw, mh = monster["size"]
                monster_center_x = mx + mw // 2
                player_x = self.loc_player[0]
                
                if direction == "left":
                    return monster_center_x < player_x  # Monster should be left of player
                else:  # direction == "right"
                    return monster_center_x > player_x  # Monster should be right of player
            
            # Only choose direction if there's a clear winner and monster is on correct side
            if monster_left is not None and monster_right is None and is_monster_on_correct_side(monster_left, "left"):
                attack_direction = "left"
                nearest_monster = monster_left
            elif monster_right is not None and monster_left is None and is_monster_on_correct_side(monster_right, "right"):
                attack_direction = "right"
                nearest_monster = monster_right
            elif monster_left is not None and monster_right is not None:
                # Both sides have monsters, check distance and side validation
                left_valid = is_monster_on_correct_side(monster_left, "left")
                right_valid = is_monster_on_correct_side(monster_right, "right")
                
                if left_valid and not right_valid:
                    attack_direction = "left"
                    nearest_monster = monster_left
                elif right_valid and not left_valid:
                    attack_direction = "right"
                    nearest_monster = monster_right
                elif left_valid and right_valid and distance_left < distance_right - 50:
                    attack_direction = "left"
                    nearest_monster = monster_left
                elif left_valid and right_valid and distance_right < distance_left - 50:
                    attack_direction = "right"
                    nearest_monster = monster_right
                # If both valid but distances too close, don't attack to avoid confusion
            
            # Debug attack direction selection
            if monster_left is not None or monster_right is not None:
                left_side_ok = is_monster_on_correct_side(monster_left, "left") if monster_left else False
                right_side_ok = is_monster_on_correct_side(monster_right, "right") if monster_right else False
                debug_text = f"L:{distance_left:.0f}({left_side_ok}) R:{distance_right:.0f}({right_side_ok}) Dir:{attack_direction}"
                cv2.putText(self.img_frame_debug, debug_text,
                           (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        if self.args.patrol:
            x, y = self.loc_player
            h, w = self.img_frame.shape[:2]
            loc_player_ratio = float(x)/float(w)
            left_ratio, right_ratio = self.cfg["patrol"]["range"]

            # Check if we need to change patrol direction
            if self.is_patrol_to_left and loc_player_ratio < left_ratio:
                self.patrol_turn_point_cnt += 1
            elif (not self.is_patrol_to_left) and loc_player_ratio > right_ratio:
                self.patrol_turn_point_cnt += 1

            if self.patrol_turn_point_cnt > self.cfg["patrol"]["turn_point_thres"]:
                self.is_patrol_to_left = not self.is_patrol_to_left
                self.patrol_turn_point_cnt = 0

            # Set command for patrol mode
            # Use proper attack range checking instead of just checking if monsters exist
            if (time.time() - self.t_patrol_last_attack > self.cfg["patrol"]["patrol_attack_interval"] and 
                len(self.monster_info) > 0 and nearest_monster is not None):
                # Check if monster is actually in attack range
                if attack_direction == "I don't care" or attack_direction == "left" or attack_direction == "right":
                    command = "attack"
                    self.t_patrol_last_attack = time.time()
            elif self.is_patrol_to_left:
                command = "walk left"
            else:
                command = "walk right"

        else:
            # get color code from img_route (只在非attacking狀態下執行)
            if self.status != "attacking":
                color_code = self.get_nearest_color_code()
                if color_code:
                    if color_code["action"] == "goal":
                        # Switch to next route map
                        self.idx_routes = (self.idx_routes+1)%len(self.img_routes)
                        logger.debug(f"Change to new route:{self.idx_routes}")
                    command = color_code["action"]

            # teleport away from edge to avoid falling off cliff
            if self.is_near_edge() and \
                time.time() - self.t_last_teleport > self.cfg["teleport"]["cooldown"]:
                command = command.replace("walk", "teleport")
                self.t_last_teleport = time.time() # update timer

        if self.cfg["key"]["teleport"] == "": # disable teleport skill
            command = command.replace("teleport", "jump")

        # Special logic for each status, overwrite color code action
        if self.status == "hunting":
            # Check MP ratio and switch to resting if too low
            if self.health_monitor.mp_ratio < 0.1:
                self.switch_status("resting")
            # Perform a random action when player stuck (but not if recently saw monsters)
            elif not self.args.patrol and self.is_player_stuck():
                command = self.get_random_action()
            elif command in ["up", "down", "jump right", "jump left"]:
                pass # Don't attack while character is on rope or jumping

        # 在attacking狀態下，只進行攻擊，不進行移動
        elif self.status == "attacking":
            if not self.args.patrol and self.is_player_stuck():
                command = self.get_random_action()          
            elif command in ["up", "down", "jump right", "jump left"]:
                pass # Don't attack while character is on rope or jumping
            elif attack_direction == "I don't care" and nearest_monster is not None and \
                time.time() - self.t_last_attack > self.cfg["directional_attack"]["cooldown"]:
                command = "attack"
                self.t_last_attack = time.time()
            elif attack_direction == "left" and nearest_monster is not None and \
                time.time() - self.t_last_attack > self.cfg["directional_attack"]["cooldown"]:
                command = "attack left"
                self.t_last_attack = time.time()
            elif attack_direction == "right" and nearest_monster is not None and \
                time.time() - self.t_last_attack > self.cfg["directional_attack"]["cooldown"]:
                command = "attack right"
                self.t_last_attack = time.time()
            else:
                command = "stop"  # 沒有可攻擊的怪物時停止動作

        elif self.status == "finding_rune":
            if self.is_player_stuck():
                command = self.get_random_action()

            # If the HP is reduced switch to hurting (other player probably help solved the rune)
            if time.time() - self.health_monitor.last_hp_reduce_time < 3:
                self.switch_status("hunting")

            # Check if finding rune timeout
            if time.time() - self.t_last_switch_status > self.cfg["rune_find"]["timeout"]:
                self.switch_status("resting")
                # TODO: terminate the script

        elif self.status == "near_rune":
            # Stay in near_rune status for only a few seconds
            if time.time() - self.t_last_switch_status > self.cfg["rune_find"]["near_rune_duration"]:
                self.switch_status("hunting")

        elif self.status == "resting":
            self.img_routes = [self.img_route_rest] # Set up resting route
            self.idx_routes = 0
            if command == "stop":
                logger.info("玩家休息中，結束程式...")
                cv2.destroyAllWindows()
                sys.exit(0)

        else:
            logger.error(f"Unknown status: {self.status}")

        # send command to keyboard controller
        self.kb.set_command(command)

        # Debug: show current command on screen
        if command and len(command) > 0:
            cv2.putText(self.img_frame_debug, f"CMD: {command}",
                       (10, 480), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Debug: show combat state
        if should_stay_for_combat:
            time_since_monster = time.time() - self.t_last_monster_seen
            cv2.putText(self.img_frame_debug, f"COMBAT WAIT: {time_since_monster:.1f}s",
                       (10, 510), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Check if need to save screenshot
        if self.kb.is_need_screen_shot:
            screenshot(self.img_frame)
            self.kb.is_need_screen_shot = False

        # Enable cached location since second frame
        self.is_first_frame = False

        #####################
        ### Debug Windows ###
        #####################
        # Don't show debug window to save system resource
        if not self.cfg["system"]["show_debug_window"]:
            return

        # Print text on debug image
        self.update_info_on_img_frame_debug()

        # Show debug image on window
        self.update_img_frame_debug()

        # Resize img_route_debug for better visualization
        if not self.args.patrol:
            self.img_route_debug = cv2.resize(
                        self.img_route_debug, (0, 0),
                        fx=self.cfg["minimap"]["debug_window_upscale"],
                        fy=self.cfg["minimap"]["debug_window_upscale"],
                        interpolation=cv2.INTER_NEAREST)
            cv2.imshow("Route Map Debug", self.img_route_debug)


    def has_monsters_in_attack_range(self):
        '''
        檢查攻擊範圍內是否有怪物
        
        Returns:
            bool: True if monsters are in attack range, False otherwise
        '''
        if self.args.attack == "aoe_skill":
            # AOE技能範圍檢測
            x0 = max(0, self.loc_player[0] - self.cfg.aoe_skill_range_x//2)
            x1 = min(self.img_frame.shape[1], self.loc_player[0] + self.cfg.aoe_skill_range_x//2)
            y0 = max(0, self.loc_player[1] - self.cfg.aoe_skill_range_y//2)
            y1 = min(self.img_frame.shape[0], self.loc_player[1] + self.cfg.aoe_skill_range_y//2)
            monsters = self.get_monsters_in_range((x0, y0), (x1, y1))
            return len(monsters) > 0
            
        elif self.args.attack == "magic_claw":
            # 魔法爪範圍檢測
            monster_left = self.get_nearest_monster(is_left=True)
            monster_right = self.get_nearest_monster(is_left=False)
            return monster_left is not None or monster_right is not None
            
        return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--disable_control',
        action='store_true',
        help='Disable simulated keyboard input'
    )

    parser.add_argument(
        '--patrol',
        action='store_true',
        help='Enable patrol mode'
    )

    # Argument to specify map name
    parser.add_argument(
        '--map',
        type=str,
        default='lost_time_1',
        help='Specify the map name'
    )

    parser.add_argument(
        "--monsters",
        type=str,
        default="evolved_ghost",
        help="Specify which monsters to load, comma-separated"
             "(e.g., --monsters green_mushroom,zombie_mushroom)"
    )

    parser.add_argument(
        '--attack',
        type=str,
        default='directional',
        help='Choose attack method, "directional", "aoe_skill"'
    )

    parser.add_argument(
        '--nametag',
        type=str,
        default='example',
        help='Choose nametag png file in nametag/'
    )

    parser.add_argument(
        '--cfg',
        type=str,
        default='edit_me',
        help='Choose customized config yaml file in config/'
    )

    try:
        mapleStoryBot = MapleStoryBot(parser.parse_args())
    except Exception as e:
        logger.error(f"MapleStoryBot Init failed: {e}")
        sys.exit(1)
    else:
        while True:
            t_start = time.time()

            # Process one game window frame
            mapleStoryBot.run_once()

            # Exit if 'q' is pressed
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            # Cap FPS to save system resource
            frame_duration = time.time() - t_start
            target_duration = 1.0 / mapleStoryBot.cfg["system"]["fps_limit_main"]
            if frame_duration < target_duration:
                time.sleep(target_duration - frame_duration)

        cv2.destroyAllWindows()
