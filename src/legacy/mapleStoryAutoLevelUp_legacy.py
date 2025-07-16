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

#
import numpy as np
import cv2

# local import
from config.legacy.config_legacy import Config
from src.utils.logger import logger
from src.utils.common import find_pattern_sqdiff, draw_rectangle, screenshot, nms, load_image, get_mask
from KeyBoardController import KeyBoardController
from GameWindowCapturorSelector import GameWindowCapturor

class MapleStoryBot:
    '''
    MapleStoryBot
    '''
    def __init__(self, args):
        self.cfg = Config # Configuration
        self.args = args
        self.status = "hunting" # 'resting', 'finding_rune', 'near_rune', 'solving_rune'
        self.idx_routes = 0 # Index of route
        self.hp_ratio = 1.0 # HP bar ratio
        self.mp_ratio = 1.0 # MP bar ratio
        self.exp_ratio = 1.0 # EXP bar ratio
        self.monster_info = [] # monster information
        self.fps = 0 # Frame per second
        self.is_first_frame = True # Disable cached location for first frame
        self.rune_detect_level = 0
        # Coordinate (top-left coordinate)
        self.loc_nametag = (0, 0) # nametag location on window
        self.loc_camera = (0, 0) # camera location on map
        self.loc_watch_dog = (0, 0) # watch dog
        self.loc_player_global = (0, 0) # player location on map
        self.loc_player = (0, 0) # player location on window
        self.loc_player_minimap = (0, 0) # Player's location on minimap
        self.loc_minimap = (0, 0)
        # Images
        self.frame = None # raw image
        self.img_frame = None # game window frame
        self.img_frame_gray = None # game window frame graysale
        self.img_frame_debug = None
        self.img_route = None # route map
        self.img_route_debug = None
        # Timers
        self.t_last_frame = time.time() # Last frame timer, for fps calculation
        self.t_last_switch_status = time.time() # Last status switches timer
        self.t_watch_dog = time.time() # Last movement timer
        self.t_last_teleport = time.time() # Last teleport timer
        self.t_patrol_last_attack = time.time() # Last patrol attack timer
        self.t_last_camera_missed = time.time() # Last camera loc missed
        # Patrol mode
        self.is_patrol_to_left = True
        self.patrol_turn_point_cnt = 0
        self.img_frame_gray_last = None

        # Set status to hunting for start
        self.switch_status("hunting")

        map_dir = "minimaps" if self.cfg.is_use_minimap else "maps"

        if args.patrol:
            # Patrol mode doesn't need map or route
            self.img_map = None
            self.img_routes = []
            self.img_route_rest = None
        else:
            # Load map for camera localization
            if self.cfg.is_use_minimap:
                self.img_map = load_image(f"{map_dir}/{args.map}/map.png",
                                        cv2.IMREAD_COLOR)
            else:
                self.img_map = load_image(f"{map_dir}/{args.map}/map.png",
                                        cv2.IMREAD_GRAYSCALE)
                self.img_map_resized = cv2.resize(
                    self.img_map, (0, 0),
                    fx=self.cfg.localize_downscale_factor,
                    fy=self.cfg.localize_downscale_factor)
            # Load route*.png images
            route_files = sorted(glob.glob(f"{map_dir}/{args.map}/route*.png"))
            route_files = [p for p in route_files if not p.endswith("route_rest.png")]
            self.img_routes = [
                cv2.cvtColor(load_image(p), cv2.COLOR_BGR2RGB) for p in route_files
            ]
            # Load rest route
            self.img_route_rest = cv2.cvtColor(
                load_image(f"{map_dir}/{args.map}/route_rest.png"), cv2.COLOR_BGR2RGB)

            # Upscale minimap route map for better debug visualization
            if self.cfg.is_use_minimap:
                img_routes_resized = []
                for img_route in self.img_routes:
                    img_routes_resized.append(cv2.resize(
                        img_route, (0, 0),
                        fx=self.cfg.minimap_upscale_factor,
                        fy=self.cfg.minimap_upscale_factor,
                        interpolation=cv2.INTER_NEAREST))
                self.img_routes = img_routes_resized
                self.img_route_rest = cv2.resize(
                            self.img_route_rest, (0, 0),
                            fx=self.cfg.minimap_upscale_factor,
                            fy=self.cfg.minimap_upscale_factor,
                            interpolation=cv2.INTER_NEAREST)

        # Load other images
        self.img_nametag = load_image("name_tag.png")
        self.img_nametag_gray = load_image("name_tag.png", cv2.IMREAD_GRAYSCALE)
        self.img_rune_warning = load_image("rune/rune_warning.png", cv2.IMREAD_GRAYSCALE)
        self.img_rune = load_image("rune/rune.png")
        self.img_rune_gray = load_image("rune/rune.png", cv2.IMREAD_GRAYSCALE)
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

    def get_minimap_location(self):
        '''
        get_minimap_location
        '''
        loc_minimap, score, is_cached = find_pattern_sqdiff(
                                            self.img_frame, self.img_map)

        return loc_minimap

    def get_player_location_on_minimap(self):
        """
        Get the player's location on the minimap by detecting a unique 4-pixel color.
        Return the player's location in minimap coordinates.
        """
        # Crop the minimap from the game screen
        x0, y0 = self.loc_minimap
        h, w, _ = self.img_route.shape
        img_minimap = self.img_frame[y0:y0 + h//4, x0:x0 + w//4]

        # Find pixels matching the player color
        mask = cv2.inRange(img_minimap,
                           self.cfg.minimap_player_color,
                           self.cfg.minimap_player_color)
        coords = cv2.findNonZero(mask)
        if coords is None or len(coords) < 4:
            logger.warning("Fail to locate player location on minimap")
            return None

        # Calculate the average location of the matching pixels
        avg = coords.mean(axis=0)[0]  # shape (1,2), so we take [0]
        loc_player_minimap = (int(round(avg[0] * self.cfg.minimap_upscale_factor)),
                              int(round(avg[1] * self.cfg.minimap_upscale_factor)))
        # Draw red circle to mark player's location on minimap
        cv2.circle(self.img_route_debug, loc_player_minimap,
                   radius=4, color=(0, 255, 255), thickness=2)

        return loc_player_minimap

    def get_nearest_color_code_on_minimap(self):
        '''
        get_nearest_color_code_on_minimap
        '''
        x0, y0 = self.loc_player_minimap
        h, w = self.img_route.shape[:2]
        x_min = max(0, x0 - self.cfg.minimap_color_code_search_range)
        x_max = min(w, x0 + self.cfg.minimap_color_code_search_range)
        y_min = max(0, y0 - self.cfg.minimap_color_code_search_range)
        y_max = min(h, y0 + self.cfg.minimap_color_code_search_range)

        nearest = None
        min_dist = float('inf')
        for y in range(y_min, y_max):
            for x in range(x_min, x_max):
                pixel = tuple(self.img_route[y, x])  # (R, G, B)
                if pixel in self.cfg.color_code:
                    dist = abs(x - x0) + abs(y - y0)
                    if dist < min_dist:
                        min_dist = dist
                        nearest = {
                            "pixel": (x, y),
                            "color": pixel,
                            "action": self.cfg.color_code[pixel],
                            "distance": dist
                        }

        # Debug
        draw_rectangle(
            self.img_route_debug,
            (x_min, y_min),
            (self.cfg.minimap_color_code_search_range*2,
             self.cfg.minimap_color_code_search_range*2),
            (0, 0, 255), "Search Range",
        )
        # Draw a straigt line from map_loc_player to color_code["pixel"]
        if nearest is not None:
            cv2.line(
                self.img_route_debug,
                self.loc_player_minimap, # start point
                nearest["pixel"],       # end point
                (0, 255, 0),            # green line
                1                       # thickness
            )

            # Print color code on debug image
            cv2.putText(
                self.img_frame_debug, f"Route Action: {nearest["action"]}",
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

    def get_nearest_monster(self, is_left = True, overlap_threshold=0.5):
        '''
        get_nearest_monster
        '''
        if is_left:
            x0 = self.loc_player[0] - self.cfg.magic_claw_range_x
        else:
            x0 = self.loc_player[0]
        y0 = self.loc_player[1] - self.cfg.magic_claw_range_y//2
        x1 = x0 + self.cfg.magic_claw_range_x
        y1 = y0 + self.cfg.magic_claw_range_y

        # Debug, magic claw hit box
        draw_rectangle(
            self.img_frame_debug, (x0, y0),
            (self.cfg.magic_claw_range_y, self.cfg.magic_claw_range_x),
            (0, 0, 255), "Attack Box"
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

    def solve_rune(self):
        '''
        Solve the rune puzzle by detecting the arrow directions and pressing corresponding keys.
        '''
        while self.is_in_rune_game():
            for arrow_idx in [0,1,2,3]:
                # Get lastest game screen frame buffer
                self.frame = self.capture.get_frame()
                # Resize game screen to 1296x759
                self.img_frame = cv2.resize(self.frame, (1296, 759),
                                            interpolation=cv2.INTER_NEAREST)

                # Crop arrow detection box
                x = self.cfg.arrow_box_start_point[0] + self.cfg.arrow_box_interval*arrow_idx
                y = self.cfg.arrow_box_start_point[1]
                size = self.cfg.arrow_box_size
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
        Detect if the player is stuck (not moving).
        If stuck for more than WATCH_DOG_TIMEOUT seconds, performs a random action.
        """
        dx = abs(self.loc_player_global[0] - self.loc_watch_dog[0])
        dy = abs(self.loc_player_global[1] - self.loc_watch_dog[1])

        current_time = time.time()
        if dx + dy > self.cfg.watch_dog_range:
            # Player moved, reset watchdog timer
            self.loc_watch_dog = self.loc_player_global
            self.t_watch_dog = current_time
            return False

        dt = current_time - self.t_watch_dog
        if dt > self.cfg.watch_dog_timeout:
            # watch dog idle for too long, player stuck
            self.loc_watch_dog = self.loc_player_global
            self.t_watch_dog = current_time
            logger.warning(f"[is_player_stuck] Player stuck for {dt} seconds.")
            return True
        return False

    def is_player_stuck_minimap(self):
        """
        Detect if the player is stuck (not moving).
        If stuck for more than WATCH_DOG_TIMEOUT seconds, performs a random action.
        """
        dx = abs(self.loc_player_minimap[0] - self.loc_watch_dog[0])
        dy = abs(self.loc_player_minimap[1] - self.loc_watch_dog[1])

        current_time = time.time()
        if dx + dy > self.cfg.watch_dog_range:
            # Player moved, reset watchdog timer
            self.loc_watch_dog = self.loc_player_minimap
            self.t_watch_dog = current_time
            return False

        dt = current_time - self.t_watch_dog
        if dt > self.cfg.watch_dog_timeout:
            # watch dog idle for too long, player stuck
            self.loc_watch_dog = self.loc_player_minimap
            self.t_watch_dog = current_time
            logger.warning(f"[is_player_stuck] Player stuck for {dt} seconds.")
            return True
        return False

    def get_nearest_color_code(self):
        '''
        get_nearest_color_code
        '''
        x0, y0 = self.loc_player_global
        h, w = self.img_route.shape[:2]
        x_min = max(0, x0 - self.cfg.color_code_search_range)
        x_max = min(w, x0 + self.cfg.color_code_search_range)
        y_min = max(0, y0 - self.cfg.color_code_search_range)
        y_max = min(h, y0 + self.cfg.color_code_search_range)

        nearest = None
        min_dist = float('inf')
        for y in range(y_min, y_max):
            for x in range(x_min, x_max):
                pixel = tuple(self.img_route[y, x])  # (R, G, B)
                if pixel in self.cfg.color_code:
                    dist = abs(x - x0) + abs(y - y0)
                    if dist < min_dist:
                        min_dist = dist
                        nearest = {
                            "pixel": (x, y),
                            "color": pixel,
                            "action": self.cfg.color_code[pixel],
                            "distance": dist
                        }

        # Debug
        draw_rectangle(
            self.img_route_debug,
            (x_min, y_min),
            (self.cfg.color_code_search_range*2, self.cfg.color_code_search_range*2),
            (0, 0, 255), "Color Search Range"
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
                self.img_frame_debug, f"Route Action: {nearest["action"]}",
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

    def get_hp_mp_exp(self):
        '''
        get_hp_mp_exp
        '''
        # HP crop
        hp_bar = self.img_frame[self.cfg.hp_bar_top_left[1]:self.cfg.hp_bar_bottom_right[1]+1,
                                self.cfg.hp_bar_top_left[0]:self.cfg.hp_bar_bottom_right[0]+1]
        # MP crop
        mp_bar = self.img_frame[self.cfg.mp_bar_top_left[1]:self.cfg.mp_bar_bottom_right[1]+1,
                                self.cfg.mp_bar_top_left[0]:self.cfg.mp_bar_bottom_right[0]+1]
        # EXP crop
        exp_bar = self.img_frame[self.cfg.exp_bar_top_left[1]:self.cfg.exp_bar_bottom_right[1]+1,
                                self.cfg.exp_bar_top_left[0]:self.cfg.exp_bar_bottom_right[0]+1]
        # HP Detection (detect empty part)
        empty_mask_hp = (hp_bar[:,:,0] == hp_bar[:,:,1]) & (hp_bar[:,:,0] == hp_bar[:,:,2])
        empty_pixels_hp = np.count_nonzero(empty_mask_hp)-6 # 6 pixel always be white
        total_pixels_hp = hp_bar.shape[0] * hp_bar.shape[1] - 6
        hp_ratio = 1 - (empty_pixels_hp / total_pixels_hp)

        # MP Detection (detect empty part)
        empty_mask_mp = (mp_bar[:,:,0] == mp_bar[:,:,1]) & (mp_bar[:,:,0] == mp_bar[:,:,2])
        empty_pixels_mp = np.count_nonzero(empty_mask_mp)-6 # 6 pixel always be white
        total_pixels_mp = mp_bar.shape[0] * mp_bar.shape[1] - 6
        mp_ratio = 1 - (empty_pixels_mp / total_pixels_mp)

        # EXP Detection (detect eexpty part)
        empty_mask_exp = (exp_bar[:,:,0] == exp_bar[:,:,1]) & (exp_bar[:,:,0] == exp_bar[:,:,2])
        eexpty_pixels_exp = np.count_nonzero(empty_mask_exp)-6 # 6 pixel always be white
        total_pixels_exp = exp_bar.shape[0] * exp_bar.shape[1] - 6
        exp_ratio = 1 - (eexpty_pixels_exp / total_pixels_exp)

        # Compute original bar dimensions
        hp_h, hp_w = hp_bar.shape[:2]
        mp_h, mp_w = mp_bar.shape[:2]
        exp_h, exp_w = exp_bar.shape[:2]

        # Overlay HP/MP/EXP text
        x_start, y_start = (250, 90)
        cv2.putText(self.img_frame_debug, f"HP: {hp_ratio*100:.1f}%", (x_start, y_start),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        cv2.putText(self.img_frame_debug, f"MP: {mp_ratio*100:.1f}%", (x_start, y_start+30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        cv2.putText(self.img_frame_debug, f"EXP: {exp_ratio*100:.1f}%", (x_start, y_start+60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        # Paste HP/MP/EXP bar on img_frame_debug
        x_start, y_start = (410, 73)
        self.img_frame_debug[y_start:y_start+hp_h, x_start:x_start+hp_w] = hp_bar
        self.img_frame_debug[y_start+30:y_start+30+mp_h, x_start:x_start+mp_w] = mp_bar
        self.img_frame_debug[y_start+60:y_start+60+exp_h, x_start:x_start+exp_w] = exp_bar

        return hp_ratio, mp_ratio, exp_ratio

    def is_rune_warning(self):
        '''
        is_rune_warning
        '''
        x0, y0 = self.cfg.rune_warning_top_left
        x1, y1 = self.cfg.rune_warning_bottom_right
        _, score, _ = find_pattern_sqdiff(
                        self.img_frame_gray[y0:y1, x0:x1],
                        self.img_rune_warning)
        if self.status == "hunting" and score < self.cfg.rune_warning_diff_thres:
            logger.info(f"[is_rune_warning] Detect rune warning on screen with score({score})")
            return True
        else:
            return False

    def is_rune_near_player(self):
        '''
        is_rune_near_player
        '''
        # Calculate bounding box
        h, w = self.img_frame.shape[:2]
        x0 = max(0, self.loc_player[0] - self.cfg.rune_detect_box_width // 2)
        y0 = max(0, self.loc_player[1] - self.cfg.rune_detect_box_height)
        x1 = min(w, self.loc_player[0] + self.cfg.rune_detect_box_width // 2)
        y1 = min(h, self.loc_player[1])

        # Debug
        draw_rectangle(
            self.img_frame_debug, (x0, y0), (y1-y0, x1-x0),
            (255, 0, 0), "Rune Detection Range"
        )

        # Find rune icon near player
        if  (x1 - x0) < self.img_rune.shape[1] or \
            (y1 - y0) < self.img_rune.shape[0]:
            return False # Skip check if box is out of range
        else:
            img_roi = self.img_frame[y0:y1, x0:x1]
            loc_rune, score, _ = find_pattern_sqdiff(
                            img_roi,
                            self.img_rune,
                            mask=get_mask(self.img_rune, (0, 255, 0)))
            # # Draw rectangle for debug
            # draw_rectangle(
            #     self.img_frame_debug,
            #     (x0 + loc_rune[0], y0 + loc_rune[1]),
            #     self.img_rune.shape,
            #     (255, 0, 255),  # purple in BGR
            #     f"Rune,{round(score, 2)}"
            # )
            detect_thres = self.cfg.rune_detect_diff_thres + self.rune_detect_level*self.cfg.rune_detect_level_coef
            if score < detect_thres:
                logger.info(f"[Rune Detect] Found rune near player with score({score})," + \
                            f"level({self.rune_detect_level}),threshold({detect_thres})")
                # Draw rectangle for debug
                draw_rectangle(
                    self.img_frame_debug,
                    (x0 + loc_rune[0], y0 + loc_rune[1]),
                    self.img_rune.shape,
                    (255, 0, 255),  # purple in BGR
                    f"Rune,{round(score, 2)}"
                )
                screenshot(self.img_frame_debug, "rune_detected")

                return True
            else:
                return False

    def is_in_rune_game(self):
        '''
        is_in_rune_game
        '''
        # Get lastest game screen frame buffer
        self.frame = self.capture.get_frame()
        # Resize game screen to 1296x759
        self.img_frame = cv2.resize(self.frame, (1296, 759), interpolation=cv2.INTER_NEAREST)

        # Crop arrow detection box
        x, y = self.cfg.arrow_box_start_point
        size = self.cfg.arrow_box_size
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

        if best_score < self.cfg.arrow_box_diff_thres:
            logger.info(f"Arrow screen detected with score({score})")
            return True
        return False

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
                elif self.cfg.monster_detect_mode == "template_free":
                    # Generate mask where pixel is exactly (0,0,0)
                    black_mask = np.all(img_roi == [0, 0, 0], axis=2).astype(np.uint8) * 255
                    cv2.imshow("Black Pixel Mask", black_mask)

                    # Shift player's location into ROI coordinate system
                    px, py = self.loc_player
                    px_in_roi = px - x0
                    py_in_roi = py - y0

                    # Define rectangle range around player (in ROI coordinate)
                    char_x_min = max(0, px_in_roi - self.cfg.character_width // 2)
                    char_x_max = min(img_roi.shape[1], px_in_roi + self.cfg.character_width // 2)
                    char_y_min = max(0, py_in_roi - self.cfg.character_height // 2)
                    char_y_max = min(img_roi.shape[0], py_in_roi + self.cfg.character_height // 2)

                    # Zero out mask inside this region (ignore player's own character)
                    black_mask[char_y_min:char_y_max, char_x_min:char_x_max] = 0

                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
                    closed_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel)
                    # cv2.imshow("Black Mask", closed_mask)

                    # draw player character bounding box

                    draw_rectangle(
                        self.img_frame_debug, (char_x_min+x0, char_y_min+y0),
                        (self.cfg.character_height, self.cfg.character_width),
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
                elif self.cfg.monster_detect_mode == "contour_only":
                    # Use only black lines contour to detect monsters
                    # Create masks (already grayscale)
                    mask_pattern = np.all(img_monster == [0, 0, 0], axis=2).astype(np.uint8) * 255
                    mask_roi = np.all(img_roi == [0, 0, 0], axis=2).astype(np.uint8) * 255

                    # Apply Gaussian blur (soften the masks)
                    img_monster_blur = cv2.GaussianBlur(mask_pattern, (self.cfg.blur_range, self.cfg.blur_range), 0)
                    img_roi_blur = cv2.GaussianBlur(mask_roi, (self.cfg.blur_range, self.cfg.blur_range), 0)

                    # Check template vs ROI size before matching
                    h_roi, w_roi = img_roi_blur.shape[:2]
                    h_temp, w_temp = img_monster_blur.shape[:2]

                    if h_temp > h_roi or w_temp > w_roi:
                        return []  # template bigger than roi, skip this matching

                    # Perform template matching
                    res = cv2.matchTemplate(img_roi_blur, img_monster_blur, cv2.TM_SQDIFF_NORMED)

                    # Apply soft threshold
                    match_locations = np.where(res <= self.cfg.monster_diff_thres)

                    h, w = img_monster.shape[:2]
                    for pt in zip(*match_locations[::-1]):
                        monster_info.append({
                            "name": monster_name,
                            "position": (pt[0] + x0, pt[1] + y0),
                            "size": (h, w),
                            "score": res[pt[1], pt[0]],
                        })
                elif self.cfg.monster_detect_mode == "grayscale":
                    img_monster_gray = cv2.cvtColor(img_monster, cv2.COLOR_BGR2GRAY)
                    img_roi_gray = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)
                    res = cv2.matchTemplate(
                            img_roi_gray,
                            img_monster_gray,
                            cv2.TM_SQDIFF_NORMED,
                            mask=mask_monster)
                    match_locations = np.where(res <= self.cfg.monster_diff_thres)
                    h, w = img_monster.shape[:2]
                    for pt in zip(*match_locations[::-1]):
                        monster_info.append({
                            "name": monster_name,
                            "position": (pt[0] + x0, pt[1] + y0),
                            "size": (h, w),
                            "score": res[pt[1], pt[0]],
                    })
                elif self.cfg.monster_detect_mode == "color":
                    res = cv2.matchTemplate(
                            img_roi,
                            img_monster,
                            cv2.TM_SQDIFF_NORMED,
                            mask=mask_monster)
                    match_locations = np.where(res <= self.cfg.monster_diff_thres)
                    h, w = img_monster.shape[:2]
                    for pt in zip(*match_locations[::-1]):
                        monster_info.append({
                            "name": monster_name,
                            "position": (pt[0] + x0, pt[1] + y0),
                            "size": (h, w),
                            "score": res[pt[1], pt[0]],
                    })
                else:
                    logger.error(f"Unexpected camera localization mode: {self.cfg.monster_detect_mode}")
                    return []

        # Apply Non-Maximum Suppression to monster detection
        monster_info = nms(monster_info, iou_threshold=0.4)

        # Detect monster via health bar
        if self.cfg.monster_detect_with_health_bar:
            # Create color mask for Monsters' HP bar
            mask = cv2.inRange(img_roi,
                               np.array(self.cfg.monster_health_bar_color),
                               np.array(self.cfg.monster_health_bar_color))

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
        draw_rectangle(
            self.img_frame_debug, (x0, y0), (y1-y0, x1-x0),
            (255, 0, 0), "Monster Detection Box"
        )

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

    def get_player_location(self):
        '''
        get player location by detecting player's nametag
        '''
        img_roi = self.img_frame_gray[self.cfg.camera_ceiling:self.cfg.camera_floor, :]

        # Pad search region to avoid edge cut-off issue (full template size)
        (pad_y, pad_x) = self.img_nametag.shape[:2]
        img_roi_padded = cv2.copyMakeBorder(
            img_roi,
            pad_y, pad_y, pad_x, pad_x,
            borderType=cv2.BORDER_REPLICATE  # replicate border for safe matching
        )

        # Adjust previous location
        if self.is_first_frame:
            last_result = None
        else:
            last_result = (
                self.loc_nametag[0] + pad_x,
                self.loc_nametag[1] - self.cfg.camera_ceiling + pad_y
            )

        # Split nametag into left and right half, detect seperately and pick highest socre
        # This localization method is more robust for occluded nametag
        h, w = self.img_nametag_gray.shape
        mask_full = get_mask(self.img_nametag, (0, 255, 0))
        nametag_variants = {
            "left": {
                "img_pattern": self.img_nametag_gray[:, :w // 2],
                "mask": mask_full[:, :w // 2],
                "last_result": last_result,
                "score_penalty": 0.0
            },
            "right": {
                "img_pattern": self.img_nametag_gray[:, w // 2:],
                "mask": mask_full[:, w // 2:],
                "last_result": (last_result[0] + w // 2, last_result[1]) if last_result else None,
                "score_penalty": 0.0
            }
        }

        # Match template for each split nametag
        matches = []
        for tag_type, data in nametag_variants.items():
            loc, score, is_cached = find_pattern_sqdiff(
                img_roi_padded,
                data["img_pattern"],
                last_result=data["last_result"],
                mask=data["mask"],
                global_threshold=0.3
            )
            w_match = data["img_pattern"].shape[1]
            h_match = data["img_pattern"].shape[0]
            score += data["score_penalty"]
            matches.append((tag_type, loc, score, w_match, h_match, is_cached))

        # Choose the best match
        matches.sort(key=lambda x: (not x[5], x[2]))
        tag_type, loc_nametag, score, w_match, h_match, is_cached = matches[0]
        if tag_type == "right":
            loc_nametag = (loc_nametag[0] - w_match, loc_nametag[1])

        # Convert back to original (unpadded) coordinates
        loc_nametag = (
            loc_nametag[0] - pad_x,
            loc_nametag[1] - pad_y + self.cfg.camera_ceiling
        )

        # Update name tag location if confidence is good
        if score < self.cfg.nametag_diff_thres:
            self.loc_nametag = loc_nametag
        loc_player = (
            self.loc_nametag[0] - self.cfg.nametag_offset[0],
            self.loc_nametag[1] - self.cfg.nametag_offset[1]
        )

        # Draw name tag detection box for debug
        draw_rectangle(
            self.img_frame_debug, self.loc_nametag, self.img_nametag.shape,
            (0, 255, 0), "")
        text = f"NameTag,{round(score, 2)}," + \
                f"{'cached' if is_cached else 'missed'}," + \
                f"{tag_type}"
        cv2.putText(self.img_frame_debug, text,
                    (self.loc_nametag[0], self.loc_nametag[1] + self.img_nametag.shape[0] + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Draw player center
        cv2.circle(self.img_frame_debug,
                loc_player, radius=3,
                color=(0, 0, 255), thickness=-1)

        return loc_player

    def get_player_location_global(self):
        '''
        get_player_location_global
        '''
        scale_factor = self.cfg.localize_downscale_factor
        # Downscale both template and search image
        img_roi = self.img_frame_gray[self.cfg.camera_ceiling:self.cfg.camera_floor, :]
        img_query = cv2.resize(img_roi, (0, 0), fx=scale_factor, fy=scale_factor)

        # Get previous frame result
        if self.is_first_frame or \
            time.time() - self.t_last_camera_missed > self.cfg.localize_cached_interval:
            last_result = None
            self.t_last_camera_missed = time.time()
        else:
            last_result = (
                int(self.loc_camera[0] * scale_factor),
                int(self.loc_camera[1] * scale_factor)
            )

        loc_camera, score, is_cached = find_pattern_sqdiff(
            self.img_map_resized,
            img_query,
            last_result=last_result,
            local_search_radius=20,
            global_threshold = 0.8)
        self.loc_camera = (
            int(loc_camera[0] / scale_factor),
            int(loc_camera[1] / scale_factor)
        )
        loc_player_global = (
            self.loc_camera[0] + self.loc_player[0],
            self.loc_camera[1] + self.loc_player[1] - self.cfg.camera_ceiling)

        # Draw camera rectangle
        camera_bottom_right = (
            self.loc_camera[0] + self.img_frame.shape[1],
            self.loc_camera[1] + self.img_frame.shape[0]
        )
        cv2.rectangle(self.img_route_debug, self.loc_camera,
                      camera_bottom_right, (0, 255, 255), 2)
        cv2.putText(
            self.img_route_debug,
            f"Camera, score={round(score, 2)}, {'cached' if is_cached else 'missed'}",
            (self.loc_camera[0], self.loc_camera[1] + 60),
            cv2.FONT_HERSHEY_SIMPLEX, 2,
            (0, 255, 0), 2
        )

        # Draw player center
        cv2.circle(self.img_route_debug,
                   loc_player_global, radius=3,
                   color=(0, 0, 255), thickness=-1)
        cv2.putText(self.img_route_debug, "Player",
                    (loc_player_global[0] - 30, loc_player_global[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        return loc_player_global

    def is_near_edge(self):
        '''
        is_near_edge
        '''
        if self.cfg.is_use_minimap:
            x0, y0 = self.loc_player_minimap
            h, w = self.img_route.shape[:2]
            x_min = max(0, x0 - self.cfg.edge_teleport_minimap_box_width//2)
            x_max = min(w, x0 + self.cfg.edge_teleport_minimap_box_width//2)
            y_min = max(0, y0 - self.cfg.edge_teleport_minimap_box_height//2)
            y_max = min(h, y0 + self.cfg.edge_teleport_minimap_box_height//2)
        else:
            x0, y0 = self.loc_player_global
            h, w = self.img_route.shape[:2]
            x_min = max(0, x0 - self.cfg.edge_teleport_box_width//2)
            x_max = min(w, x0 + self.cfg.edge_teleport_box_width//2)
            y_min = max(0, y0 - self.cfg.edge_teleport_box_height//2)
            y_max = min(h, y0 + self.cfg.edge_teleport_box_height//2)

        # Debug: draw search box
        draw_rectangle(
            self.img_route_debug,
            (x_min, y_min),
            (y_max - y_min, x_max - x_min),
            (0, 0, 255), "Edge Check"
        )

        # Find mask of matching pixels
        roi = self.img_route[y_min:y_max, x_min:x_max]
        mask = np.all(roi == self.cfg.edge_teleport_color_code, axis=2)
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
        action = random.choice(list(self.cfg.color_code.values()))
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
            f"Press 'F1' to {"pause" if self.kb.is_enable else "start"} Bot",
            f"Press 'F2' to save screenshot{" : Saved" if dt_screenshot < 0.7 else ""}"]
        for idx, text in enumerate(text_list):
            cv2.putText(
                self.img_frame_debug, text,
                (10, text_y_start + text_y_interval*idx),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),
                2, cv2.LINE_AA
            )

        # Don't draw minimap in patrol mode
        if self.args.patrol:
            return

        # mini-map on debug image
        if self.cfg.is_use_minimap:
            # Compute crop region with boundary check
            crop_w, crop_h = 300, 300
            x0 = max(0, self.loc_player_minimap[0] - crop_w // 2)
            y0 = max(0, self.loc_player_minimap[1] - crop_h // 2)
            x1 = min(self.img_route_debug.shape[1], x0 + crop_w)
            y1 = min(self.img_route_debug.shape[0], y0 + crop_h)

            # Crop region
            mini_map_crop = self.img_route_debug[y0:y1, x0:x1]
            mini_map_crop = cv2.resize(mini_map_crop,
                                    (int(mini_map_crop.shape[1] // 1.5),
                                     int(mini_map_crop.shape[0] // 1.5)),
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
        else:
            # Compute crop region with boundary check
            crop_w, crop_h = 400, 400
            x0 = max(0, self.loc_player_global[0] - crop_w // 2)
            y0 = max(0, self.loc_player_global[1] - crop_h // 2)
            x1 = min(self.img_route_debug.shape[1], x0 + crop_w)
            y1 = min(self.img_route_debug.shape[0], y0 + crop_h)

            # Crop region
            mini_map_crop = self.img_route_debug[y0:y1, x0:x1]
            mini_map_crop = cv2.resize(mini_map_crop,
                                    (mini_map_crop.shape[1] // 2, mini_map_crop.shape[0] // 2),
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
                   self.img_frame_debug[self.cfg.camera_ceiling:self.cfg.camera_floor, :])
        # Update FPS timer
        self.t_last_frame = time.time()

    def run_once(self):
        '''
        Process with one game window frame
        '''
        # Get lastest game screen frame buffer
        self.frame = self.capture.get_frame()

        # Resize game screen to 1296x759
        self.img_frame = cv2.resize(self.frame, (1296, 759), interpolation=cv2.INTER_NEAREST)

        # Grayscale game window
        self.img_frame_gray = cv2.cvtColor(self.img_frame, cv2.COLOR_BGR2GRAY)

        # Image for debug use
        self.img_frame_debug = self.img_frame.copy()

        # Get current route image
        if not self.args.patrol:
            self.img_route = self.img_routes[self.idx_routes]
            self.img_route_debug = cv2.cvtColor(self.img_route, cv2.COLOR_RGB2BGR)

        # Get minimap location
        if self.is_first_frame and self.cfg.is_use_minimap:
            self.loc_minimap = self.get_minimap_location()

        # Debug
        if self.cfg.is_use_minimap:
            h, w = self.img_map.shape[:2]
            draw_rectangle(
                self.img_frame_debug,
                self.loc_minimap,
                (h, w),
                (0, 0, 255), "minimap",thickness=1
            )

        # Detect HP/MP/EXP bar on UI
        self.hp_ratio, self.mp_ratio, self.exp_ratio = self.get_hp_mp_exp()

        # Check whether "PLease remove runes" warning appears on screen
        if self.is_rune_warning():
            self.rune_detect_level = 0
            self.switch_status("finding_rune")

        # Get player location in game window
        self.loc_player = self.get_player_location()

        # Get player location on map
        if self.cfg.is_use_minimap:
            loc_player_minimap = self.get_player_location_on_minimap()
            if loc_player_minimap:
                self.loc_player_minimap = loc_player_minimap
        else:
            if not self.args.patrol:
                self.loc_player_global = self.get_player_location_global()

        # Check whether a rune icon is near player
        if self.is_rune_near_player():
            self.switch_status("near_rune")

        # Check whether we entered the rune mini-game
        if self.status == "near_rune":
            # stop character
            self.kb.set_command("stop")
            time.sleep(0.1) # Wait for character to stop
            self.kb.disable() # Disable kb thread during rune solving

            # Attempt to trigger rune
            if not self.args.disable_control:
                self.kb.press_key("up", 0.02)
            time.sleep(1) # Wait rune game to pop up

            # If entered the game, start solving rune
            if self.is_in_rune_game():
                self.solve_rune() # Blocking until runes solved
                self.rune_detect_level = 0 # reset rune detect level
                self.switch_status("hunting")

            # Restore kb thread
            self.kb.enable()

        # Get all monster near player
        if self.args.attack == "aoe_skill":
            # Search monster near player
            x0 = max(0, self.loc_player[0] - self.cfg.aoe_skill_range_x//2)
            x1 = min(self.img_frame.shape[1], self.loc_player[0] + self.cfg.aoe_skill_range_x//2)
            y0 = max(0, self.loc_player[1] - self.cfg.aoe_skill_range_y//2)
            y1 = min(self.img_frame.shape[0], self.loc_player[1] + self.cfg.aoe_skill_range_y//2)
        elif self.args.attack == "magic_claw":
            # Search monster nearby magic claw range
            dx = self.cfg.magic_claw_range_x + self.cfg.monster_search_margin
            dy = self.cfg.magic_claw_range_y + self.cfg.monster_search_margin
            x0 = max(0, self.loc_player[0] - dx)
            x1 = min(self.img_frame.shape[1], self.loc_player[0] + dx)
            y0 = max(0, self.loc_player[1] - dy)
            y1 = min(self.img_frame.shape[0], self.loc_player[1] + dy)

        # Get monster in skill range
        self.monster_info = self.get_monsters_in_range((x0, y0), (x1, y1))

        if self.args.attack == "aoe_skill":
            if len(self.monster_info) == 0:
                attack_direction = None
            else:
                attack_direction = "I don't care"
        elif self.args.attack == "magic_claw":
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

            # Choose attack direction
            attack_direction = None
            if distance_left < distance_right:
                attack_direction = "left"
            elif distance_right < distance_left:
                attack_direction = "right"

        command = ""

        if self.args.patrol:
            x, y = self.loc_player
            h, w = self.img_frame.shape[:2]
            loc_player_ratio = float(x)/float(w)
            left_ratio, right_ratio = self.cfg.patrol_range

            # Check if we need to change patrol direction
            if self.is_patrol_to_left and loc_player_ratio < left_ratio:
                self.patrol_turn_point_cnt += 1
            elif (not self.is_patrol_to_left) and loc_player_ratio > right_ratio:
                self.patrol_turn_point_cnt += 1

            if self.patrol_turn_point_cnt > self.cfg.turn_point_thres:
                self.is_patrol_to_left = not self.is_patrol_to_left
                self.patrol_turn_point_cnt = 0

            # Set command for patrol mode
            if time.time() - self.t_patrol_last_attack > self.cfg.patrol_attack_interval:
                command = "attack"
                self.t_patrol_last_attack = time.time()
            elif self.is_patrol_to_left:
                command = "walk left"
            else:
                command = "walk right"

        else:
            # get color code from img_route
            if self.cfg.is_use_minimap:
                color_code = self.get_nearest_color_code_on_minimap()
            else:
                color_code = self.get_nearest_color_code()
            if color_code:
                if color_code["action"] == "goal":
                    # Switch to next route map
                    self.idx_routes = (self.idx_routes+1)%len(self.img_routes)
                    logger.debug(f"Change to new route:{self.idx_routes}")
                command = color_code["action"]

            # teleport away from edge to avoid fall off
            if self.is_near_edge() and \
                time.time() - self.t_last_teleport > self.cfg.teleport_cooldown:
                command = command.replace("walk", "teleport")
                self.t_last_teleport = time.time() # update timer

        # Special logic for each status, overwrite color code action
        if self.status == "hunting":
            # Perform a random action when player stuck
            if self.cfg.is_use_minimap and not self.args.patrol and \
                self.is_player_stuck_minimap():
                command = self.get_random_action()
            elif not self.cfg.is_use_minimap and not self.args.patrol and \
                self.is_player_stuck():
                command = self.get_random_action()
            elif command in ["up", "down"]:
                pass # Don't attack or heal while character is on rope
            # elif self.hp_ratio <= self.cfg.heal_ratio:
            #     command = "heal"
            # elif self.mp_ratio <= self.cfg.add_mp_ratio:
            #     command = "add mp"
            elif attack_direction == "I don't care":
                command = "attack"
            elif attack_direction == "left":
                command = "attack left"
            elif attack_direction == "right":
                command = "attack right"
            # WIP: teleport while walking is unstable
            # elif command[:4] == "walk":
            #     if self.cfg.is_use_teleport_to_walk and \
            #         time.time() - self.t_last_teleport > self.cfg.teleport_cooldown:
            #         command = command.replace("walk", "teleport")
            #         self.t_last_teleport = time.time() # update timer

        elif self.status == "finding_rune":
            if self.is_player_stuck():
                command = self.get_random_action()
            # Check if finding rune timeout
            if time.time() - self.t_last_switch_status > self.cfg.rune_finding_timeout:
                self.rune_detect_level = 0 # reset level
                self.switch_status("resting")
            # Check if need to raise level to lower the detection threshold
            self.rune_detect_level = int(time.time() - self.t_last_switch_status) // self.cfg.rune_detect_level_raise_interval

        elif self.status == "near_rune":
            # Stay in near_rune status for only a few seconds
            if time.time() - self.t_last_switch_status > self.cfg.near_rune_duration:
                self.switch_status("hunting")

        elif self.status == "resting":
            self.img_routes = [self.img_route_rest] # Set up resting route
            self.idx_routes = 0

        else:
            logger.error(f"Unknown status: {self.status}")

        # send command to keyboard controller
        self.kb.set_command(command)

        #############
        ### Debug ###
        #############
        # Print text on debug image
        self.update_info_on_img_frame_debug()

        # Show debug image on window
        self.update_img_frame_debug()

        # Check if need to save screenshot
        if self.kb.is_need_screen_shot:
            screenshot(mapleStoryBot.img_frame)
            self.kb.is_need_screen_shot = False

        # Resize img_route_debug for better visualization
        if not self.args.patrol:
            h, w = self.img_route_debug.shape[:2]
            if not self.cfg.is_use_minimap:
                self.img_route_debug = cv2.resize(self.img_route_debug, (w // 2, h // 2),
                                        interpolation=cv2.INTER_NEAREST)
            cv2.imshow("Route Map Debug", self.img_route_debug)

        # Enable cached location since second frame
        self.is_first_frame = False

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
        default='magic_claw',
        help='Choose attack method, "magic_claw", "aoe_skill"'
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
            target_duration = 1.0 / mapleStoryBot.cfg.fps_limit
            if frame_duration < target_duration:
                time.sleep(target_duration - frame_duration)

        cv2.destroyAllWindows()
