'''
Execute this script:
python mapleStoryAutoLevelUp.py --map cloud_balcony --monster brown_windup_bear,pink_windup_bear
'''
# Standard import
import time
import threading

# Libarary Import
import mss
import cv2
import numpy as np

# local import
from src.utils.logger import logger
from src.utils.common import get_game_window_title_by_token, load_image, resize_window, get_window_rect

class GameWindowCapturor:
    '''
    GameWindowCapturor using mss for better VMware compatibility
    '''
    def __init__(self, cfg, test_image_name = None):
        self.cfg = cfg
        self.frame = None
        self.lock = threading.Lock()
        self.is_terminated = False
        self.fps = 0
        self.fps_limit = cfg["system"]["fps_limit_window_capturor"]
        self.t_last_run = 0.0
        self.capture_control = None
        self.window_title = ""
        self.window_rect = None
        self.capture_thread = None
        self.sct = None

        # If use test image as input, disable the whole capture thread
        if test_image_name is not None:
            self.frame = load_image(f"test/{test_image_name}.png")
            return

        # Get game window title
        self.window_title = get_game_window_title_by_token(cfg["game_window"]["title"])

        resize_window(self.window_title)
        
        if self.window_title is None:
            raise RuntimeError(
                f"[GameWindowCapturor] Unable to find window title containing: {cfg['game_window']['title']}"
            )
        else:
            logger.info(f"[GameWindowCapturor] Found game window title: {self.window_title}")

        # Get window rectangle for mss capture
        self.window_rect = get_window_rect(self.window_title)
        if self.window_rect is None:
            raise RuntimeError(
                f"[GameWindowCapturor] Unable to get window rectangle for: {self.window_title}"
            )

        # Create mss instance
        self.sct = mss.mss()

        # Start capturing thread
        self.is_terminated = False
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()

        logger.info("[GameWindowCapturor] Init done")

    def _capture_loop(self):
        '''
        Main capture loop running in separate thread
        '''
        while not self.is_terminated:
            try:
                # Update window rectangle in case window moved
                current_rect = get_window_rect(self.window_title)
                if current_rect:
                    self.window_rect = current_rect

                # Capture screenshot
                screenshot = self.sct.grab(self.window_rect)
                
                # Convert to numpy array and then to BGR format
                frame = np.array(screenshot)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                
                # Store frame with lock
                with self.lock:
                    self.frame = frame
                
                self.limit_fps()
                
            except Exception as e:
                logger.error(f"[GameWindowCapturor] Capture error: {e}")
                time.sleep(0.1)  # Brief pause on error

    def get_frame(self):
        '''
        Safely get latest game window frame.
        '''
        with self.lock:
            if self.frame is None:
                return None
            return self.frame.copy()

    def stop(self):
        '''
        Stop capturing thread
        '''
        self.is_terminated = True
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
        if self.sct:
            self.sct.close()
        logger.info("[GameWindowCapturor] Terminated")

    def limit_fps(self):
        '''
        Limit FPS
        '''
        # If the loop finished early, sleep to maintain target FPS
        target_duration = 1.0 / self.fps_limit  # seconds per frame
        frame_duration = time.time() - self.t_last_run
        if frame_duration < target_duration:
            time.sleep(target_duration - frame_duration)

        # Update FPS
        self.fps = round(1.0 / (time.time() - self.t_last_run))
        self.t_last_run = time.time()
        # logger.info(f"FPS = {self.fps}")
