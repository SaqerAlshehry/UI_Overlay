import cv2
import numpy as np

class CoinCalibrator:
    def __init__(self):
        self.PHYSICAL_DIAMETER_MM = 23.0 # Saudi 1 Riyal coin
        self.history = []
        self.ema_radius = None
        self.alpha = 0.20 # EMA smoothing factor
        self.stable_frames_required = 5
        self.current_mm_per_px = None
        self.lost_frames = 0

    def update(self, gray_frame):
        """
        Takes a grayscale frame, looks for the coin, and updates the scale.
        Returns the current scale (mm/px) or None if not calibrated.
        """
        blurred = cv2.GaussianBlur(gray_frame, (7, 7), 1.5)

        circles = cv2.HoughCircles(
            blurred, 
            cv2.HOUGH_GRADIENT, 
            dp=1.2, 
            minDist=100, 
            param1=50, 
            param2=30, 
            minRadius=30, 
            maxRadius=150
        )

        if circles is not None:
            self.lost_frames = 0
            circles = np.round(circles[0, :]).astype("int")
            x, y, r = circles[0]

            self.history.append(r)
            if len(self.history) > self.stable_frames_required:
                self.history.pop(0)

            # Stability Lock: check if last 5 frames are within a 3% radius spread
            if len(self.history) == self.stable_frames_required:
                min_r = min(self.history)
                max_r = max(self.history)
                spread = (max_r - min_r) / min_r

                if spread <= 0.03: 
                    # Apply EMA Smoothing
                    if self.ema_radius is None:
                        self.ema_radius = r
                    else:
                        self.ema_radius = (self.alpha * r) + ((1 - self.alpha) * self.ema_radius)

                    # Calculate final scale
                    self.current_mm_per_px = self.PHYSICAL_DIAMETER_MM / (2 * self.ema_radius)
        else:
            self.lost_frames += 1
            if self.lost_frames > 5:
                self.current_mm_per_px = None
                self.ema_radius = None
                self.history.clear()

        return self.current_mm_per_px, circles