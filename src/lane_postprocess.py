import numpy as np
import cv2

class LaneHighlighter:
    def __init__(self, lane_color=(0, 255, 0), alpha=0.6, thickness=2):
        """
        lane_color: RGB color of the lane lines
        alpha: transparency of overlay
        thickness: thickness of drawn lane lines
        """
        self.lane_color = lane_color
        self.alpha = alpha
        self.thickness = thickness

    def highlight_lanes(self, image, class_mask):
        """
        Draws lane markings on top of the image based on the segmentation mask.
        class_mask: HxW array where 2 = lane pixels
        """
        lane_mask = (class_mask == 2).astype(np.uint8)

        # Find contours of all lanes
        contours, _ = cv2.findContours(lane_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        overlay = image.copy()

        for cnt in contours:
            if len(cnt) > 5:  # ignore tiny blobs
                cv2.polylines(overlay, [cnt], isClosed=False, color=self.lane_color, thickness=self.thickness)

        # Blend overlay with original image
        result = cv2.addWeighted(overlay, self.alpha, image, 1 - self.alpha, 0)

        return result
