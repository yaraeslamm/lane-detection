# # src/lane_postprocess.py
# import numpy as np
# import cv2


# class LanePostProcessor:
#     def __init__(self, alpha=0.9):
#         self.prev_left = None
#         self.prev_right = None
#         self.alpha = alpha

#     def get_lane_points(self, mask):
#         """
#         Extract centerline points from lane mask
#         """
#         h, w = mask.shape
#         left_pts, right_pts = [], []

#         for y in range(h - 1, int(h * 0.6), -3):
#             xs = np.where(mask[y] == 2)[0]
#             if len(xs) < 20:
#                 continue

#             mid = xs[len(xs) // 2]

#             left_xs = xs[xs < mid]
#             right_xs = xs[xs >= mid]

#             if len(left_xs) > 0:
#                 left_pts.append((int(np.mean(left_xs)), y))
#             if len(right_xs) > 0:
#                 right_pts.append((int(np.mean(right_xs)), y))

#         return np.array(left_pts), np.array(right_pts)

#     def fit_line(self, points):
#         """
#         Fit straight line (more stable than poly)
#         """
#         if len(points) < 8:
#             return None
#         x = points[:, 0]
#         y = points[:, 1]
#         return np.polyfit(y, x, 1)  # linear fit

#     def smooth(self, prev, curr):
#         if curr is None:
#             return prev
#         if prev is None:
#             return curr
#         return self.alpha * prev + (1 - self.alpha) * curr

#     def draw_line(self, image, line, color):
#         if line is None:
#             return image

#         h = image.shape[0]
#         ys = np.array([h, int(h * 0.6)])
#         xs = np.polyval(line, ys).astype(int)

#         cv2.line(image, (xs[0], ys[0]), (xs[1], ys[1]), color, 8)
#         return image

#     def process(self, image, class_mask):
#         left_pts, right_pts = self.get_lane_points(class_mask)

#         left_line = self.fit_line(left_pts)
#         right_line = self.fit_line(right_pts)

#         left_line = self.smooth(self.prev_left, left_line)
#         right_line = self.smooth(self.prev_right, right_line)

#         self.prev_left = left_line
#         self.prev_right = right_line

#         out = image.copy()
#         out = self.draw_line(out, left_line, (0, 255, 255))
#         out = self.draw_line(out, right_line, (0, 255, 255))

#         return out
# src/lane_postprocessor.py
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
