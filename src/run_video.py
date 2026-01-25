# src/run_video.py
import cv2
from moviepy.editor import VideoFileClip
from inference import RoadSegmentationModel
# from lane_postprocess import LanePostProcessor
from lane_postprocess import LaneHighlighter


MODEL_PATH = "/Users/funnyflea7/Downloads/road_model_industryV1_myDATASET.keras"
INPUT_VIDEO = "/Users/funnyflea7/Desktop/portofolio projects/in progress/lane-detection/lane-detection/to work/inpuut.mp4"
OUTPUT_VIDEO = "/Users/funnyflea7/Desktop/portofolio projects/in progress/lane-detection/lane-detection/to work/run_video_output#4.mp4"


segmenter = RoadSegmentationModel(MODEL_PATH)
# lane_processor = LanePostProcessor()
lane_drawer = LaneHighlighter(lane_color=(0, 255, 255), alpha=0.6, thickness=3)


def process_frame(frame):
    class_mask = segmenter.predict(frame)

    # optional segmentation overlay
    color_mask = frame.copy()
    color_mask[class_mask == 1] = (255, 0, 0)   # road
    color_mask[class_mask == 2] = (0, 255, 0)   # lane
    frame = cv2.addWeighted(frame, 0.6, color_mask, 0.4, 0)

    # frame = lane_processor.process(frame, class_mask)
    highlighted = lane_drawer.highlight_lanes(frame, class_mask)
    return highlighted


clip = VideoFileClip(INPUT_VIDEO, audio=False)
out = clip.fl_image(process_frame)
out.write_videofile(OUTPUT_VIDEO, audio=False)
