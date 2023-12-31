import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import time
import pyrealsense2 as rs
import numpy as np

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()


# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))
# device_as_playback = device.as_playback()

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start realsense streaming pipeline
pipeline.start(config)

# configure mediapipe FaceLandmarker configurations
MODEL_PATH = './face_landmarker.task'

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

def print_result(result: FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    print('face landmarker result: {}'.format(result))

face_landmarker_options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode = VisionRunningMode.LIVE_STREAM,
    result_callback=print_result
)

try:
    with FaceLandmarker.create_from_options(face_landmarker_options) as landmarker:
        # while True:
        # The landmarker is initialized. Use it here.

        # grab realsense frames
            frames = pipeline.wait_for_frames()
            # device_as_playback.pause()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            # if not depth_frame or not color_frame:
            #     continue
            
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            depth_colormap_dim = depth_colormap.shape
            color_colormap_dim = color_image.shape

            # If depth and color resolutions are different, resize color image to match depth image for display
            if depth_colormap_dim != color_colormap_dim:
                color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
                images = np.hstack((color_image, depth_colormap))
            else:
                images = np.hstack((color_image, depth_colormap))

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=color_image)
            print("timestamp: {}", round(time.perf_counter() * 1000))
            
            landmarker.detect_async(mp_image, round(time.perf_counter() * 1000))

            cv2.namedWindow('realSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('realSense', images)
            cv2.waitKey()
            # if cv2.waitKey() & 0xFF == ord('q'):
            #     break
            
            # device_as_playback.resume()
finally:
    # Stop streaming
    pipeline.stop()