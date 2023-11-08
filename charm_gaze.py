import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import time

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

with FaceLandmarker.create_from_options(face_landmarker_options) as landmarker:
    # The landmarker is initialized. Use it here.
    webcam_video_capture = cv2.VideoCapture(0)
    while webcam_video_capture.isOpened():
        success, cv_image = webcam_video_capture.read()
        if not success:                            # no frame input
            print("Ignoring empty camera frame.")
            continue
        cv_image.flags.writeable = False
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv_image)
        print("timestamp: {}", round(time.perf_counter() * 1000))
        
        landmarker.detect_async(mp_image, round(time.perf_counter() * 1000))
        cv2.imshow('base cam', cv_image)
        cv2.waitKey(1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    webcam_video_capture.release()