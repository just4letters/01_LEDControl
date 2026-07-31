import os
import urllib.request
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles

print("🎬 Starting Modern MediaPipe Tasks API Demo...")

# 1. Download the required AI Model File if it doesn't exist
model_path = 'face_landmarker.task'
if not os.path.exists(model_path):
    print("⏳ Downloading face_landmarker.task model...")
    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    urllib.request.urlretrieve(url, model_path)
    print("✅ Download complete!")

# 2. Configure the Face Landmarker Options
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1,
    running_mode=vision.RunningMode.IMAGE
)

# 3. Create the Landmarker instance
detector = vision.FaceLandmarker.create_from_options(options)

# Open the default webcam
cap = cv2.VideoCapture(0)
print("Press the 'ESC' key to close the window.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # 4. Convert the OpenCV BGR image to RGB, then to MediaPipe's Image object
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # 5. Process the image using the detector
    detection_result = detector.detect(mp_image)

    # 6. Draw the results back onto the original frame
    if detection_result.face_landmarks:
        for face_landmarks in detection_result.face_landmarks:
            drawing_utils.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_styles.get_default_face_mesh_tesselation_style()
            )

    # Flip the image horizontally for a selfie-view display
    cv2.imshow('Modern Tasks API Demo', cv2.flip(frame, 1))
    
    # Exit if 'ESC' is pressed
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()