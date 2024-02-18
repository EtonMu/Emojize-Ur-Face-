# classify.py
import cv2
import joblib
import mediapipe as mp

# Load the pre-trained model
model_path = "emotion_detection_model.pkl"
model = joblib.load(model_path)

# Initialize MediaPipe solutions
mp_solutions = mp.solutions
face_detection = mp_solutions.face_detection.FaceDetection(min_detection_confidence=0.5)
face_mesh = mp_solutions.face_mesh.FaceMesh(static_image_mode=True)

def get_face_bbox(frame):
    """Detect face in the frame and return bounding box."""
    results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            return bbox
    return None

def preprocess_face(frame, bbox):
    """Crop and preprocess face for the model."""
    x, y, w, h = bbox
    face = frame[y:y+h, x:x+w]
    face_resized = cv2.resize(face, (48, 48))
    face_grayscale = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
    return face_grayscale.flatten()

def classify(frame):
    """Classify the frame to return an emoji index."""
    bbox = get_face_bbox(frame)
    if bbox:
        face_data = preprocess_face(frame, bbox)
        emoji_index = model.predict([face_data])[0]
        return emoji_index
    return None

# Example usage
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        emoji_index = classify(frame)
        if emoji_index is not None:
            print(f"Emoji Index: {emoji_index}")
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
