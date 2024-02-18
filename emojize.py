# emojize.py
import cv2
from classify import classify

# Function to add overlay text for instructions
def put_instruction_text(frame, text, position, font=cv2.FONT_HERSHEY_SIMPLEX,
                         font_scale=0.7, font_thickness=2, color=(255, 255, 255)):
    cv2.putText(frame, text, position, font, font_scale, color, font_thickness, cv2.LINE_AA)

def overlay_image(background, overlay, x, y):
    # Function to overlay an image on background without considering transparency
    overlay_height, overlay_width = overlay.shape[:2]
    background_height, background_width = background.shape[:2]

    if x >= background_width or y >= background_height:
        return background

    if x + overlay_width > background_width:
        overlay_width = background_width - x
        overlay = overlay[:, :overlay_width]

    if y + overlay_height > background_height:
        overlay_height = background_height - y
        overlay = overlay[:overlay_height]

    background[y:y+overlay_height, x:x+overlay_width] = overlay
    return background

# Set up the window
window_name = 'Emojize Ur Face!'
cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

cap = cv2.VideoCapture(0)
prev_emoji_index = None  # Initialize with None for the first frame

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    emoji_index = classify(frame_rgb)

    # Add instructions on the screen with a semi-transparent background
    overlay = frame.copy()
    alpha = 0.4  # Transparency factor.
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    put_instruction_text(frame, "Press 'q' to quit", (10, 30), font_scale=0.7, color=(255, 255, 255))

    if emoji_index != prev_emoji_index and emoji_index is not None:
        emoji_img = cv2.imread(f"output/emoji_{emoji_index}.jpg")  # Adjusted to load JPG
        if emoji_img is None:
            print(f"Failed to load image at path: output/emoji_{emoji_index}.jpg")
            continue
        prev_emoji_index = emoji_index

    if prev_emoji_index is not None and emoji_img is not None:
        frame = overlay_image(frame, emoji_img, 10, 50)  # Adjusted function call

    cv2.imshow(window_name, frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
