import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile

st.set_page_config(page_title="SnapScan - Face Detection", layout="wide")

st.title("📸 SnapScan - Live Face Match (OpenCV Only)")

# Step 1: Upload Target Image
uploaded_image = st.file_uploader("Upload a target face image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    target_img = Image.open(uploaded_image).convert("RGB")
    target_array = np.array(target_img)
    gray_target = cv2.cvtColor(target_array, cv2.COLOR_RGB2GRAY)

    st.image(target_img, caption="Target Face", use_container_width=True)

    start_detection = st.button("Start Live Detection")

    if start_detection:
        # Save target image to temp file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        target_img.save(temp_file.name)

        # Load Haar Cascade for face detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        # Open webcam
        cap = cv2.VideoCapture(0)
        stframe = st.empty()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

            match_found = False

            for (x, y, w, h) in faces:
                face_roi = gray_frame[y:y+h, x:x+w]
                face_resized = cv2.resize(face_roi, (gray_target.shape[1], gray_target.shape[0]))

                # Template matching
                res = cv2.matchTemplate(face_resized, gray_target, cv2.TM_CCOEFF_NORMED)
                similarity = np.max(res)

                if similarity > 0.6:  # Adjust threshold for accuracy
                    color = (0, 255, 0)  # Green for match
                    label = "MATCH FOUND"
                    match_found = True
                else:
                    color = (0, 0, 255)  # Red for no match
                    label = "No Match"

                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            stframe.image(frame, channels="BGR", use_container_width=True)

            # Press "q" in terminal to stop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
