import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np

st.title("Deteksi Orang Pake YOLO")
st.markdown("Pakenya pretrained model, gaada yang keren, ini cuman implementasinya aja, males ngetrain datanya lama")
st.markdown("cek github ane buat source codenya hehehe")

st.sidebar.title("Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5)

model = YOLO("yolo-Weights/yolov8n.pt")

classNames = ["person"]

st.sidebar.info("Click 'Start' to initiate the webcam.")
start_detection = st.sidebar.button("Start Detection")
stop_detection = st.sidebar.button("Stop Detection")  # Stop button

if start_detection:
    stframe = st.empty()

    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    previous_person_count = -1  

    while True:
        success, img = cap.read()
        if not success:
            st.error("Failed to access the webcam.")
            break

        results = model(img, stream=True)

        person_count = 0

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                confidence = float(box.conf[0])
                if confidence >= confidence_threshold:
                    cls = int(box.cls[0])

                    if classNames[cls] == "person":
                        person_count += 1

                        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                        cv2.putText(
                            img,
                            f"{classNames[cls]} {confidence:.2f}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 0, 0),
                            2,
                        )

        if person_count != previous_person_count:
            st.sidebar.info(f"Persons detected: {person_count}")
            previous_person_count = person_count

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        stframe.image(img_rgb, channels="RGB", use_column_width=True)

        if stop_detection:
            st.warning("Camera stopped.")
            break

    cap.release()
    cv2.destroyAllWindows()
