import streamlit as st
import cv2
import numpy as np
import tempfile
import pandas as pd

st.set_page_config(layout="wide")

st.title("🚗 Smart Traffic Analytics Dashboard")
st.caption("Real-time AI-based Traffic Monitoring System")

# Layout
col1, col2 = st.columns([2,1])

uploaded_file = st.file_uploader("Upload Traffic Video", type=["mp4"])

video_path = "traffic.mp4"

if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

cap = cv2.VideoCapture(video_path)
fgbg = cv2.createBackgroundSubtractorMOG2()

# Counters
total_count = 0
car, bike, bus, truck = 0, 0, 0, 0

line_y = 300

frame_display = col1.empty()
stats_display = col2.empty()

# To avoid duplicate counting
detected_ids = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (800, 600))

    fgmask = fgbg.apply(frame)

    kernel = np.ones((5,5), np.uint8)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area > 800:
            x, y, w, h = cv2.boundingRect(cnt)
            cx = x + w // 2
            cy = y + h // 2

            # Unique ID (simple approximation)
            object_id = (cx, cy)

            # Classification (based on width)
            if w < 40:
                label = "Bike"
                vehicle_type = "bike"
            elif w < 80:
                label = "Car"
                vehicle_type = "car"
            elif w < 120:
                label = "Bus"
                vehicle_type = "bus"
            else:
                label = "Truck"
                vehicle_type = "truck"

            # COUNT ONLY WHEN CROSSING LINE (FIXED LOGIC)
            if abs(cy - line_y) < 5 and object_id not in detected_ids:
                detected_ids.append(object_id)
                total_count += 1

                if vehicle_type == "car":
                    car += 1
                elif vehicle_type == "bike":
                    bike += 1
                elif vehicle_type == "bus":
                    bus += 1
                elif vehicle_type == "truck":
                    truck += 1

            # Draw box
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(frame, label, (x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

    # Draw counting line
    cv2.line(frame, (0,line_y), (800,line_y), (0,0,255), 2)

    # Show total
    cv2.putText(frame, f"Total: {total_count}", (10,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

    # Display video
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_display.image(frame)

    # 📊 Dashboard
    data = {
        "Vehicle": ["Car", "Bike", "Bus", "Truck"],
        "Count": [car, bike, bus, truck]
    }

    df = pd.DataFrame(data)

    with stats_display.container():
        st.subheader("📊 Live Stats")
        st.write(f"Total Vehicles: {total_count}")
        st.bar_chart(df.set_index("Vehicle"))

cap.release()
cv2.destroyAllWindows()