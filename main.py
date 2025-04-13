import cv2
import os
import face_recognition
import numpy as np
import pandas as pd
from datetime import datetime
import webbrowser

# -------------------- Config --------------------
image_folder = "images"
excel_file = "attendance_log.xlsx"
entry_exit_buffer = 10  # seconds buffer between entry and exit

# -------------------- Load known images --------------------
known_face_encodings = []
known_face_ids = []  # list of (name, roll_no)

image_files = {
    "1111.png": "Anushka 2310990376",
    "1112.png": "Jiya 2310990619",
    "1113.png": "Jhanak 2310990618",
    "1114.png": "Ishita 2310990777",
    "1115.png": "Prerika 2410991589",
    "1116.png": "Yogesh 2310991246",
    "1117.png": "Sahil 2310992524",
    "1118.png": "Tarun 2310992514",
    "1119.png": "Raghav 2310992187",
    "1120.png": "Sukrit 2310992537",
    "1121.png": "Swapnil 2410992312",
    "1122.png": "Samarth 2410992202",
    "1123.png": "Prannay 2410992179",
    "1124.png": "Saksham 2410992201"
}

print("⏳ Loading known images...")

for filename, info in image_files.items():
    path = os.path.join(image_folder, filename)
    img = cv2.imread(path)
    if img is None:
        print(f"⚠️ Could not load {filename}")
        continue

    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(rgb_img)

    if encodings:
        encoding = encodings[0]
        name, roll_no = info.split(" ", 1)
        roll_no = str(int(roll_no))  # Fix scientific notation
        known_face_encodings.append(encoding)
        known_face_ids.append((name, roll_no))
    else:
        print(f"⚠️ No face found in {filename}")

print("✅ Loaded known faces:", known_face_ids)

# -------------------- Attendance Log --------------------
reset_file = True  # Change to False if you want to keep old data

columns = ["Name", "Roll Number", "Date", "Entry Time", "Exit Time"]

if reset_file or not os.path.exists(excel_file):
    df = pd.DataFrame(columns=columns)
    df.to_excel(excel_file, index=False)



last_seen = {}

# -------------------- Attendance Function --------------------
def log_attendance(name, roll_no, mode):
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    roll_no = str(int(roll_no))  # Ensure numeric format

    df = pd.read_excel(excel_file, dtype={"Roll Number": str})

    if mode == "entry":
        # Entry not already logged for today
        mask = (df["Name"] == name) & (df["Roll Number"] == roll_no) & (df["Date"] == date_str)
        if not mask.any():
            new_row = {
                "Name": name,
                "Roll Number": roll_no,
                "Date": date_str,
                "Entry Time": time_str,
                "Exit Time": ""
            }
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            last_seen[(name, roll_no)] = now

    elif mode == "exit":
        # Update the most recent entry with missing Exit Time
        mask = (df["Name"] == name) & (df["Roll Number"] == roll_no) & \
               (df["Date"] == date_str) & (df["Exit Time"] == "")
        if mask.any():
            idx = df[mask].index[-1]
            df.at[idx, "Exit Time"] = time_str
            if (name, roll_no) in last_seen:
                del last_seen[(name, roll_no)]

    df.to_excel(excel_file, index=False)

# -------------------- Start Webcam --------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Could not open webcam.")
    exit()

print("📸 Starting camera. Press 'q' to quit.")

while True:
    success, frame = cap.read()
    if not success:
        print("❌ Failed to capture frame.")
        break

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small)
    face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

        name, roll_no = "Unknown", ""

        if True in matches:
            best_match_index = np.argmin(face_distances)
            name, roll_no = known_face_ids[best_match_index]

            now = datetime.now()

            if (name, roll_no) in last_seen:
                diff = (now - last_seen[(name, roll_no)]).total_seconds()
                if diff > entry_exit_buffer:
                    log_attendance(name, roll_no, "exit")
            else:
                log_attendance(name, roll_no, "entry")

        y1, x2, y2, x1 = [v * 4 for v in face_location]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{name} {roll_no}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Face Attendance", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Open Excel file after execution
import webbrowser
webbrowser.open(os.path.abspath(excel_file))