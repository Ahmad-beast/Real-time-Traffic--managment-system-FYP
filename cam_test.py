import cv2
import math
import time
import numpy as np
from ultralytics import YOLO

# --- Setup ---
# Camera Settings (0 internal, 1 external)
cap = cv2.VideoCapture(0) 
cap.set(3, 1280) # Width
cap.set(4, 720)  # Height

# YOLO Model Load karein
print("Model load ho raha hai, thora intezar karein...")
model = YOLO("yolov8n.pt") 

# Sirf Car detect karne ke liye Class Names (COCO Dataset)
# Index 2 'car' hai, 3 'motorcycle', 5 'bus', 7 'truck'
target_class_id = 2  # Sirf Car ke liye

# --- UI Setup ---
def nothing(x):
    pass

cv2.namedWindow("Camera & Settings")
cv2.createTrackbar("Brightness", "Camera & Settings", 110, 200, nothing) # 100 is normal
cv2.createTrackbar("Contrast", "Camera & Settings", 12, 30, nothing)     # 10 is normal (1.0)
cv2.createTrackbar("Saturation", "Camera & Settings", 60, 150, nothing)  # 50 is normal

# --- Helper Function for Image Enhancement ---
def apply_enhancements(img, brightness_val, contrast_val, saturation_val):
    # 1. Contrast aur Brightness (Efficient way)
    alpha = contrast_val / 10.0
    beta = brightness_val - 100
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    
    # 2. Saturation (Colors ko tez karne ke liye)
    # 50 se upar value ho to saturation badhayenge
    sat_offset = saturation_val - 50
    if sat_offset != 0:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        s = cv2.add(s, sat_offset) # Saturation add ki
        final_hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    
    return img

# --- Main Loop ---
p_time = 0

while True:
    success, frame = cap.read()
    if not success:
        print("Camera frame nahi mila!")
        break

    # 1. Sliders ki values read karein
    b_val = cv2.getTrackbarPos("Brightness", "Camera & Settings")
    c_val = cv2.getTrackbarPos("Contrast", "Camera & Settings")
    s_val = cv2.getTrackbarPos("Saturation", "Camera & Settings")

    # 2. Frame ko Enhance karein (Dullness fix)
    frame = apply_enhancements(frame, b_val, c_val, s_val)

    # 3. YOLO Detection
    results = model(frame, stream=True, verbose=False) # stream=True fast hota hai

    car_count = 0

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Class ID check karein
            cls = int(box.cls[0])
            conf = math.ceil((box.conf[0] * 100)) / 100

            # Agar object 'Car' hai aur confidence > 0.4 hai
            if cls == target_class_id and conf > 0.4:
                car_count += 1
                
                # Bounding Box draw karein
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Box aur Text
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'Car {conf}', (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 4. FPS Calculation
    c_time = time.time()
    fps = 1 / (c_time - p_time)
    p_time = c_time

    # 5. Dashboard Display (Car Count & FPS)
    # Background strip for text
    cv2.rectangle(frame, (0, 0), (250, 110), (0, 0, 0), -1) 
    
    cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, f'Total Cars: {car_count}', (10, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # Settings info
    cv2.putText(frame, f'B:{b_val} C:{c_val} S:{s_val}', (10, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow("Camera & Settings", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()







# import cv2

# def list_cameras():
#     # 0 se 5 tak ports check karenge
#     for i in range(5):
#         cap = cv2.VideoCapture(i)
#         if cap.isOpened():
#             print(f"Camera Index {i}: Available ✅")
#             cap.release()
#         else:
#             print(f"Camera Index {i}: Not Available ❌")

# print("Checking cameras...")
# list_cameras()