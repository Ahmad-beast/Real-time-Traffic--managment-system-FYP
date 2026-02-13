import cv2
import time
import serial
import serial.tools.list_ports
import numpy as np
from flask import Flask, render_template, Response, jsonify
from ultralytics import YOLO

app = Flask(__name__)

# --- CONFIGURATION ---
MIN_GREEN_TIME = 5   
MAX_GREEN_TIME = 30  
TIME_PER_CAR = 2     

# --- ARDUINO AUTO-CONNECT ---
arduino = None
try:
    ports = list(serial.tools.list_ports.comports())
    port_found = None
    for p in ports:
        if "usbmodem" in p.device or "usbserial" in p.device or "Arduino" in p.description:
            port_found = p.device
            break
    if port_found:
        arduino = serial.Serial(port_found, 9600, timeout=1)
        time.sleep(2)
        print(f"✅ Arduino Connected on {port_found}")
    else:
        print("⚠️ Arduino Not Found (Simulation Mode)")
except:
    arduino = None

model = YOLO('yolov8n.pt')

# Cameras Setup
caps = []
for idx in [0, 1, 2, 3]:
    cap = cv2.VideoCapture(idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    caps.append(cap)

# Global Variables
current_lane = 1
system_mode = "AUTO"  
lane_counts = [0, 0, 0, 0]
start_time = time.time()
allocated_time = MIN_GREEN_TIME
time_left = 0

def process_ai(frame, lane_idx):
    results = model(frame, stream=True, verbose=False, conf=0.4)
    count = 0
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            if cls in [2, 3, 5, 7]: 
                count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return count, frame

def generate_frames():
    global current_lane, lane_counts, start_time, allocated_time, time_left
    frame_skip = 0
    
    while True:
        frame_skip += 1
        run_ai = (frame_skip % 3 == 0) 
        frames = []
        
        for i, cap in enumerate(caps):
            success, frame = cap.read()
            if not success:
                frame = np.zeros((240, 320, 3), dtype=np.uint8)
                cv2.putText(frame, "CAM LOST", (80, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            else:
                frame = cv2.resize(frame, (320, 240))
                
                if run_ai:
                    cnt, frame = process_ai(frame, i)
                    lane_counts[i] = cnt
                else:
                    cv2.putText(frame, f"Cars: {lane_counts[i]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

                if (i + 1) == current_lane:
                    cv2.rectangle(frame, (0,0), (320, 240), (0, 165, 255), 4) # Orange Border
                    
            frames.append(frame)

        # LOGIC
        elapsed = time.time() - start_time
        time_left = allocated_time - elapsed

        if system_mode == "AUTO":
            if time_left <= 0:
                current_lane += 1
                if current_lane > 4: current_lane = 1
                
                cars = lane_counts[current_lane - 1]
                if cars == 0:
                    allocated_time = MIN_GREEN_TIME
                else:
                    allocated_time = min(MAX_GREEN_TIME, MIN_GREEN_TIME + (cars * TIME_PER_CAR))
                
                start_time = time.time() 
                if arduino:
                    try: arduino.write(str(current_lane).encode())
                    except: pass

        elif system_mode == "MANUAL":
            time_left = 999 

        row1 = np.hstack((frames[0], frames[1]))
        row2 = np.hstack((frames[2], frames[3]))
        grid = np.vstack((row1, row2))

        ret, buffer = cv2.imencode('.jpg', grid, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# ROUTES
@app.route('/')
def index(): return render_template('index.html')

@app.route('/video_feed')
def video_feed(): return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle_mode/<mode>')
def toggle_mode(mode):
    global system_mode, start_time, allocated_time
    
    new_mode = mode.upper()
    
    if new_mode == "AUTO" and system_mode == "MANUAL":
        start_time = time.time()
        allocated_time = MIN_GREEN_TIME 
        
    system_mode = new_mode
    return jsonify({"status": "success", "mode": system_mode})

@app.route('/manual_switch/<int:lane>')
def manual_switch(lane):
    global current_lane, system_mode, start_time, allocated_time
    if system_mode == "MANUAL":
        current_lane = lane
        start_time = time.time() 
        allocated_time = 999 
        if arduino:
            try: arduino.write(str(lane).encode())
            except: pass
        return jsonify({"status": "success"})
    return jsonify({"status": "error", "msg": "Enable Manual Mode First"})

@app.route('/get_data')
def get_data():
    return jsonify({
        "current_lane": current_lane,
        "time_left": int(time_left) if time_left < 900 else "∞",
        "counts": lane_counts,
        "mode": system_mode
    })

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)