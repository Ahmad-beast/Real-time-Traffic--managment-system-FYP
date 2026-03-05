# 🚦 AI-Powered Smart Traffic Management System

![Project Status](https://img.shields.io/badge/Status-Active-success)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![YOLOv8](https://img.shields.io/badge/AI-YOLOv8-orange)
![Arduino](https://img.shields.io/badge/Hardware-Arduino-teal)

## 📜 Project Overview
This project is an **Intelligent Traffic Management System** designed to reduce congestion by optimizing traffic signal timers in real-time. 

It uses **Computer Vision (YOLOv8)** to detect vehicle density across 4 lanes (using 4 camera feeds) and dynamically adjusts the Green Light duration based on traffic load. It also includes a **Web Dashboard** for monitoring and manual control.

## ✨ Key Features
* **Real-time Vehicle Detection:** Uses YOLOv8 to count Cars, Buses, Trucks, and Bikes.
* **Dynamic Signal Timing:**
    * *High Traffic:* Increases Green light duration (Max 30s).
    * *Low Traffic:* Reduces Green light duration (Min 5s).
* **4-Way Camera Feed:** Stitches 4 camera inputs into a single grid view.
* **Dual Modes:**
    * **AUTO Mode:** AI fully controls the traffic lights.
    * **MANUAL Mode:** User can control lights via the Web Dashboard.
* **IoT Integration:** Communicates with **Arduino** via Serial to switch physical traffic lights.
* **Web Dashboard:** Built with Flask, HTML/CSS (Bootstrap) for live monitoring.

## 🛠️ Tech Stack
### Software
* **Python:** Core logic.
* **OpenCV:** Image processing and video capture.
* **Ultralytics YOLO:** Object detection model.
* **Flask:** Web server for the dashboard.
* **NumPy:** Matrix operations.

### Hardware
* **Arduino Uno:** Microcontroller for lights.
* **LEDs:** Red, Green, Yellow LEDs (representing traffic signals).
* **Webcams:** 4 Cameras (or simulated video feeds).

---

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone [[https://github.com/your-username/smart-traffic-system.git](https://github.com/your-username/smart-traffic-system.git)](https://github.com/Ahmad-beast/Real-time-Traffic--managment-system-FYP.git)

cd smart-traffic-system
