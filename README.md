# 🧠 Personal AI Trainer

This project is a real-time pose recognition system using MediaPipe and OpenCV. It allows the user to collect pose data via webcam, label it, and then use a trained K-Nearest Neighbors (KNN) classifier to predict poses in real time.

---

## 📁 Project Structure

- `train.py`: Captures pose landmarks using webcam, labels them based on user input, and stores the data in a CSV file.
- `predict.py`: Loads the CSV data, trains a KNN classifier using scikit-learn, and performs real-time prediction on live webcam feed.
- `data.csv`: Stores the captured pose landmark data and corresponding labels.

---

## 🧠 Methodology

### 🔹 Data Collection (`train.py`)
- Uses `mediapipe` to extract 33 pose landmarks per frame.
- Converts landmarks to flattened x, y, z values for each keypoint.
- User assigns a label during recording via keyboard input.
- Saves labeled data to `data.csv`.

### 🔹 Model Training and Inference (`predict.py`)
- Loads data from `data.csv` using `pandas`.
- Extracts features and labels.
- Trains a `KNeighborsClassifier` using scikit-learn.
- Captures webcam feed and uses MediaPipe to extract real-time landmarks.
- Predicts the pose label and displays it on-screen using OpenCV.

---

## 📦 Requirements

Install dependencies using:

```bash
pip install opencv-python mediapipe pandas scikit-learn
