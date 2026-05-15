import os
import json
import cv2
import mediapipe as mp
import numpy as np
from hindi_dict import pose_to_hindi

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arccos(np.clip(np.dot(a-b, c-b) / (np.linalg.norm(a-b) * np.linalg.norm(c-b)), -1.0, 1.0))
    return np.degrees(radians)

def analyze_pose(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = pose.process(img_rgb)

    if not result.pose_landmarks:
        return {"error": "No human pose detected."}

    landmarks = result.pose_landmarks.landmark
    h, w = img.shape[:2]
    points = [(int(l.x * w), int(l.y * h)) for l in landmarks]

    # Calculate angles for all major joints
    angles = {}
    missing_parts = []
    # Define which joints belong to which body part
    joint_groups = {
        'arms': ['left_elbow', 'right_elbow', 'left_shoulder', 'right_shoulder', 'left_wrist', 'right_wrist'],
        'legs': ['left_knee', 'right_knee', 'left_hip', 'right_hip', 'left_ankle', 'right_ankle', 'left_heel', 'right_heel'],
        'torso': ['neck', 'torso']
    }
    # Try to calculate each angle, mark as missing if fails or if landmark is out of frame
    angle_defs = {
        "left_elbow": [11, 13, 15],
        "right_elbow": [12, 14, 16],
        "left_shoulder": [13, 11, 23],
        "right_shoulder": [14, 12, 24],
        "left_knee": [23, 25, 27],
        "right_knee": [24, 26, 28],
        "left_hip": [11, 23, 25],
        "right_hip": [12, 24, 26],
        "left_wrist": [13, 15, 17],
        "right_wrist": [14, 16, 18],
        "left_ankle": [25, 27, 31],
        "right_ankle": [26, 28, 32],
        "left_heel": [27, 29, 31],
        "right_heel": [28, 30, 32],
        "neck": [0, 11, 12],
        "torso": [11, 23, 24],
    }
    # Helper: check if a landmark is in frame (confidence > 0.5 and inside image)
    def is_landmark_valid(idx):
        l = landmarks[idx]
        return (l.visibility > 0.5 and 0 <= l.x <= 1 and 0 <= l.y <= 1)
    for key, idxs in angle_defs.items():
        if all(is_landmark_valid(i) for i in idxs):
            try:
                angles[key] = calculate_angle(points[idxs[0]], points[idxs[1]], points[idxs[2]])
            except Exception:
                angles[key] = None
        else:
            angles[key] = None
    # Check which body parts are missing (if more than half the joints are missing, mark as missing)
    for part, joints in joint_groups.items():
        missing = sum(1 for j in joints if angles[j] is None)
        if missing >= len(joints) // 2:
            missing_parts.append(part)
    # If any major part is missing, return a message
    if missing_parts:
        msg = "Could not detect: " + ", ".join(missing_parts) + ". "
        msg += "Please ensure your entire body is visible in the frame."
        # Draw landmarks on the image
        mp_drawing = mp.solutions.drawing_utils
        annotated_img = img.copy()
        mp_drawing.draw_landmarks(annotated_img, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        annotated_path = image_path.replace('.jpg', '_landmarks.jpg').replace('.png', '_landmarks.png')
        cv2.imwrite(annotated_path, annotated_img)
        return {
            "pose_name": "Partial Detection",
            "pose_hindi": "आंशिक पहचान",
            "accuracy": 0,
            "corrections": [msg],
            "annotated_image": annotated_path
        }

    print("Detected angles:", angles)

    # Draw landmarks on the image
    mp_drawing = mp.solutions.drawing_utils
    annotated_img = img.copy()
    mp_drawing.draw_landmarks(annotated_img, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    # Save annotated image
    annotated_path = image_path.replace('.jpg', '_landmarks.jpg').replace('.png', '_landmarks.png')
    cv2.imwrite(annotated_path, annotated_img)

    # Use absolute path for ideal_poses directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    ideal_dir = os.path.join(base_dir, '..', 'ideal_poses')
    min_diff = float('inf')
    matched_pose = None
    corrections = []

    # Improved matching: use strict threshold and weighted sum
    buffer = 10  # degrees buffer for all detections
    for filename in os.listdir(ideal_dir):
        with open(os.path.join(ideal_dir, filename)) as f:
            ideal_angles = json.load(f)
        pose_name = filename.replace('_angles.json', '').replace('_', ' ').title()
        total_diff = 0
        temp_corrections = []
        for key in angles:
            if key in ideal_angles:
                diff = abs(angles[key] - ideal_angles[key])
                weight = 1.5 if 'knee' in key or 'shoulder' in key else 1.0
                # Apply buffer: treat differences within buffer as zero
                if diff <= buffer:
                    diff = 0
                total_diff += diff * weight
                # Only suggest correction if difference is more than 2 degrees beyond buffer
                if diff > 2:
                    temp_corrections.append(f"Adjust your {key.replace('_', ' ')} (off by {diff:.1f}°)")
        avg_diff = total_diff / len(ideal_angles)
        if avg_diff < min_diff and avg_diff < 25:
            min_diff = avg_diff
            matched_pose = pose_name
            corrections = temp_corrections

    accuracy = max(0, 100 - min_diff)
    return {
        "pose_name": matched_pose if matched_pose else "Unknown Pose",
        "pose_hindi": pose_to_hindi.get(matched_pose, "N/A") if matched_pose else "N/A",
        "accuracy": round(accuracy, 2),
        "corrections": corrections,
        "annotated_image": annotated_path
    }
