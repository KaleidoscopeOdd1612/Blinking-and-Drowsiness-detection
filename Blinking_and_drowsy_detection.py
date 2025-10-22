import cv2
import dlib
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
import threading
import numpy as np
import os
import json

# Check current working directory
print(f"Current working directory: {os.getcwd()}")

cap = cv2.VideoCapture("video/testvideo_3.mp4") # "video/blinking v1.mp4"
print("Camera working" if cap.isOpened() else "Camera NOT working") # check if camera working or not

fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_limit = math.floor((fps/1000)*100 +0.9)
print(f'frame_limit : {frame_limit}')
print(f'total_frames : {total_frames}')
print(f'window_size : ({width},{height})')

# Define codec and create VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('video.mp4', fourcc, fps, (width, height))

EAR_list = deque(maxlen=int(fps*3))
frames_check = deque(maxlen=1)
frames_count = 0
frames = deque(maxlen=int(fps*3))
EAR_list_full = []
frames_full = []
EAR_raw = deque(maxlen=(frame_limit+1))
edge = False
edge_drowsy = False
frame_counter = 0
blink = 0
drowsy_blink = 0
threshold = []
abs_threshold = []
threshold_positive = 100
threshold_negative = -100
total_blinks = []
total_drowsy_blinks = []

hog_face_detector = dlib.get_frontal_face_detector()

dlib_facelandmark = dlib.shape_predictor("pre-trained_models/shape_predictor_68_face_landmarks.dat/shape_predictor_68_face_landmarks.dat")

def landmarks_display(L_points, R_points, face_landmarks, frame):
    for n in L_points + R_points:
        x = face_landmarks.part(n).x
        y = face_landmarks.part(n).y
        cv2.circle(frame, (x, y), 1, (0, 255, 255), 1)

def blink_detection(points, face_landmarks):
    A = math.dist([face_landmarks.part(points[1]).x , face_landmarks.part(points[1]).y],[face_landmarks.part(points[5]).x , face_landmarks.part(points[5]).y])
    B = math.dist([face_landmarks.part(points[2]).x , face_landmarks.part(points[2]).y],[face_landmarks.part(points[4]).x , face_landmarks.part(points[4]).y])
    C = math.dist([face_landmarks.part(points[0]).x , face_landmarks.part(points[0]).y],[face_landmarks.part(points[3]).x , face_landmarks.part(points[3]).y])
    EAR = (A+B) / (C*2)
    return EAR

def append_value(value):
    global frames_count, blink

    EAR_list.append(value)
    EAR_list_full.append(value)
    frames.append(frames_count)
    frames_full.append(frames_count)
    total_blinks.append(blink)
    total_drowsy_blinks.append(drowsy_blink)

def blink_detector():
    global edge,edge_drowsy, frame_counter, blink, drowsy_blink, threshold_positive, threshold_negative

    if len(EAR_list) > 1 and len(EAR_list) < fps:
        threshold.append(EAR_list[-2] - EAR_list[-1])
        abs_threshold.append(abs(EAR_list[-2] - EAR_list[-1]))
    elif len(EAR_list) >= fps:
        if max(threshold) < 0.05:
            threshold_positive = max(abs_threshold) + 0.05
        else:
            threshold_positive = max(abs_threshold) + 0.01
        if min(threshold) > -0.05:
            threshold_negative = min(threshold) - 0.04
        else:
            threshold_negative = min(threshold) + 0.01

    if frame_counter > 0:
        frame_counter -= 1

    if len(EAR_raw) > frame_limit:
        filter = EAR_raw[0] - EAR_raw[-1]
        
        if filter > threshold_positive:
            edge = True
            edge_drowsy = True
            frame_counter = int(frame_limit*6)
        
        if filter < threshold_negative and edge:
            blink += 1
            edge = False
            edge_drowsy = False
            frame_counter = 0

        if edge_drowsy and frame_counter == 0:
            drowsy_blink += 1
            edge = False
            edge_drowsy = False

def blink_count(frame):
    # Display blink count on frame
    cv2.putText(frame, f'Blinks: {blink}', 
            (10, 120),  # Position (x, y)
            cv2.FONT_HERSHEY_SIMPLEX,  # Font
            4,  # Font scale
            (255, 255, 0),  # Color (BGR) - Green
            4,  # Thickness
            cv2.LINE_AA)  # Line type
    
    cv2.putText(frame, f'Drowsy: {drowsy_blink}', 
            (10, 2040),  # Position (x, y)
            cv2.FONT_HERSHEY_SIMPLEX,  # Font
            4,  # Font scale
            (255, 255, 0),  # Color (BGR) - Green
            4,  # Thickness
            cv2.LINE_AA)  # Line type

L_points = [i for i in range(36, 42)]
R_points = [i for i in range(42, 48)]

def camera_loop():
    global frames_count # because it is not a list

    while True:
        _, frame = cap.read()

        key = cv2.waitKey(1)
        if key == 32:
            break

        if len(frames_check) > 0 and np.array_equal(frame, frames_check[-1]):
            continue
        
        frames_check.append(frame.copy())  # Store copy of frame

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = hog_face_detector(gray)

        for face in faces:
            frames_count += 1
            face_landmarks = dlib_facelandmark(gray, face)

            landmarks_display(L_points, R_points, face_landmarks, frame)
            L_EAR = blink_detection(L_points, face_landmarks)
            EAR_raw.append(L_EAR)
            append_value(EAR_raw[-1])
            blink_detector()

        cv2.namedWindow("Face Landmarks", cv2.WINDOW_NORMAL)

        blink_count(frame)
        out.write(frame)
        cv2.imshow("Face Landmarks", frame)

        key = cv2.waitKey(1)
        if key == 32:
            break

loop = threading.Thread(target=camera_loop, daemon=True)
loop.start()

fig, ax = plt.subplots()
ax.set_xlabel('Frames')  
ax.set_ylabel('EAR Value')
ax.set_xlim(0, 10)  # Initial x-axis range
ax.set_ylim(0, 0.5)  # Initial y-axis range (adjust based on typical EAR values)

line, = ax.plot([], [], 'b-')
blink_text = ax.text(0.1, 0.05, '', ha='center', va='bottom', transform=ax.transAxes, fontsize=12)
drowsy_blink_text = ax.text(0.1, 0.9, '', ha='center', va='bottom', transform=ax.transAxes, fontsize=12)

def update_plot(frame): 
    if len(EAR_list) > 0:  # Only update if we have data
        line.set_data(frames, EAR_list)
        ax.set_xlim(min(frames), max(frames))

        # Update the text content instead of creating new text
        blink_text.set_text(f'Blinks: {blink}')
        drowsy_blink_text.set_text(f'Drowsy: {drowsy_blink}')
    
    return line,

# Use frames=None for infinite animation with repeat=False
# interval in milliseconds between updates
ani = FuncAnimation(fig, update_plot, interval=int((1000)/((5/6)*fps)), blit=False, repeat=False)
plt.show()

loop.join()

data = {
    "frames_full": frames_full,
    "EAR_list_full": EAR_list_full
}

with open("my_list.json", "w") as file:
    json.dump(data, file)

cap.release()
out.release
cv2.destroyAllWindows()

print(f"Total blinks : {blink}")
print(f"threshold positive : {threshold_positive}")
print(f"threshold negative : {threshold_negative}")

# Final plot
fig, ax = plt.subplots(figsize=(12, 6))
ax.set_ylim(0, 0.5)  # Initial y-axis range (adjust based on typical EAR values)
ax.plot(frames_full, EAR_list_full, 'b-', linewidth=1)
ax.set_xlabel('Frames')
ax.set_ylabel('EAR Value')
ax.set_title('Complete EAR Timeline')
ax.grid(True)
plt.show()

fig, ax = plt.subplots()
ax.set_xlabel('Frames')  
ax.set_ylabel('EAR Value')
ax.set_xlim(0, 10)  # Initial x-axis range
ax.set_ylim(0, 0.5)  # Initial y-axis range (adjust based on typical EAR values)

line, = ax.plot([], [], 'b-')
blink_text = ax.text(0.1, 0.05, '', ha='center', va='bottom', transform=ax.transAxes, fontsize=12)
drowsy_blink_text = ax.text(0.1, 0.9, '', ha='center', va='bottom', transform=ax.transAxes, fontsize=12)

def update_plot(frame_idx): 
    if frame_idx < len(EAR_list_full):
        # Calculate start index (show last 100 frames)
        start_idx = max(0, frame_idx - 99)
        
        # Slice to get last 100 frames
        line.set_data(frames_full[start_idx:frame_idx+1], 
                     EAR_list_full[start_idx:frame_idx+1])
        
        # Update x-axis to show moving window
        ax.set_xlim(frames_full[start_idx], frames_full[frame_idx])
        # Update the text content instead of creating new text
        blink_text.set_text(f'Blinks: {total_blinks[frame_idx]}')
        drowsy_blink_text.set_text(f'Drowsy: {total_drowsy_blinks[frame_idx]}')
    
    return line,

# Use frames=None for infinite animation with repeat=False
# interval in milliseconds between updates
ani = FuncAnimation(fig, update_plot, frames=len(frames_full), interval=int((1000)/(fps)), blit=False, repeat=False)
# Save as video
ani.save('output.mp4', writer='ffmpeg', fps=fps)

plt.show()