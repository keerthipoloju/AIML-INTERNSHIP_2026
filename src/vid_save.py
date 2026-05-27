import cv2


cap = cv2.VideoCapture(r"C:\Users\keert\OneDrive\Documents\computer vision\1.mp4")

print("FPS:", cap.get(cv2.CAP_PROP_FPS))

import os

video_path = r"C:\Users\keert\OneDrive\Documents\computer vision\1.mp4"

if not os.path.exists(video_path):
    print("Error: File not found at", video_path)
    exit()

cap = cv2.VideoCapture(r"C:\Users\keert\OneDrive\Documents\computer vision\1.mp4")

if not cap.isOpened():
    print("Error: Could not open video file")
    exit()

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')   # Use 'mp4v' for mp4 output
out = cv2.VideoWriter("output.avi", fourcc, fps, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        print("Finished reading video or failed to capture frame.")
        break

    cv2.imshow("Video Stream", frame)
    out.write(frame)  # Save frame to output file

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
