import cv2


def start_cam(cam_queue, on_err):
    cap = cv2.VideoCapture(1)
    while True:
        ret, frame = cap.read()
        if not ret:
            on_err()
            break

        cam_queue.put(cv2.resize(frame, (0, 0), fx=0.5, fy=0.5))

    cap.release()
