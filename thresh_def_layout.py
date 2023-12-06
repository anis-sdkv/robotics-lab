import PySimpleGUI as sg
import numpy as np
import cv2
import queue
import threading
from start_cam import start_cam


def apply_sliders(frame, sliders_values):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    low = np.array((sliders_values[0], sliders_values[2], sliders_values[4]))
    high = np.array((sliders_values[1], sliders_values[3], sliders_values[5]))
    mask = cv2.inRange(hsv, low, high)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, max_contour, -1, (0, 0, 255), 2)

        x, y, w, h = cv2.boundingRect(max_contour)
        cv2.rectangle(mask, (x, y), (x + w, y + h), (255, 255, 0), 2)
        cv2.putText(mask, f'({x}, {y})', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        m = cv2.moments(max_contour)
        if m["m00"] != 0:
            c_x = int(m["m10"] / m["m00"])
            c_y = int(m["m01"] / m["m00"])
            cv2.circle(mask, (c_x, c_y), 5, (0, 0, 255), -1)

    return mask


def thresh_def_layout():
    camera_queue = queue.Queue(maxsize=10)
    cam_thread = threading.Thread(target=start_cam,
                                  args=(camera_queue, lambda: sg.popup_error("Failed to grab frame.")), daemon=True)
    sliders_values = [i % 2 * 256 for i in range(6)]

    layout = [
        [sg.Image(filename="", key="-ORIGINAL-"), sg.Image(filename="", key="-MASK-")],
        [sg.Button("Start Camera"), sg.Button("Exit")],
        [sg.Slider(range=(0, 255), default_value=sliders_values[i], orientation="v", size=(10, 20), key=f"-SLIDER{i}-",
                   enable_events=True)
         for i in range(6)],
    ]
    window = sg.Window("Thresh definition", layout)

    while True:
        event, values = window.read(timeout=20)

        if event == sg.WINDOW_CLOSED or event == "Exit":
            break

        elif event == "Start Camera" and not cam_thread.is_alive():
            cam_thread.start()

        for i in range(6):
            slider_key = f"-SLIDER{i}-"
            if event == slider_key:
                sliders_values[i] = int(values[slider_key])

        try:
            frame = camera_queue.get_nowait()
            mask = apply_sliders(frame, sliders_values)
            img_bytes = cv2.imencode(".png", frame)[1].tobytes()
            window["-ORIGINAL-"].update(data=img_bytes)
            img_bytes = cv2.imencode(".png", mask)[1].tobytes()
            window["-MASK-"].update(data=img_bytes)

        except queue.Empty:
            pass

    window.close()
