import math
import queue
import threading
import PySimpleGUI as sg
import cv2
import numpy as np

from start_cam import start_cam


def get_border_points(height, width, rho, sin_t, cos_t):
    border_points = []

    if sin_t != 0:
        item = width * (-cos_t / sin_t) + rho / sin_t
        if not math.isnan(item) and height > item >= 0:
            border_points.append((width, item))

        item = rho / sin_t
        if not math.isnan(item) and height > item >= 0:
            border_points.append((0, item))

    if cos_t != 0:
        item = height * (-sin_t / cos_t) + rho / cos_t
        if not math.isnan(item) and width > item >= 0:
            border_points.append((item, height))

        item = rho / cos_t
        if not math.isnan(item) and width > item >= 0:
            border_points.append((item, 0))

    return [(int(i[0]), int(i[1])) for i in border_points]


def get_mask(m_in):
    contours, _ = cv2.findContours(m_in, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    mask = np.zeros(m_in.shape, np.uint8)
    for i in sorted(contours, key=lambda x: cv2.arcLength(x, True), reverse=True)[:3]:
        x, y, w, h = cv2.boundingRect(i)
        if abs(1 - (w / h)) < 0.2 and w * h > m_in.shape[0] * m_in.shape[1] / 4:
            cv2.rectangle(mask, (x, y), (x + w, y + h), color=(255, 255, 255), thickness=-1)
            break

    return mask


def unite_nearby_lines(lines):
    far_lines = []
    for rho, theta in lines[:, 0]:
        for i, line in enumerate(far_lines):
            if abs(line[0] / line[2] - rho) < 15 and abs(line[1] / line[2] - theta) < 10 * np.pi / 180:
                far_lines[i] = (line[0] + rho, line[1] + theta, line[2] + 1)
                break
        else:
            far_lines.append((rho, theta, 1))
    return [(i[0] / i[2], i[1] / i[2]) for i in far_lines]


def intersection_point(rho1, theta1, rho2, theta2):
    denominator = np.sin(theta1 - theta2)

    if denominator != 0:
        x = (rho2 * np.sin(theta1) - rho1 * np.sin(theta2)) / denominator
        y = (rho1 * np.cos(theta2) - rho2 * np.cos(theta1)) / denominator
        return x, y
    else:
        return None  # Линии параллельны, точки пересечения нет


def draw_lines(mat, lines):
    for rho, theta in lines:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * a)
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * a)
        cv2.line(mat, (x1, y1), (x2, y2), (255, 255, 255), 2)


def get_cell_state(cell):
    contours, _ = cv2.findContours(cell, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return 0
    m = max(contours, key=cv2.contourArea)
    m_area = cv2.contourArea(m)
    hull_area = cv2.contourArea(cv2.convexHull(m))
    if hull_area == 0 or m_area < cell.shape[0] * cell.shape[1] * 0.1:
        return 0
    hull_diff = cv2.contourArea(m) / hull_area
    if hull_diff > 0.75:
        return 1
    elif hull_diff > 0.25:
        return -1
    else:
        return 0


def get_intersections(edges, sliders):
    vertical_lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=sliders[2],
                                    min_theta=-10 * np.pi / 180,
                                    max_theta=10 * np.pi / 180)
    horizontal_lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=sliders[2],
                                      min_theta=80 * np.pi / 180,
                                      max_theta=100 * np.pi / 180)
    if vertical_lines is None or horizontal_lines is None:
        return None

    vertical_lines = sorted(unite_nearby_lines(vertical_lines), key=lambda x: x[0])
    horizontal_lines = sorted(unite_nearby_lines(horizontal_lines), key=lambda x: x[0])

    if len(vertical_lines) != 2 or len(horizontal_lines) != 2:
        return None

    intersections = [
        intersection_point(horizontal_lines[0][0], horizontal_lines[0][1], vertical_lines[0][0], vertical_lines[0][1]),
        intersection_point(horizontal_lines[0][0], horizontal_lines[0][1], vertical_lines[1][0], vertical_lines[1][1]),
        intersection_point(horizontal_lines[1][0], horizontal_lines[1][1], vertical_lines[0][0], vertical_lines[0][1]),
        intersection_point(horizontal_lines[1][0], horizontal_lines[1][1], vertical_lines[1][0], vertical_lines[1][1]),
    ]
    return [(int(p[0]), int(p[1])) for p in intersections]


def erase_game_borders(edges, intersections, cell_size):
    game_borders = [
        [(intersections[0][0] - cell_size[0], intersections[0][1]),
         (intersections[1][0] + cell_size[0], intersections[1][1])],
        [(intersections[1][0], intersections[1][1] - cell_size[1]),
         (intersections[3][0], intersections[3][1] + cell_size[1])],
        [(intersections[3][0] + cell_size[0], intersections[3][1]),
         (intersections[2][0] - cell_size[0], intersections[2][1])],
        [(intersections[2][0], intersections[2][1] + cell_size[1]),
         (intersections[0][0], intersections[0][1] - cell_size[1])]
    ]

    for p1, p2 in game_borders:
        cv2.line(edges, p1, p2, color=(0, 0, 0), thickness=6)

    return edges


def get_result_image(edges, intersections, cell_size):
    line_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    game_cells = [
        (intersections[0][0] - cell_size[0], intersections[0][1] - cell_size[1]),
        (intersections[0][0], intersections[0][1] - cell_size[1]),
        (intersections[1][0], intersections[1][1] - cell_size[1]),

        (intersections[0][0] - cell_size[0], intersections[0][1]),
        (intersections[0][0], intersections[0][1]),
        (intersections[1][0], intersections[1][1]),

        (intersections[2][0] - cell_size[0], intersections[2][1]),
        (intersections[2][0], intersections[2][1]),
        (intersections[3][0], intersections[3][1]),
    ]

    for i, (x, y) in enumerate(game_cells):
        cv2.rectangle(line_image, (x, y), (x + cell_size[0], y + cell_size[1]), color=(255, 255, 0), thickness=1)

    game_state = [0] * 9
    for i, cell in enumerate(game_cells):
        game_state[i] = get_cell_state(edges[cell[1]:cell[1] + cell_size[1], cell[0]: cell[0] + cell_size[0]])
        cv2.putText(line_image, f'({round(game_state[i], 2)})', (int(game_cells[i][0]), int(game_cells[i][1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0),
                    2)

    return line_image


def distance_between_points(x1, y1, x2, y2):
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance


def find_lines(frame, sliders):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, sliders[0], sliders[1], apertureSize=3)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    mask = get_mask(edges)
    bitwise = cv2.bitwise_and(edges, edges, mask=mask)

    intersections = get_intersections(bitwise, sliders)
    if intersections is not None:
        cell_size = (intersections[3][0] - intersections[0][0], intersections[3][1] - intersections[0][1])
        bitwise = erase_game_borders(bitwise, intersections, cell_size)
        return get_result_image(bitwise, intersections, cell_size)

    return bitwise


def cross_zero_layout():
    camera_queue = queue.Queue(maxsize=3)
    cam_thread = threading.Thread(target=start_cam,
                                  args=(camera_queue, lambda: sg.popup_error("Failed to grab frame.")), daemon=True)
    sliders_values = [0] * 4

    layout = [
        [sg.Image(filename="", key="-ORIGINAL-"), sg.Image(filename="", key="-LINES-")],
        [sg.Button("Start Camera"), sg.Button("Exit")],
        [sg.Slider(range=(0, 300), default_value=sliders_values[i], orientation="v", size=(10, 20),
                   key=f"-SLIDER{i}-",
                   enable_events=True)
         for i in range(len(sliders_values))]
    ]
    window = sg.Window("Thresh definition", layout)

    while True:
        event, values = window.read(timeout=20)

        if event == sg.WINDOW_CLOSED or event == "Exit":
            break

        elif event == "Start Camera" and not cam_thread.is_alive():
            cam_thread.start()

        for i in range(len(sliders_values)):
            slider_key = f"-SLIDER{i}-"
            if event == slider_key:
                sliders_values[i] = int(values[slider_key])

        try:
            frame = camera_queue.get_nowait()
            img_bytes = cv2.imencode(".png", frame)[1].tobytes()
            window["-ORIGINAL-"].update(data=img_bytes)

            lines = find_lines(frame, sliders_values)
            img_bytes = cv2.imencode(".png", lines)[1].tobytes()
            window["-LINES-"].update(data=img_bytes)

        except queue.Empty:
            pass

    window.close()
