import cv2
import PySimpleGUI as sg


def resize_image(img):
    k = values["-RESIZE-"] / 100
    h, w = int(img.shape[0] * k), int(img.shape[1] * k)
    resized_image = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    return resized_image


def process_image(img):
    _, thresh = cv2.threshold(img, values["-THRESH-"], 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    width, height = img.shape[:2]
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > width * height // 6:
            cv2.drawContours(img, contour, -1, (255, 255, 255), 2)
            m = cv2.moments(contour)
            center_x = int(m['m10'] / m['m00'])
            center_y = int(m['m01'] / m['m00'])
            cv2.circle(img, (center_x, center_y), 3, (0, 0, 255), -1)
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return img, thresh


# Создаем простой интерфейс
layout = [
    [sg.Text("Выберите изображение:")],
    [sg.FileBrowse(key="-FILE-")],
    [sg.Image(key="-CONTOURS_IMG-"), sg.Image(key="-THRESH_IMG-")],
    [sg.Text("Thresh:")],
    [sg.Slider(range=(1, 255), orientation="h", size=(40, 15), default_value=127, key="-THRESH-")],
    [sg.Text("Resize:")],
    [sg.Slider(range=(1, 200), orientation="h", size=(40, 15), default_value=100, key="-RESIZE-")],
    [sg.Button("Обновить изображение"), sg.Button("Выход")]
]

window = sg.Window("Obj detection", layout, resizable=True)

while True:
    event, values = window.read()

    if event == sg.WINDOW_CLOSED or event == "Выход":
        break
    elif event == "Обновить изображение":
        file_path = values["-FILE-"]

        if file_path:
            try:
                image = resize_image(cv2.imread(file_path, 0))
                contours_res, thresh_res = process_image(image)

                window["-CONTOURS_IMG-"].update(data=cv2.imencode('.png', contours_res)[1].tobytes())
                window["-THRESH_IMG-"].update(data=cv2.imencode('.png', thresh_res)[1].tobytes())

            except Exception as e:
                sg.popup_error(f"Ошибка загрузки изображения: {e}")

window.close()
