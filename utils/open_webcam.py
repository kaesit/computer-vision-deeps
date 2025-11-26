import cv2

def open_webcam(web_cam_resource: int, window_name:str, width:int, height:int):
    cap = cv2.VideoCapture(web_cam_resource)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    cv2.namedWindow(window_name)
    cv2.resizeWindow(window_name, width, height)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        cv2.imshow(window_name, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break