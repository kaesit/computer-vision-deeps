import cv2


def open_image (image_path:str, window_name:str="Image Viewer", width:int=800, height:int=600):
    
    img = cv2.imread(image_path)
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

