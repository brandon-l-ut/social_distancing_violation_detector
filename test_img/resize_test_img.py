import cv2 

img = cv2.imread("experiment/raw/IMG_5660.jpg")
img = cv2.resize(img, (540, 720))
cv2.imwrite("experiment/resized/test_24.jpg", img)