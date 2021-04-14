import cv2 

img = cv2.imread("raw/IMG_5587.jpg")
img = cv2.resize(img, (540, 720))
cv2.imwrite("resized/test_7.jpg", img)