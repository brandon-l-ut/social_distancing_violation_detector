import cv2 

img = cv2.imread("ipm/raw/IMG_5610.jpg")
img = cv2.resize(img, (540, 720))
cv2.imwrite("ipm/resized/test_3.jpg", img)