import cv2 

img = cv2.imread("disnet/raw/IMG_5614.jpg")
img = cv2.resize(img, (540, 720))
cv2.imwrite("disnet/resized/test_2.jpg", img)