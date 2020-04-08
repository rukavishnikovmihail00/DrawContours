import numpy as np
import cv2

fullbody_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')
cap = cv2.VideoCapture(0)
while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fullbody = fullbody_cascade.detectMultiScale(gray, 1.3, 3)
    for (x, y, w, h) in fullbody:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        crop = img[y:y + h, x:x + w]
        ret, thresh = cv2.threshold(gray, 127, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(crop, contours, -1, (0, 255, 0), 3)


    cv2.imshow('img', img)
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()


