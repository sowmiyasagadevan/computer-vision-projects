import cv2
import numpy as np


cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 1)
    edges = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            obj_cor = len(approx)

            x, y, w, h = cv2.boundingRect(approx)

            if obj_cor == 3:
                shape_type = "Triangle"
            elif obj_cor == 4:
                asp_ratio = w / float(h)
                shape_type = "Square" if 0.95 < asp_ratio < 1.05 else "Rectangle"
            elif obj_cor > 4:
                shape_type = "Circle"
            else:
                shape_type = "Unknown"

            cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)
            cv2.putText(frame, shape_type, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.imshow("Shape Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
