import cv2
import numpy as np
lower_bound = np.array([100, 150, 0])  
upper_bound = np.array([140, 255, 255]) 
canvas = None

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    if canvas is None:
        canvas = np.zeros_like(frame)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    center = None

    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > 500:
            ((x, y), radius) = cv2.minEnclosingCircle(largest)
            M = cv2.moments(largest)
            if M["m00"] != 0:
                cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                center = (cx, cy)
                cv2.circle(frame, center, 5, (0, 255, 0), -1)
                cv2.circle(canvas, center, 5, (255, 0, 0), -1)
    output = cv2.add(frame, canvas)
    cv2.imshow("Air Draw", output)
    cv2.imshow("Mask", mask)

    key = cv2.waitKey(1)
    if key == ord('c'):
        canvas = None 
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
