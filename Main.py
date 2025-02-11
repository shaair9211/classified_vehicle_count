from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *
import numpy as np
import pandas as pd

cap = cv2.VideoCapture(r"C:\Users\shaai\PycharmProjects\pythonProject\Videos\UniversityRoad.mp4") # For Video

model = YOLO(r'C:\Users\shaai\PycharmProjects\pythonProject\YOLO-Weights\farman.pt')

classNames = ["car", "bike", "bus", "truck", "rickshaw"]
tracker = Sort(max_age=10, min_hits=3, iou_threshold=0.25)

limits = [600, 100, 600, 550]
totalCounts = []

cars = []
bikes = []
buses = []
trucks = []
rickshaws = []

frame_counts = []
frame_number = 0

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

while True:
    success, img = cap.read()
    if not success:
        print("Video ended.", flush=True)
        break
    #imgRegion = cv2.bitwise_and(img,mask)

    results = model(img, stream=True, conf=0.5)

    detections = np.empty((0, 5))
    classes = []

    current_frame_counts = {
        "Frame": frame_number,
        "Car": 0,
        "Bike": 0,
        "Truck": 0,
        "Bus": 0,
        "Rickshaw": 0
    }

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if currentClass == "car" or currentClass == "truck" or currentClass == "bus" or currentClass == "bike" or currentClass == "rickshaw" and conf > 0.7:
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))
                classes.append(currentClass)


    resultsTracker = tracker.update(detections, classes)
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)

    index = 0
    for result in resultsTracker:
        x1, y1, x2, y2, id, class_name = result
        # print(result)
        x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 0))
        cvzone.putTextRect(img, f'{int(id)}', (max(0, x1), max(35, y1)), scale=2, thickness=3, offset=10)

        cy, cx = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cy, cx), 5, (255, 0, 255), cv2.FILLED)

        if limits[1] < cx < limits[3] and limits[0] - 15 < cy < limits[2] + 15:
            if class_name == "car" and id not in cars:
                cars.append(id)
                current_frame_counts["Car"] += 1
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)
            elif class_name == "bike" and id not in bikes:
                bikes.append(id)
                current_frame_counts["Bike"] += 1
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)
            elif class_name == "bus" and id not in buses:
                buses.append(id)
                current_frame_counts["Bus"] += 1
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)
            elif class_name == "truck" and id not in trucks:
                trucks.append(id)
                current_frame_counts["Truck"] += 1
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)
            elif class_name == "rickshaw" and id not in rickshaws:
                rickshaws.append(id)
                current_frame_counts["Rickshaw"] += 1
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)
            index += 1

    frame_counts.append(current_frame_counts)

    cvzone.putTextRect(img, f'car: {len(cars)}', (50, 50))
    cvzone.putTextRect(img, f'bike: {len(bikes)}', (50, 100))
    cvzone.putTextRect(img, f'truck: {len(trucks)}', (50, 150))
    cvzone.putTextRect(img, f'bus: {len(buses)}', (50, 200))
    cvzone.putTextRect(img, f'rickshaw: {len(rickshaws)}', (50, 250))

    cv2.imshow("Frame", img)
    out.write(img)
    if cv2.waitKey(1) == ord("q"):
        break

print('aaaaaaaaa')
cap.release()
out.release()

df = pd.DataFrame(frame_counts)
df.to_excel("vehicle_counts.xlsx", index=False)


