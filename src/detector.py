#Load YOLO → run detection on a frame → return detection results.
import cv2
from ultralytics import YOLO

class ObjectDetector:

    def __init__(self, model_path, confidence_threshold=0.5):

        # load the model
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold


    def detect(self, frame):
        results = self.model(frame, verbose=False)  # Run detection on the frame
        return results[0]  # Return the first result (assuming single frame input)  


    def annotate(self, frame, result):
        boxes = result.boxes.xyxy
        scores = result.boxes.conf
        classes = result.boxes.cls
        names = result.names

        for box, score, class_id in zip(boxes, scores, classes):

            if float(score) < self.confidence_threshold:
                continue

            x1, y1, x2, y2 = map(int, box)
            class_name = names[int(class_id)]

            label = f"{class_name} {float(score):.2f}"

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        return frame