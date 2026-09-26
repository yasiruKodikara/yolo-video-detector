import cv2

class VideoProcessor:

    def __init__(self, detector):
        self.detector = detector

    def process(self, input_path, output_path):
        # Open input video
        cap = cv2.VideoCapture(input_path)

        if not cap.isOpened():
            raise RuntimeError("Could not open input video")

        #additional information about the video
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0

        # Create a window to display the video
        cv2.namedWindow("YOLO Video Detector", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("YOLO Video Detector", 860, 740)

        print("Width:", width)
        print("Height:", height)
        print("FPS:", fps)
        print("Frame count:", frame_count)
        print("Duration:", duration, "seconds")

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        writer = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            (width, height)
        )

        frame_number = 0

        try:
            while True:
                ret, frame = cap.read() # Read a frame from the video

                if not ret:
                    break  # Break the loop if no frame is read (end of video)


                frame_number+=1

                print(
                    f"\rProcessing: {frame_number}/{frame_count} frames",
                    end=""
                )

                
                # Run detection on the frame
                results = self.detector.detect(frame)

                # Extract detections
                result = results[0]
        
                boxes = result.boxes.xyxy
                scores = result.boxes.conf
                classes = result.boxes.cls
                names = result.names
        
                print("\nDetections:")
                print("Boxes:", boxes)
                print("Scores:", scores)
                

                # Annotate the frame with detection results

                # YOLO provides a built-in method to plot the results on the frame
                annotated_frame = results.plot()

                # Alternatively, you can use the custom annotate method
                # annotated_frame = self.detector.annotate(frame, results)

                # Write the annotated frame to the output video
                writer.write(annotated_frame)

                cv2.imshow("YOLO Video Detector", annotated_frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            cap.release()
            writer.release()
            cv2.destroyAllWindows()

        

    


