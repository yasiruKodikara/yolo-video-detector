from src.detector import ObjectDetector

from src.video import VideoProcessor

def main():
    model_path = "models/yolo26n.pt"  # Path to the YOLOv8 model
    input_path = "input/test.mp4"  # Path to the input video
    output_path = "output/output_video.mp4"  # Path to save the output video

    detector = ObjectDetector(model_path)  # Initialize the object detector

    video_processor = VideoProcessor(detector)  # Initialize the video processor with the detector

    video_processor.process(input_path, output_path)  # Process the video

if __name__ == "__main__":
    main()


