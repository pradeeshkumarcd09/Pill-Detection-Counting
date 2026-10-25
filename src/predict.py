from pathlib import Path
import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(
        description="Detect and count pills in an image."
    )
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--conf", type=float, default=0.60)
    args = parser.parse_args()

    model_path = Path(__file__).resolve().parents[1] / "models" / "best.pt"
    model = YOLO(str(model_path))

    result = model.predict(
        source=args.image,
        conf=args.conf,
        imgsz=640,
        verbose=False
    )[0]

    print("Image:", args.image)
    print("Detected pill count:", len(result.boxes))
    print(f"Confidence threshold: {args.conf}")

    output_path = Path("pill_detection_output.jpg")
    result.save(filename=str(output_path))
    print("Annotated image saved to:", output_path.resolve())

if __name__ == "__main__":
    main()
