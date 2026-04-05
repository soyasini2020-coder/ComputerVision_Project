import argparse
import importlib
import json
from pathlib import Path
from typing import Dict, List
import cv2


FOOD_CLASS_MAP = {
    "apple": "apple",
    "banana": "banana",
    "orange": "orange",
    "sandwich": "sandwich",
    "pizza": "pizza",
    "hot dog": "hot dog",
    "donut": "donut",
    "cake": "cake",
    "broccoli": "broccoli",
    "carrot": "carrot",
}

STRICT_KNOWN_FOODS = {
    "apple",
    "banana",
    "orange",
    "pizza",
    "hot dog",
    "donut",
    "cake",
    "broccoli",
    "carrot",
}

DEFAULT_IMAGE_CANDIDATES = [
    "input.jpg",
    "input.jpeg",
    "input.png",
]


def load_calorie_db(db_path: Path) -> Dict:
    with db_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def detect_food_yolo(
    image_path: Path, min_confidence: float = 0.25
) -> tuple[List[Dict], List[object]]:
    YOLO = importlib.import_module("ultralytics").YOLO

    model = YOLO("yolov8n.pt")
    results = model.predict(source=str(image_path), conf=min_confidence, verbose=False)

    detections: List[Dict] = []
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls.item())
            class_name = result.names[cls_id]
            confidence = float(box.conf.item())

            if class_name in FOOD_CLASS_MAP:
                mapped_label = FOOD_CLASS_MAP[class_name]
                detections.append(
                    {
                        "label": mapped_label,
                        "confidence": round(confidence, 3),
                    }
                )

    return detections, results


def save_detection_visualization(image_path: Path, results: List[object], output_path: Path) -> None:
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Unable to read image for visualization: {image_path}")

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls.item())
            class_name = result.names[cls_id]
            if class_name not in FOOD_CLASS_MAP:
                continue

            conf = float(box.conf.item())
            label = FOOD_CLASS_MAP[class_name]
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                img,
                f"{label} {conf:.2f}",
                (x1, max(15, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), img)


def estimate_calories(detections: List[Dict], calorie_db: Dict, portion: str) -> Dict:
    items = []
    total = 0.0

    for det in detections:
        label = det["label"]
        if label not in calorie_db:
            continue

        kcal_per_100g = calorie_db[label]["kcal_per_100g"]
        grams = calorie_db[label]["grams"][portion]
        calories = (kcal_per_100g * grams) / 100.0

        items.append(
            {
                "label": label,
                "confidence": det["confidence"],
                "portion": portion,
                "grams": grams,
                "kcal_per_100g": kcal_per_100g,
                "estimated_calories": round(calories, 1),
            }
        )
        total += calories

    return {"items": items, "total_calories": round(total, 1)}


def resolve_image_path(image_arg: str | None) -> Path:
    if image_arg:
        image_path = Path(image_arg)
        if image_path.exists():
            return image_path
        raise FileNotFoundError(f"Image not found: {image_path}")

    for candidate in DEFAULT_IMAGE_CANDIDATES:
        candidate_path = Path(candidate)
        if candidate_path.exists():
            return candidate_path

    expected = ", ".join(DEFAULT_IMAGE_CANDIDATES)
    raise FileNotFoundError(
        "No image provided. Use --image <file> or add one of these files in the project root: "
        f"{expected}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO food detection and calorie estimator")
    parser.add_argument(
        "--image",
        required=False,
        help="Path to input image (optional if input.jpg/input.png exists)",
    )
    parser.add_argument(
        "--portion",
        default="medium",
        choices=["small", "medium", "large"],
        help="Portion size assumption",
    )
    parser.add_argument(
        "--db",
        default=str(Path(__file__).with_name("calorie_db.json")),
        help="Path to calorie database JSON",
    )
    parser.add_argument(
        "--output-image",
        default="outputs/detected_food.jpg",
        help="Path to save visualization with bounding boxes",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.25,
        help="Minimum YOLO confidence threshold (0.0 to 1.0)",
    )
    args = parser.parse_args()

    if not 0.0 <= args.min_confidence <= 1.0:
        raise ValueError("--min-confidence must be between 0.0 and 1.0")

    image_path = resolve_image_path(args.image)

    calorie_db = load_calorie_db(Path(args.db))

    detections, raw_results = detect_food_yolo(
        image_path,
        min_confidence=args.min_confidence,
    )
    save_detection_visualization(image_path, raw_results, Path(args.output_image))

    report = estimate_calories(detections, calorie_db, args.portion)

    if not report["items"]:
        print("Food not detected or not in supported class list.")
        print(f"Detection image saved to: {args.output_image}")
        print("Total estimated calories: 0.0")
        return

    print("Detected items and estimated calories:")
    for item in report["items"]:
        print(
            f"- {item['label']} (conf={item['confidence']}) | "
            f"portion={item['portion']} | grams={item['grams']} | "
            f"kcal={item['estimated_calories']}"
        )

    print(f"Detection image saved to: {args.output_image}")
    print(f"Total estimated calories: {report['total_calories']}")


if __name__ == "__main__":
    main()
