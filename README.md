# Food Object Detection and Calorie Estimation (Simple Starter)

This repository is a minimal implementation for the project proposal.

## What this starter does
- Detects food objects in an image using YOLOv8
- Estimates calories using a small food calorie database
- Supports manual portion size assumptions: `small`, `medium`, `large`
- Saves a visualization image with bounding boxes

## Why this matches professor feedback
- Uses Kaggle dataset mainly for training/validation/internal test
- Includes plan to collect new custom images for external real-world testing
- Includes portion-size handling in calorie estimation (manual baseline)

## Setup
```bash
pip install -r requirements.txt
```

## Run (YOLO mode)
```bash
python src/main.py --image your_image.jpg --portion medium --output-image outputs/detected_food.jpg
```

## Easiest way (no filename typing)
1. Put your photo in the project root and name it `input.jpg` or `input.png`.
2. Run:

```bash
python src/main.py --portion medium --output-image outputs/detected_food.jpg
```

The script will automatically use `input.jpg`, `input.jpeg`, or `input.png` if found.

If you use conda and packages are installed in a specific environment, run with the full interpreter path:
```bash
/opt/anaconda3/envs/ocv/bin/python src/main.py --image your_image.jpg --portion medium --output-image outputs/detected_food.jpg
```

Notes:
- YOLO mode uses `yolov8n.pt` pretrained model.
- Accuracy is limited until you fine-tune on food-specific dataset.

## Verified quick tests
```bash
curl -L "https://upload.wikimedia.org/wikipedia/commons/8/8a/Banana-Single.jpg" -o banana_real.jpg
/opt/anaconda3/envs/ocv/bin/python src/main.py --image banana_real.jpg --portion medium --output-image outputs/banana_detected.jpg
```

## Suggested Next Steps
1. Collect 80-120 custom phone images as external test set.
2. Compare performance on different lighting/angles.
3. Add improved portion-size estimation (area/depth/segmentation based).

## External Real-World Test Photos
- Put your own phone images in `data/external_test/`
- Run inference image-by-image with `src/main.py`

## External Test (Simple)
Test external images one by one with the main script:

```bash
python src/main.py --image data/external_test/Apple.png --portion medium --output-image outputs/apple_result.jpg
```
