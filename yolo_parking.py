import cv2
import json
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import os
import argparse


SLOT_IOU_THRESHOLD = 0.2           # IoU threshold to consider a slot occupied
CAR_CLASS_IDS = {2, 3, 5, 7, 67}  # car-related classes from COCO (car, bus, truck, etc.)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Parking lot occupancy + car parking time visualization"
    )
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to input video (e.g. parking/source/parking.mp4)",
    )
    parser.add_argument(
        "--slots",
        type=str,
        default="parking/parking-lot/parking_slots.json",
        required=False,
        help="Path to parking_slots.json (from ROI tool)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output annotated video (e.g. parking/output/parking_results.mp4)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8m.pt",
        help="YOLO model weights path (default: yolov8m.pt)",
    )
    
    return parser.parse_args()


def iou(boxA, boxB):
    """Compute IoU between two boxes: (x1,y1,x2,y2)."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    if interArea == 0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return interArea / float(boxAArea + boxBArea - interArea)


def main():
    args = parse_args()

    VIDEO_PATH = args.video
    OUTPUT_VIDEO_PATH = args.output
    PARKING_SLOTS_JSON = args.slots
    MODEL_PATH = args.model
    

    # Make sure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_VIDEO_PATH), exist_ok=True)

    # -----------------------
    # 1. Load YOLO model
    # -----------------------
    print(f"[INFO] Loading YOLO model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    # -----------------------
    # 2. Load parking slot config
    # -----------------------
    print(f"[INFO] Loading parking slots: {PARKING_SLOTS_JSON}")
    with open(PARKING_SLOTS_JSON, "r") as f:
        cfg = json.load(f)

    slots = cfg["slots"]

    slot_rects = {}  # slot_id -> (x1, y1, x2, y2)
    slot_occupied_frames = defaultdict(int)

    for s in slots:
        slot_id = s["id"]
        slot_rects[slot_id] = (s["x1"], s["y1"], s["x2"], s["y2"])

    # For per-car parking time (only counted while inside a slot)
    car_parked_frames = defaultdict(int)

    # -----------------------
    # 3. Video capture + writer
    # -----------------------
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Source FPS: {fps}, Size: {width}x{height}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))

    frame_idx = 0
    total_slots = len(slot_rects)

    # Info panel geometry (top-right)
    panel_w = 260
    panel_h = 160
    panel_x2 = width - 10
    panel_x1 = panel_x2 - panel_w
    panel_y1 = 10
    panel_y2 = panel_y1 + panel_h

    # -----------------------
    # 4. Main loop
    # -----------------------
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # Run YOLO tracking (persist=True keeps IDs consistent across frames)
        results = model.track(frame, persist=True, verbose=False)[0]

        car_boxes = []
        car_ids = []

        # Collect car boxes + IDs (no drawing yet)
        if results.boxes is not None:
            for box in results.boxes:
                cls_id = int(box.cls[0])
                if cls_id not in CAR_CLASS_IDS:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                track_id = box.id
                if track_id is None:
                    continue

                track_id = int(track_id)
                car_boxes.append((x1, y1, x2, y2))
                car_ids.append(track_id)

        # For this frame, track slot occupancy and which cars are in each slot
        slots_occupied_this_frame = set()
        cars_in_slots_this_frame = defaultdict(set)  # slot_id -> set(car_id)

        for slot_id, slot_box in slot_rects.items():
            for car_box, car_id in zip(car_boxes, car_ids):
                overlap = iou(slot_box, car_box)
                if overlap >= SLOT_IOU_THRESHOLD:
                    slots_occupied_this_frame.add(slot_id)
                    cars_in_slots_this_frame[slot_id].add(car_id)

        # Update frame counters & compute live time for each car in slots
        car_live_seconds = {}  # car_id -> parking time in seconds (current frame)
        for slot_id in slots_occupied_this_frame:
            slot_occupied_frames[slot_id] += 1
            for car_id in cars_in_slots_this_frame[slot_id]:
                car_parked_frames[car_id] += 1
                car_live_seconds[car_id] = car_parked_frames[car_id] / fps

        # Draw slot ROIs (red = occupied, green = free)
        for slot_id, slot_box in slot_rects.items():
            x1, y1, x2, y2 = map(int, slot_box)
            color = (0, 0, 255) if slot_id in slots_occupied_this_frame else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            text = f"Slot {slot_id}: {'Occ' if slot_id in slots_occupied_this_frame else 'Free'}"
            cv2.putText(
                frame,
                text,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )

        # Draw car boxes + ID + time in the middle of the box
        for (x1, y1, x2, y2), car_id in zip(car_boxes, car_ids):
            x1_i, y1_i, x2_i, y2_i = map(int, [x1, y1, x2, y2])

            # Car bounding box
            cv2.rectangle(frame, (x1_i, y1_i), (x2_i, y2_i), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"ID {car_id}",
                (x1_i, y1_i - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

            # If this car is currently considered parked (inside any slot), show its time
            if car_id in car_live_seconds:
                t_sec = car_live_seconds[car_id]
                # center of bbox
                cx = int((x1_i + x2_i) / 2)
                cy = int((y1_i + y2_i) / 2)

                label = f"{t_sec:.1f}s"
                cv2.putText(
                    frame,
                    label,
                    (cx - 20, cy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )

        # ---- Parking summary info panel (top-right) ----
        occupied_slots = len(slots_occupied_this_frame)
        free_slots = total_slots - occupied_slots

        text_lines = [
            "Parking Status",
            f"Total: {total_slots}",
            f"Occupied: {occupied_slots}",
            f"Free: {free_slots}",
        ]

        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (int(panel_x1), int(panel_y1)),
            (int(panel_x2), int(panel_y2)),
            (0, 0, 0),
            -1,
        )
        alpha = 0.6
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        y_text = panel_y1 + 30
        line_spacing = 35
        for i, txt in enumerate(text_lines):
            cv2.putText(
                frame,
                txt,
                (int(panel_x1 + 10), int(y_text + i * line_spacing)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
            )

        # Show
        cv2.imshow("Parking Monitor", frame)

        # Save to output video
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # -----------------------
    # 5. Print summary
    # -----------------------
    # print("=== SLOT OCCUPANCY (seconds) ===")
    # for slot_id, frames in slot_occupied_frames.items():
    #     seconds = frames / fps
    #     print(f"Slot {slot_id}: {seconds:.2f} s")

    # print("\n=== CAR PARKING TIME (seconds) ===")
    # for car_id, frames in car_parked_frames.items():
    #     seconds = frames / fps
    #     print(f"Car ID {car_id}: {seconds:.2f} s")

    # print(f"\n[INFO] Saved annotated video to: {OUTPUT_VIDEO_PATH}")


if __name__ == "__main__":
    main()
