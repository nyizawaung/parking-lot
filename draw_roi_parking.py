import cv2
import json
import os

VIDEO_PATH = "parking/source/parking.mp4"          # <-- change to your video path
OUTPUT_JSON = "parking/parking_slots.json"  # output file name

# Globals for mouse callback
drawing = False
start_x, start_y = -1, -1
current_rect = None  # (x1, y1, x2, y2)
slots = []           # list of rects: dicts with id, x1, y1, x2, y2

def mouse_callback(event, x, y, flags, param):
    global drawing, start_x, start_y, current_rect, slots

    # Left button pressed: start drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_x, start_y = x, y
        current_rect = None

    # Mouse move: update current rectangle if drawing
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            current_rect = (start_x, start_y, x, y)

    # Left button released: finalize rectangle
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        # Normalize coordinates (x1 < x2, y1 < y2)
        x1, y1 = start_x, start_y
        x2, y2 = x, y
        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])

        # Ignore very small rectangles
        if abs(x2 - x1) > 5 and abs(y2 - y1) > 5:
            slot_id = len(slots) + 1
            rect = {
                "id": slot_id,
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2)
            }
            slots.append(rect)
            print(f"Added slot {slot_id}: {rect}")

        current_rect = None


def main():
    global current_rect, slots

    # --- Load first frame from video ---
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError("Cannot read first frame from video.")

    clone = frame.copy()

    window_name = "Draw Parking Slots"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    print("Instructions:")
    print(" - Left click + drag: draw a rectangle for a parking slot")
    print(" - 'u': undo last slot")
    print(" - 'r': reset all slots")
    print(" - 's': save to JSON and exit")
    print(" - 'q' or ESC: exit without saving")

    while True:
        disp = clone.copy()

        # Draw existing slots
        for slot in slots:
            x1, y1, x2, y2 = slot["x1"], slot["y1"], slot["x2"], slot["y2"]
            cv2.rectangle(disp, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(disp, f"ID {slot['id']}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Draw current rectangle in green (while dragging)
        if current_rect is not None:
            x1, y1, x2, y2 = current_rect
            cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 1)

        cv2.imshow(window_name, disp)
        key = cv2.waitKey(10) & 0xFF

        # Undo last
        if key == ord('u'):
            if slots:
                removed = slots.pop()
                print(f"Removed slot {removed['id']}")
            else:
                print("No slots to undo.")

        # Reset all
        elif key == ord('r'):
            slots = []
            print("All slots cleared.")

        # Save and exit
        elif key == ord('s'):
            data = {"slots": slots}
            with open(OUTPUT_JSON, "w") as f:
                json.dump(data, f, indent=2)
            print(f"Saved {len(slots)} slots to {OUTPUT_JSON}")
            break

        # Quit without saving
        elif key == ord('q') or key == 27:  # ESC
            print("Exit without saving.")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
