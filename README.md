# parking-lot
Parking lot monitoring system with customizable parking area, display total parking time, available/occupied slots information.


# How to use
Define parking slots first by running  -> python draw_roi_parking.py
Instructions:
    Left click + drag: draw a rectangle for a parking slot
    'u': undo last slot
    'r': reset all slots
    's': save to JSON and exit
    'q' or ESC: exit without saving


After defining the parking slots, run the parking video by using
python yolo_parking.py --video {source} --output {output.mp4} --slots {parking_slots.json} --model {default is yolov8m}