This system detects vehicles, tracks them over time, and provides real-time analytics for any parking area.

🔧 What it can do
Customizable parking regions (draw your own slots with an interactive ROI tool)
Real-time count of occupied vs. available parking spaces
Per-vehicle parking duration tracked automatically
Easy to adapt for new videos, layouts, or models

⚙️ How it works
Define parking slots interactively using draw_roi_parking.py
Run the detection + tracking pipeline with yolo_parking.py
Get a fully annotated output video with live stats

💡 Why it’s useful
A fast, lightweight prototype for:
Parking management & monitoring
Garage automation
Drone/aerial analytics
Students or engineers learning applied computer vision

# How to run
Define parking slots first by running  -> python draw_roi_parking.py
Instructions:
    Left click + drag: draw a rectangle for a parking slot
    'u': undo last slot
    'r': reset all slots
    's': save to JSON and exit
    'q' or ESC: exit without saving


After defining the parking slots, run the parking video by using
python yolo_parking.py --video {source} --output {output.mp4} --slots {parking_slots.json} --model {default is yolov8m}
