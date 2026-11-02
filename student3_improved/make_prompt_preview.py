from __future__ import annotations
import argparse
from pathlib import Path
import cv2

p = argparse.ArgumentParser(description="Save frame 0 with a normalized coordinate grid for SAM2 prompt editing.")
p.add_argument("--input", required=True, type=Path)
p.add_argument("--output", type=Path, default=Path("data/student3_improved/frame0_prompt_grid.jpg"))
a = p.parse_args()
cap = cv2.VideoCapture(str(a.input))
ok, frame = cap.read()
cap.release()
if not ok:
    raise SystemExit(f"Could not read first frame from {a.input}")
h, w = frame.shape[:2]
for x in range(0, 101, 10):
    px = round(w * x / 100)
    cv2.line(frame, (px, 0), (px, h - 1), (0, 255, 255), 1)
    cv2.putText(frame, str(x), (min(px + 2, w - 20), 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1, cv2.LINE_AA)
for y in range(0, 101, 10):
    py = round(h * y / 100)
    cv2.line(frame, (0, py), (w - 1, py), (0, 255, 255), 1)
    cv2.putText(frame, str(y), (2, min(py + 12, h - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1, cv2.LINE_AA)
a.output.parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(a.output), frame)
print(f"Saved prompt grid: {a.output.resolve()}")
