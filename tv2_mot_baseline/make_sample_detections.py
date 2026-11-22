"""Tạo video mẫu + detections.csv GIẢ LẬP (đúng định dạng detr_demo.py) để thử mot_from_detr.py khi chưa có file thật.
Chạy:  python make_sample_detections.py
Tạo:   data/sample.mp4   outputs/detections.csv
"""
import csv, os
import mot_video_demo as mv

os.makedirs("data", exist_ok=True); os.makedirs("outputs", exist_ok=True)
objs, n = mv.SCENARIO_OBJS["occlusion"]
mv.render_video("data/sample.mp4", objs, n)
dets = mv.detections_from_video(mv.read_video("data/sample.mp4"))
with open("outputs/detections.csv", "w", newline="", encoding="utf-8") as fp:
    w = csv.writer(fp)
    w.writerow(["frame_index", "time_s", "label", "score", "x1", "y1", "x2", "y2"])
    for i, d in enumerate(dets):
        for b in d:
            w.writerow([i, round(i / mv.FPS, 3), "person", 0.95, *map(int, b)])
print("Đã tạo data/sample.mp4 và outputs/detections.csv")
print("Chạy tiếp: python mot_from_detr.py --input data/sample.mp4 --detections outputs/detections.csv")
