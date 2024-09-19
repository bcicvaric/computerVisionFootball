import cv2

def read_video(video_path):
    frames = list()
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    return frames

def save_video(frames, output_video_path, fps=24):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frames[0].shape[1], frames[0].shape[0]))
    for frame in frames:
        out.write(frame)

    out.release()

def measure_distance(p1, p2, measure_type='l2'):
    if measure_type == 'l2':
        return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5
    if measure_type == 'lxy':
        return p1[0] - p2[0], p1[1] - p2[1]