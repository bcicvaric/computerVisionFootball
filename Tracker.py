from utils import measure_distance
import cv2
import pickle
import numpy as np
from ultralytics import YOLO
import supervision as sv
import pandas as pd
import os

class Tracker:

    def __init__(self, model_path):

        court_width = 68
        court_length = 23.32

        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
        self.max_ball_distance = 80

        self.pixel_vertices = np.array([
            [110, 1035], [265, 275], [910, 260], [1640, 915]
        ])

        self.target_vertices = np.array([
            [0, court_width], [0, 0], [court_length, 0], [court_length, court_width]
        ])

        self.pixel_vertices = self.pixel_vertices.astype(np.float32)
        self.target_vertices = self.target_vertices.astype(np.float32)

        self.perspective_trf = cv2.getPerspectiveTransform(self.pixel_vertices, self.target_vertices)

        self.frame_window = 12
        self.frame_rate = 24

    def add_speed_to_tracks(self, tracks):

        total_frames = len(tracks['players'])
        for frame_num in range(0, total_frames - self.frame_window, self.frame_window):

            last_frame = frame_num + self.frame_window
            for track_id, _ in tracks['players'][frame_num].items():

                # SKIP PLAYERS THAT ARE NOT IN THE LAST FRAME
                if track_id not in tracks['players'][last_frame]:
                    continue
                # SKIP PLAYERS THAT DONT HAVE transformed_position IN THE FIRST OR LAST FRAME
                if 'transformed_position' not in tracks['players'][frame_num][track_id]:
                    continue
                # SKIP PLAYERS THAT DONT HAVE transformed_position IN THE FIRST OR LAST FRAME
                if 'transformed_position' not in tracks['players'][last_frame][track_id]:
                    continue
                start_position = tracks['players'][frame_num][track_id]['transformed_position']
                end_position = tracks['players'][last_frame][track_id]['transformed_position']

                distance_covered = measure_distance(start_position, end_position)
                time_elapsed = (last_frame - frame_num) / self.frame_rate
                speed_kmh = distance_covered / time_elapsed * 3.6

                for frame_num_in_batch in range(frame_num, last_frame):
                    try:
                        tracks['players'][frame_num_in_batch][track_id]['speed'] = speed_kmh
                    except:
                        pass

    def interpolate_ball(self, ball_positions):
        ball_positions = [x.get(1, {}).get('bbox', []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        # Interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        int_ball_positions = [{1: {'bbox': x}} for x in df_ball_positions.to_numpy().tolist()]

        return int_ball_positions

    def detect_frames(self, frames, batch_size=24):

        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:(i + 24)], conf=0.1)
            detections += detections_batch

        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):

        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks

        tracks = {
            "ball": [],
            "players": [],
            "referees": []
        }

        detections = self.detect_frames(frames)

        for frame_num, detection in enumerate(detections):

            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}

            # CONVERT TO SUPERVISION DETECTION FORMAT
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # REPLACE GOALKEEPERS WITH PLAYERS
            for obj_num, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == 'goalkeeper':
                    detection_supervision.class_id[obj_num] = cls_names_inv['player']

            # TRACK OBJECTS
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks["players"].append({})
            tracks["ball"].append({})
            tracks["referees"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                class_id = frame_detection[3]
                track_id = frame_detection[4]

                if class_id == cls_names_inv["player"]:
                    tracks["players"][frame_num][track_id] = {'bbox': bbox}
                if class_id == cls_names_inv["referee"]:
                    tracks["referees"][frame_num][track_id] = {'bbox': bbox}

            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                class_id = frame_detection[3]
                track_id = frame_detection[4]

                if class_id == cls_names_inv["ball"]:
                    tracks["ball"][frame_num][1] = {'bbox': bbox}

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks

    def transform_point(self, point):

        p = (int(point[0]), int(point[1]))
        is_inside = cv2.pointPolygonTest(self.pixel_vertices, p, False) >= 0
        if not is_inside:
            return None
        reshaped_point = point.reshape(-1, 1, 2).astype(np.float32)
        transform_point = cv2.perspectiveTransform(reshaped_point, self.perspective_trf)

        return transform_point.reshape(-1, 2)

    def add_positions_to_tracks(self, tracks, camera_movement_per_frame):

        for object_, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    # ADD POSITION
                    position = (int((bbox[0] + bbox[2]) / 2), int(bbox[3]))
                    tracks[object_][frame_num][track_id]['position'] = position
                    # ADD ADJUSTED POSITION
                    camera_movement = camera_movement_per_frame[frame_num]
                    adjusted_position = (
                    int((bbox[0] + bbox[2]) / 2) + camera_movement[0], int(bbox[3]) + camera_movement[1])
                    tracks[object_][frame_num][track_id]['adjusted_position'] = adjusted_position
                    # ADD TRANSFORMED POSITION
                    adjusted_position = np.array(adjusted_position)
                    transformed_position = self.transform_point(adjusted_position)
                    if transformed_position is not None:
                        transformed_position = transformed_position.squeeze().tolist()
                        tracks[object_][frame_num][track_id]['transformed_position'] = transformed_position

    def draw_elipse(self, frame, bbox, color, track_id):

        elipse_center = (int((bbox[0] + bbox[2]) / 2), int(bbox[3]))

        cv2.ellipse(frame, center=elipse_center,
                    axes=(40, 16),
                    angle=0.0,
                    startAngle=-45,
                    endAngle=225,
                    color=color,
                    thickness=2,
                    lineType=cv2.LINE_4)

        if track_id is not None:
            x1 = elipse_center[0] - 25
            y1 = elipse_center[1] + 2
            x2 = elipse_center[0] + 25
            y2 = elipse_center[1] + 30

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, cv2.FILLED)

            x1_text = elipse_center[0] - 22
            y1_text = elipse_center[1] + 22

            cv2.putText(frame, f"{track_id}", (x1_text, y1_text), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        return frame

    def draw_triangle(self, frame, bbox, color, track_id=None):

        y0 = int(bbox[1])
        x0 = int((bbox[0] + bbox[2]) / 2)

        triangle_points = np.array([[x0, y0], [x0 - 10, y0 - 20], [x0 + 10, y0 - 20]])
        cv2.drawContours(frame, np.int32([triangle_points]), 0, color, cv2.FILLED)
        cv2.drawContours(frame, np.int32([triangle_points]), 0, (0, 0, 0), 2)

        return frame

    def assign_ball_to_player(self, players, ball_bbox):

        min_player_ball_distance = 80
        assigned_player = -1

        ball_center = (int((ball_bbox[0] + ball_bbox[2]) / 2), int((ball_bbox[1] + ball_bbox[3]) / 2))

        for player_id, player in players.items():
            player_bbox = player['bbox']
            player_center = (int((player_bbox[0] + player_bbox[2]) / 2), player_bbox[3])
            m = measure_distance(player_center, ball_center)

            if m < self.max_ball_distance:
                min_player_ball_distance = m
                assigned_player = player_id

        return assigned_player

    def draw_anotations(self, frames, tracks):

        output_video_frames = []

        for frame_num, frame in enumerate(frames):

            frame = frame.copy()

            # draw reference rectangle in first 1 sec
            if frame_num < 24:
                cv2.drawContours(frame, np.int32([self.pixel_vertices]), 0, (0, 0, 255), 2)

            player_dict = tracks['players'][frame_num]
            ball_dict = tracks['ball'][frame_num]
            referee_dict = tracks['referees'][frame_num]

            # Draw Players & stats:
            for track_id, player in player_dict.items():
                color = (255, 120, 120) if player['team'] == 1 else (120, 120, 255)
                frame = self.draw_elipse(frame, player['bbox'], color, track_id)

                if player.get('has_ball', False):
                    self.draw_triangle(frame, player['bbox'], (0, 0, 255))

                if 'speed' in player:
                    speed = player['speed']
                    text = f'{speed:.1f} km/h'
                    x1_text = int((player['bbox'][0] + player['bbox'][2]) / 2 - 30)
                    y1_text = int((player['bbox'][3] + 43))
                    cv2.putText(frame, text, (x1_text, y1_text)
                                , cv2.FONT_HERSHEY_SIMPLEX, 0.5 + 0.1 * int(speed/5), (0, 0, speed*2), 2)

            # Draw Ball Cursor:
            for _, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball['bbox'], (0, 255, 0))

            output_video_frames.append(frame)

        return output_video_frames