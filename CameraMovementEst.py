from utils import measure_distance
import cv2
import pickle
import numpy as np
import os

class CameraMovementEst:

    def __init__(self, frame):

        self.min_reg_distance = 5

        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

        first_frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        mask_features = np.zeros_like(first_frame_gray)
        mask_features[:, :100] = 1
        mask_features[:, -100:] = 1

        self.features = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=3,
            blockSize=7,
            mask=mask_features
        )

    def get_camera_movement(self, frames, read_from_stub=False, stub_path=None):

        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                # pickle.load(tracks, f)
                camera_movement = pickle.load(f)
            return camera_movement

        camera_movement = [[0, 0]] * len(frames)

        old_gray = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)
        old_features = cv2.goodFeaturesToTrack(old_gray, **self.features)

        for frame_num in range(1, len(frames)):
            frame_gray = cv2.cvtColor(frames[frame_num], cv2.COLOR_RGB2GRAY)
            new_features, _, _ = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, old_features, None, **self.lk_params)

            max_dist = 0
            cam_move_x, cam_move_y = 0, 0

            for i, (new, old) in enumerate(zip(new_features, old_features)):

                new_coord = new.ravel()
                old_coord = old.ravel()

                dist = measure_distance(old_coord, new_coord)
                if dist > max_dist:
                    max_dist = dist
                    cam_move_x, cam_move_y = measure_distance(old_coord, new_coord, 'lxy')

            if max_dist > self.min_reg_distance:
                camera_movement[frame_num] = [cam_move_x, cam_move_y]
                old_features = cv2.goodFeaturesToTrack(old_gray, **self.features)

            old_gray = frame_gray.copy()

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(camera_movement, f)

        return camera_movement

    def draw_camera_movement(self, frames, camera_movement_per_frame):

        output_frames = []

        for frame_num, frame in enumerate(frames):
            frame = frame.copy()
            overlay = frame.copy()

            cv2.rectangle(overlay, (0, 0), (500, 70), (255, 255, 255), -1)
            alpha = 0.6
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            x_movement, y_movement = camera_movement_per_frame[frame_num]

            frame = cv2.putText(frame, f"Cam X Movement: {x_movement:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 0, 0), 3)
            frame = cv2.putText(frame, f"Cam Y Movement: {y_movement:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 0, 0), 3)

            output_frames.append(frame)

        return output_frames