from sklearn.cluster import KMeans

class TeamAssigner:

    def __init__(self):

        self.team_colors = {}
        self.player_team_dict = {}

    def get_player_color(self, frame, bbox):

        cropped_img = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        return self.get_jersey_color(cropped_img)

    def get_jersey_color(self, image):

        top_half_img = image[0:int(image.shape[0] / 2), :]
        image_2d = top_half_img.reshape(-1, 3)

        kmeans = KMeans(n_clusters=2, random_state=123)
        kmeans.fit(image_2d)

        labels = kmeans.labels_
        cluster_img = labels.reshape(top_half_img.shape[0], top_half_img.shape[1])

        corner_clusters = [cluster_img[0, 0], cluster_img[-1, 0], cluster_img[0, -1], cluster_img[-1, -1]]
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)

        player_cluster = 1 - non_player_cluster

        jersey_color = kmeans.cluster_centers_[player_cluster]

        return jersey_color

    def assign_team_color(self, frame, player_detections):

        player_colors = []

        for _, player_detection in player_detections.items():
            bbox = player_detection['bbox']
            cropped_img = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
            player_color = self.get_jersey_color(cropped_img)

            player_colors.append(player_color)

        self.kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1)
        self.kmeans.fit(player_colors)

        labels = self.kmeans.labels_

        self.team_colors[1] = self.kmeans.cluster_centers_[0]
        self.team_colors[2] = self.kmeans.cluster_centers_[1]

    def get_player_team(self, frame, player_bbox, player_id):

        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        player_color = self.get_player_color(frame, player_bbox)

        team_id = self.kmeans.predict(player_color.reshape(-1, 3))[0] + 1

        self.player_team_dict[player_id] = team_id

        return team_id