import numpy as np
from trackers.cmc import CMC
from trackers import metrics
from trackers.units import Track
from trackers import linear_assignment

def apply_cmc(track, warp_matrix):
    # Placeholder for CMC application
    pass

class Tracker:
    def __init__(self, metric, vid_name, opt):
        self.metric = metric
        self.opt = opt  # Store opt
        self.tracks = []
        self.next_id = 1
        try:
            self.cmc = CMC(vid_name)
        except FileNotFoundError:
            print(f"No GMC file found for {vid_name}, disabling camera motion compensation.")
            self.cmc = None

    def initiate_track(self, detection):
        self.tracks.append(Track(detection.to_cxcyah(), self.next_id, self.opt, detection.confidence, detection.feature))
        self.next_id += 1

    def predict(self):
        for track in self.tracks:
            track.predict()

    def camera_update(self):
        if self.cmc is not None:
            warp_matrix = self.cmc.get_warp_matrix()
            for track in self.tracks:
                apply_cmc(track, warp_matrix)

    def gated_metric(self, tracks, detections, track_indices, detection_indices):
        targets = np.array([tracks[i].track_id for i in track_indices])
        features = np.array([detections[i].feature for i in detection_indices])
        cost_matrix = self.metric.distance(features, targets)
        cost_matrix_min = np.min(cost_matrix)
        cost_matrix_max = np.max(cost_matrix)
        cost_matrix = linear_assignment.gate_cost_matrix(cost_matrix, tracks, detections,
                                                         track_indices, detection_indices,
                                                         self.opt.gating_lambda)  # Pass gating_lambda
        return cost_matrix, cost_matrix_min, cost_matrix_max

    def match(self, detections):
        confirmed_tracks = [i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [i for i, t in enumerate(self.tracks) if not t.is_confirmed()]
        matches_a, _, unmatched_detections = \
            linear_assignment.min_cost_matching([self.gated_metric, metrics.iou_constraint, True],
                                                self.opt.max_distance, self.tracks,
                                                detections, confirmed_tracks)
        unmatched_tracks_a = list(set(confirmed_tracks) - set(k for k, _ in matches_a))
        candidates = unconfirmed_tracks + [k for k in unmatched_tracks_a if self.tracks[k].time_since_update == 1]
        unmatched_tracks_a = [k for k in unmatched_tracks_a if self.tracks[k].time_since_update != 1]
        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching([metrics.iou_cost, None, True], self.opt.max_iou_distance, self.tracks,
                                                detections, candidates, unmatched_detections)
        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def update(self, detections):
        matches, unmatched_tracks, unmatched_detections = self.match(detections)
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(detections[detection_idx])
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        for detection_idx in unmatched_detections:
            if detections[detection_idx].confidence >= self.opt.conf_thresh:
                self.initiate_track(detections[detection_idx])
        self.tracks = [t for t in self.tracks if not t.is_deleted()]
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
        self.metric.partial_fit(np.asarray(features), np.asarray(targets), active_targets)