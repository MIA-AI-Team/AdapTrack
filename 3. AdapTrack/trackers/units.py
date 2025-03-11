import numpy as np
from trackers.kalman_filter import KalmanFilter

class Detection(object):
    def __init__(self, tlbr, confidence, feature=None):
        # Save box
        self.tlbr = tlbr
        self.tlwh = tlbr.copy()
        self.tlwh[2:] -= self.tlwh[:2]

        # Save others
        self.confidence = confidence
        self.feature = feature

    def to_cxcyah(self):
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret


class TrackState:
    Tentative = 1
    Confirmed = 2
    Deleted = 3


class Track:
    def __init__(self, cxcyah, track_id, opt, score=None, feature=None):
        # Save opt for later use
        self.opt = opt

        # Set parameters
        self.track_id = track_id

        # Initialization
        self.hits = 1
        self.time_since_update = 0
        self.state = TrackState.Tentative

        # Set scores
        self.scores = []
        if score is not None:
            self.scores.append(score)

        # Set features
        self.features = []
        if feature is not None:
            feature /= np.linalg.norm(feature)
            self.features.append(feature)

        # Set Kalman filter
        self.kf = KalmanFilter()
        self.mean, self.covariance = self.kf.initiate(cxcyah)

    def predict(self):
        # Predict Kalman state
        self.mean, self.covariance = self.kf.predict(self.mean, self.covariance)
        self.time_since_update += 1

    def update(self, detection):
        # Update Kalman state
        self.mean, self.covariance = self.kf.update(self.mean, self.covariance,
                                                    detection.to_cxcyah(), detection.confidence)

        # Normalize feature if provided
        if detection.feature is not None:
            feature = detection.feature / np.linalg.norm(detection.feature)
            # Update feature with EMA
            beta = (detection.confidence - self.opt.conf_thresh) / (1 - self.opt.conf_thresh)
            alpha = self.opt.ema_beta + (1 - self.opt.ema_beta) * (1 - beta)
            smooth_feat = alpha * self.features[-1] + (1 - alpha) * feature
            self.features = [smooth_feat / np.linalg.norm(smooth_feat)]
        else:
            self.features = [self.features[-1]] if self.features else []

        # Update others
        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self.opt.min_len:
            self.state = TrackState.Confirmed

    def to_tlwh(self):
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    def to_tlbr(self):
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret

    def mark_missed(self):
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self.opt.max_age:
            self.state = TrackState.Deleted

    def is_tentative(self):
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        return self.state == TrackState.Deleted