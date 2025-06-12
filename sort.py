from collections import deque
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
import numpy as np
from scipy.optimize import linear_sum_assignment # For Hungarian algorithm

class KalmanBoxTracker:
    count = 0
    def __init__(self, bbox):
        # define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1,0,0,0,1,0,0],
                              [0,1,0,0,0,1,0],
                              [0,0,1,0,0,0,1],
                              [0,0,0,1,0,0,0],
                              [0,0,0,0,1,0,0],
                              [0,0,0,0,0,1,0],
                              [0,0,0,0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0,0,0,0],
                              [0,1,0,0,0,0,0],
                              [0,0,1,0,0,0,0],
                              [0,0,0,1,0,0,0]])

        self.kf.R[2:,2:] *= 10.
        self.kf.P[4:,4:] *= 1000. # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q = Q_discrete_white_noise(dim=7, dt=0.1, var=0.1)

        self.kf.x[:4] = self.convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = deque([])
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = deque([])
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(self.convert_bbox_to_z(bbox))

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if((self.kf.x[6]+self.kf.x[2]) < 0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if(self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self.convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return self.convert_x_to_bbox(self.kf.x)

    @staticmethod
    def convert_bbox_to_z(bbox):
        """
        Converts bbox (x,y,w,h) to measurement z (cx,cy,s,r) where
        cx,cy is center x,y, s is scale/area, r is aspect ratio.
        """
        w = bbox[2]
        h = bbox[3]
        x = bbox[0]
        y = bbox[1]
        cx = x + w/2.
        cy = y + h/2.
        s = w * h # scale is just area
        r = w / float(h)
        return np.array([cx, cy, s, r]).reshape((4, 1))

    @staticmethod
    def convert_x_to_bbox(x):
        """
        Converts a state vector x (cx,cy,s,r,vx,vy,vs) to bbox (x,y,w,h)
        """
        cx = x[0]
        cy = x[1]
        s = x[2]
        r = x[3]
        w = np.sqrt(s * r)
        h = s / w
        x = cx - w/2.
        y = cy - h/2.
        return np.array([x, y, w, h]).reshape((1, 4))


def iou_batch(bb_test, bb_gt):
    """
    Computes IOU between two bounding boxes in the form [x,y,w,h]
    """
    x_test, y_test, w_test, h_test = bb_test[0], bb_test[1], bb_test[2], bb_test[3]
    x_gt, y_gt, w_gt, h_gt = bb_gt[0], bb_gt[1], bb_gt[2], bb_gt[3]

    xx1 = np.maximum(x_test, x_gt)
    yy1 = np.maximum(y_test, y_gt)
    xx2 = np.minimum(x_test + w_test, x_gt + w_gt)
    yy2 = np.minimum(y_test + h_test, y_gt + h_gt)

    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((w_test*h_test + w_gt*h_gt) - wh)
    return(o)

class Sort:
    def __init__(self, max_age=1, min_hits=3):
        """
        Sets key parameters for SORT
        max_age: maximum number of frames to keep a track alive without a hit
        min_hits: minimum number of hits to be counted as a track
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.trackers = []
        self.frame_count = 0

    def update(self, detections):
        """
        Params:
          detections: a numpy array of detections in the format [[x,y,w,h],[x,y,w,h],...]
        Returns:
          A numpy array of tracks in the format [[x,y,w,h,id],[x,y,w,h,id],...]
        """
        self.frame_count += 1
        # Predict current location of existing trackers
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        for t, trk in enumerate(self.trackers):
            pos = trk.predict()[0]
            trks[t, :] = [pos[0], pos[1], pos[2], pos[3], 0] # 0 is placeholder for ID
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.fix_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        if detections.shape[0] > 0:
            # Calculate cost matrix (IOU)
            iou_matrix = np.zeros((detections.shape[0], trks.shape[0]), dtype=np.float32)
            for d, det in enumerate(detections):
                for t, trk in enumerate(trks):
                    iou_matrix[d, t] = iou_batch(det, trk)

            matched_indices = linear_sum_assignment(-iou_matrix) # Minimize negative IOU (maximize IOU)
            matched_indices = np.asarray(matched_indices).T

            unmatched_detections = []
            for d, det in enumerate(detections):
                if d not in matched_indices[:, 0]:
                    unmatched_detections.append(d)

            unmatched_trackers = []
            for t, trk in enumerate(trks):
                if t not in matched_indices[:, 1]:
                    unmatched_trackers.append(t)

            # Filter out low IOU matches
            matches = []
            for m in matched_indices:
                if iou_matrix[m[0], m[1]] < 0.3: # IOU threshold
                    unmatched_detections.append(m[0])
                    unmatched_trackers.append(m[1])
                else:
                    matches.append(m.reshape(1, 2))
            matches = np.concatenate(matches, axis=0) if len(matches) > 0 else np.empty((0, 2), dtype=int)

            # Update matched trackers with assigned detections
            for m in matches:
                self.trackers[m[1]].update(detections[m[0], :])

            # Create new trackers for unmatched detections
            for i in unmatched_detections:
                self.trackers.append(KalmanBoxTracker(detections[i, :]))
        else: # No detections this frame
            unmatched_trackers = list(range(len(self.trackers)))

        # Remove dead trackers
        ret = []
        for i in reversed(range(len(self.trackers))):
            d = self.trackers[i]
            if d.time_since_update > self.max_age:
                self.trackers.pop(i)
            elif (d.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                x = d.get_state()[0]
                ret.append(np.concatenate((x, [d.id])).reshape(1, -1))

        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))