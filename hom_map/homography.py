import cv2 as cv
import numpy as np
from feat_detector import feature_detector as f
from feat_detector import matcher as m


class Homography:
    @staticmethod
    def estimate(new_frame, prev_kp, prev_desc, world_points, cam_matr):
        detect = f.FeatureDetector('SUPERPOINT', [1, 0.01, 0.2])
        match = m.Matcher('ORB')
        new_kp, new_desc = detect.detect_and_compute([new_frame])
        new_kp, new_desc = detect.compute_orb([new_frame], new_kp)
        matches = match.matching(prev_desc, new_desc[0])
        pts3d = np.array([world_points[matches[j].queryIdx] for j in range(len(matches))])
        pts1 = np.array([prev_kp[matches[j].queryIdx] for j in range(len(matches))])
        pts2 = np.array([new_kp[0][matches[j].trainIdx].pt for j in range(len(matches))])
        pts1 = pts1.astype(int)
        pts2 = pts2.astype(int)
        F, mask = cv.findFundamentalMat(pts1, pts2, cv.FM_RANSAC, 4, 0.99)
        pts3d = np.array(pts3d[mask.ravel() == 1])
        pts2 = np.array(pts2[mask.ravel() == 1])
        return cv.solvePnP(pts3d, np.float32(pts2), cam_matr, np.array([]), flags=cv.SOLVEPNP_ITERATIVE)
