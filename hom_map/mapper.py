import cv2 as cv
import numpy as np
from feat_detector import feature_detector as f
from feat_detector import matcher as m


class Mapper:
    def __init__(self):
        self.map3d = []
        self.map2d = []

    def decompose(self, intr, proj):
        mat = intr.dot(proj)
        R = np.array([[mat[i][j] for j in range(3)] for i in range(3)])
        t = np.array([mat[0][3], mat[1][3], mat[2][3]])
        return R, t

    def update_map(self, proj_mat1, proj_mat2, image1, image2, intrinsic):
        detect = f.FeatureDetector('SUPERPOINT', [1, 0.01, 0.2])
        match = m.Matcher('ORB')
        kp, des = detect.detect_and_compute([image1, image2])
        kp, des = detect.compute_orb([image1, image2], kp)
        matches = match.matching(des[0], des[1])
        pts1 = np.array([kp[0][matches[j].queryIdx].pt for j in range(len(matches))])
        pts2 = np.array([kp[1][matches[j].trainIdx].pt for j in range(len(matches))])
        des2 = np.array([des[1][matches[j].trainIdx] for j in range(len(matches))])
        pts1 = pts1.astype(int)
        pts2 = pts2.astype(int)
        F, mask = cv.findFundamentalMat(pts1, pts2, cv.FM_RANSAC, 4, 0.99)
        pts1 = np.array(pts1[mask.ravel() == 1])
        pts2 = np.array(pts2[mask.ravel() == 1])
        des2 = np.array(des2[mask.ravel() == 1])
        pts = cv.triangulatePoints(proj_mat1, proj_mat2, np.float32(pts1.T),
                                   np.float32(pts2.T))
        pts3d = np.array([np.array([point[0], point[1], point[2]]) / point[3] for point in pts.T])
        R, t = self.decompose(np.linalg.inv(intrinsic), proj_mat2)
        vec, j = cv.Rodrigues(R)
        projected, j = cv.projectPoints(pts3d, vec, t, np.array(intrinsic), np.array([]))
        projected = np.array([projected[i][0] for i in range(len(projected))])
        err = projected - np.float32(pts2)
        pts3d = np.array([pts3d[i] for i in range(len(pts3d)) if np.sqrt(err[i][0]**2 + err[i][1]**2) < 20])
        pts2 = np.array([pts2[i] for i in range(len(pts2)) if np.sqrt(err[i][0]**2 + err[i][1]**2) < 20])
        des2 = np.array([des2[i] for i in range(len(des2)) if np.sqrt(err[i][0]**2 + err[i][1]**2) < 20])
        self.map3d.append(pts3d)
        self.map2d.append([pts2, des2])

