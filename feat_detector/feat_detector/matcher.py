import pkg_resources
import cv2 as cv
import matplotlib.pyplot as plt
import feat_detector.superpoint as superpoint
import numpy as np


class Matcher:
    def __init__(self, backend):
        self.back = backend

    def sift_match(self, des1, des2):
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)

        flann = cv.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)
        # ratio test as per Lowe's paper
        return good

    def orb_match(self, des1, des2):
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH,
                            table_number=6,
                            key_size=12,
                            multi_probe_level=1)
        search_params = dict(checks=50)
        flann = cv.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        matches = [match for match in matches if len(match) == 2]
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)
        # ratio test as per Lowe's paper
        return good

    def sup_match(self, des1, des2):
        return superpoint.sp_match(des1, des2, 0.1)

    def matching(self, des1, des2):
        if self.back == 'ORB':
            return self.orb_match(des1, des2)
        if self.back == 'SIFT':
            return self.sift_match(des1, des2)
        if self.back == 'SUPERPOINT':
            return self.sup_match(des1, des2)
        return -1