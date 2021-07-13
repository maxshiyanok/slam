import pkg_resources
import cv2 as cv
import matplotlib.pyplot as plt
import feat_detector.superpoint as superpoint
import numpy as np


class FeatureDetector:
    def __init__(self, backend, argv):
        self.back = backend
        self.argv = argv

    def sift(self, image):
        sift = cv.SIFT_create(nfeatures=self.argv[0], contrastThreshold=self.argv[1])
        kp = []
        des = []
        for i in image:
            gray = cv.cvtColor(i, cv.COLOR_BGR2GRAY)
            kp1, des1 = sift.detectAndCompute(gray, None)
            kp.append(kp1)
            des.append(des1)
        return kp, des

    def orb(self, image):
        orb = cv.ORB_create(nfeatures=self.argv[0], fastThreshold=self.argv[1])
        kp = []
        des = []
        for i in image:
            gray = cv.cvtColor(i, cv.COLOR_BGR2GRAY)
            kp1, des1 = orb.detectAndCompute(gray, None)
            kp.append(kp1)
            des.append(des1)
        return kp, des

    def mysuperpoint(self, image):
        sp = superpoint.SuperPointFrontend(pkg_resources.resource_stream(__name__, 'data//superpoint_v1.pth'),
                                           self.argv[0], self.argv[1], self.argv[2])
        kp2 = []
        des = []
        for i in image:
            kp1, des1, heatmap = sp.get_features(i)
            kp2.append(np.array(kp1).astype(int))
            des.append(np.array(des1).astype(int))
        kp = [[cv.KeyPoint(kp2[i][0][j], kp2[i][1][j], 1) for j in range(len(kp2[i][0]))] for i in range(len(kp2))]
        return kp, des

    def detect_and_compute(self, image):
        if self.back == 'ORB':
            return self.orb(image)
        if self.back == 'SIFT':
            return self.sift(image)
        if self.back == 'SUPERPOINT':
            return self.mysuperpoint(image)
        return -1

    def compute_orb(self, image, kp):
        orb = cv.ORB_create()
        kp1 = []
        des = []
        for i in range(len(image)):
            gray = cv.cvtColor(image[i], cv.COLOR_BGR2GRAY)
            kp2, des1 = orb.compute(gray, kp[i])
            kp1.append(kp2)
            des.append(des1)
        return kp1, des


# detect = FeatureDetector('SUPERPOINT', [1, 0.01, 0.1, 0.5])
# img = []
# image = cv.imread("C://Users//USER/Desktop//test//3.jpg")
# image1 = cv.imread("C://Users//USER/Desktop//test//4.jpg")
# img.append(image)
# img.append(image1)
# kp, des = detect.compute(img)
# match = detect.matching(des[0],des[1])
# print(len(kp[0][0]))
# print(len(match[0]))
# view(cv.drawKeypoints(image, kp[0], None, color=(0, 255,0)))
