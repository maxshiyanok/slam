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
        sp = superpoint.SuperPointFrontend( pkg_resources.resource_stream(__name__, 'data//superpoint_v1.pth'), self.argv[0], self.argv[1], self.argv[2])
        kp = []
        des = []
        for i in image:
            kp1, des1, heatmap = sp.get_features(i)
            kp.append(np.array(kp1).astype(int))
            des.append(np.array(des1).astype(int))
        return kp, des

    def compute(self, image):
        if self.back == 'ORB':
            return self.orb(image)
        if self.back == 'SIFT':
            return self.sift(image)
        if self.back == 'SUPERPOINT':
            return self.mysuperpoint(image)
        return -1
    
    def sift_match(self, des1, des2):
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)
        
        flann = cv.FlannBasedMatcher(index_params,search_params)
        matches = flann.knnMatch(des1,des2,k=2)
        good = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good.append([m])
        # ratio test as per Lowe's paper
        return good
        
    def orb_match(self, des1, des2):
        FLANN_INDEX_LSH = 6
        index_params= dict(algorithm=FLANN_INDEX_LSH,
                   table_number=6,
                   key_size=12,
                   multi_probe_level=1)
        search_params = dict(checks=50)
        flann = cv.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        matches = [match for match in matches if len(match) == 2]
        good = []
        for m, n in matches:
            if m.distance < 0.75*n.distance:
                good.append(m)
        # ratio test as per Lowe's paper
        return good
    
    def sup_match(self, des1, des2):
        sp = superpoint.SuperPointFrontend(pkg_resources.resource_stream(__name__, 'data//superpoint_v1.pth'), self.argv[0], self.argv[1], self.argv[2])
        return sp.sp_match(des1, des2, self.argv[3])
    
    def matching(self, des1, des2):
        if self.back == 'ORB':
            return self.orb_match(des1, des2)
        if self.back == 'SIFT':
            return self.sift_match(des1, des2)
        if self.back == 'SUPERPOINT':
            return self.sup_match(des1, des2)
        return -1


def view(image, size=(10, 10)):
    plt.figure(figsize=size)
    plt.imshow(image)
    plt.show()


#detect = FeatureDetector('SUPERPOINT', [1, 0.01, 0.1, 0.5])
#img = []
#image = cv.imread("C://Users//USER/Desktop//test//3.jpg")
#image1 = cv.imread("C://Users//USER/Desktop//test//4.jpg")
#img.append(image)
#img.append(image1)
#kp, des = detect.compute(img)
#match = detect.matching(des[0],des[1])
#print(len(kp[0][0]))
#print(len(match[0]))
#view(cv.drawKeypoints(image, kp[0], None, color=(0, 255,0)))