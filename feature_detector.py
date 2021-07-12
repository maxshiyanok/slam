from feat_detector import feature_detector as f
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os


def cmp(a, b):
    return a[0] == b[0] and a[1] == b[1]


def build_traj(inliers, indexes, row, column):
    traj = [inliers[row][0][column], inliers[row][1][column]]
    indexes[row][column] = 1
    isfound = True
    while isfound and row < len(inliers) - 1:
        isfound = False
        old_coord = inliers[row][1][column]
        row += 1
        for i in range(len(inliers[row][0])):
            if cmp(old_coord, inliers[row][0][i]):
                isfound = True
                indexes[row][i] = 1
                traj.append(inliers[row][1][i])
                column = i
    return traj


# intrinsic = [[525, 0, 319.5], [0, 525, 239.5], [0, 0, 1]]
directory = 'C:/myshit/slam/tests/00_test_slam_input/'

images = []

for filename in os.listdir(directory + 'rgb_with_poses/'):
    image = cv.imread(directory + 'rgb_with_poses/' + filename)
    images.append(image)

# detect = f.FeatureDetector("ORB", [300, 5])
detect = f.FeatureDetector('SUPERPOINT', [1, 0.1, 0.2, 0.1])

orb = cv.ORB_create(fastThreshold=5)
kp1, des1 = detect.compute(images)
kp = [[cv.KeyPoint(kp1[i][0][j], kp1[i][1][j], 1) for j in range(len(kp1[i][0]))] for i in range(len(kp1))]
kpdes = [orb.compute(images[i], kp[i]) for i in range(len(images))]
kp = [item[0] for item in kpdes]
des = [item[1] for item in kpdes]

inliers = []
for i in range(len(des) - 1):
    matches = detect.orb_match(des[i], des[i+1])
    pts1 = np.array([kp[i][matches[j].queryIdx].pt for j in range(len(matches))])
    pts2 = np.array([kp[i+1][matches[j].trainIdx].pt for j in range(len(matches))])
    pts1 = pts1.astype(int)
    pts2 = pts2.astype(int)
    # fig = plt.figure(figsize=(10, 10))
    # fig.add_subplot(2, 2, 1)
    # plt.imshow(cv.drawKeypoints(images[i], kp[i], None, color=(0,255,0)))
    # fig.add_subplot(2, 2, 2)
    # plt.imshow(cv.drawKeypoints(images[i+1], kp[i+1], None, color=(0, 255, 0)))
    # plt.show()
    F, mask = cv.findFundamentalMat(pts1, pts2, cv.FM_RANSAC, 4, 0.99)
    # pts1 = np.array([pts1])
    # pts2 = np.array([pts2])
    # correct_pts1, correct_pts2 = cv.correctMatches(F, pts1, pts2)
    pts1 = np.array(pts1[mask.ravel() == 1])
    pts2 = np.array(pts2[mask.ravel() == 1])

    inliers.append([pts1, pts2])


# E, mask = cv.findEssentialMat(np.array(inliers[0][0]).astype(float), np.array(inliers[0][1]).astype(float), np.array(intrinsic), cv.FM_RANSAC,  0.99, 2)
# R1, R2, t = cv.decomposeEssentialMat(E)
# print(R1)
indexes = [[0 for j in range(len(inliers[i][0]))] for i in range(len(inliers))]
traj = []
i = 0
for i in range(len(indexes)):
    for j in range(len(indexes[i])):
        if indexes[i][j] == 0:
            loc_traj = build_traj(inliers, indexes, i, j)
            traj.append([i] + loc_traj)

traj1 = [tr for tr in traj if len(tr) > 3]
print(inliers)
print(traj1)
''' 
img1 = np.copy(images[0])
    img2 = np.copy(images[1])
    for i in range(len(pts1)):
        img1 = cv.circle(img1, (pts1[i][0], pts1[i][1]), radius=3, color=(0, 0, 255), thickness=-1)
        img2 = cv.circle(img2, (pts2[i][0], pts2[i][1]), radius=3, color=(0, 0, 255), thickness=-1)
    plt.imshow(img1)
    plt.show()
    plt.imshow(img2)
    plt.show()
detect = f.FeatureDetector("ORB", [300, 5])
image = cv.imread("C:/myshit/slam/tests/00_test_slam_input/rgb_with_poses/000004.png")
kp, des = detect.compute([image])
img = np.copy(image)
print(len(kp[0]))
cv.drawKeypoints(image, kp[0], img)
# for i in range(len(kp[0][0])):
#    img = cv.circle(img, (kp[0][0][i], kp[0][1][i]), radius=3, color=(0, 0, 255), thickness=-1)
plt.figure(figsize=(20, 20))
plt.imshow(img)
plt.show()
img = np.copy(images[0])
for i in range(len(inliers[0][0])):
    img = cv.circle(images[0], tuple(inliers[0][0][i]), radius=3, color=(0, 0, 255), thickness=-1)
plt.figure(figsize=(20, 20))
plt.imshow(img)
plt.show()
'''
