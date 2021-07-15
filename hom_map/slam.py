from homography import Homography as Homography
from mapper import Mapper as Mapper
import cv2 as cv
from scipy.spatial.transform import Rotation as R
import numpy as np
import os

file = open('tests/llff_fern_slam_input/intrinsics.txt', 'r')
line = file.readlines()[1].split()
intr = np.array([[float(line[0]), 0, float(line[2])], [0, float(line[1]), float(line[3])], [0, 0, 1]])
file.close()
file = open('tests/llff_fern_slam_input/known_poses.txt', 'r')
proj_matrices = []

write = open('output.txt', 'w')
write.write('#frame_id tx ty tz qx qy qz qw\n')
for line in file.readlines()[1:4]:
    write.write(line)
    data = line.split()
    index = int(data[0])
    t = np.float32([[data[1], data[2], data[3]]])
    q = np.float32([data[4], data[5], data[6], data[7]])
    r = R.from_quat(q)
    mat = r.as_matrix()
    P = np.concatenate((mat, t.T), axis=1)
    matr = intr.dot(P)
    proj_matrices.append(matr)

directory = 'tests/llff_fern_slam_input/images/'

images = []

for filename in os.listdir(directory):
    image = cv.imread(directory + filename)
    images.append(image)

map = Mapper()
map.update_map(proj_matrices[0], proj_matrices[1], images[0], images[1], intr)
map.update_map(proj_matrices[1], proj_matrices[2], images[1], images[2], intr)

for i in range(4, len(images)):
    pts2d = map.map2d[-1]
    a, rvec, tvec = Homography.estimate(images[i], pts2d[0], pts2d[1], map.map3d[-1], intr)
    r = R.from_rotvec([a for a in rvec.transpose()])
    print(tvec)
    # print(r.as_quat())
    Rot = r.as_matrix()
    P = np.concatenate((Rot[0], np.float32([tvec]).T[0]), axis=1)
    matr = intr.dot(P)
    proj_matrices.append(matr)
    map.update_map(proj_matrices[-2], proj_matrices[-1], images[i - 1], images[i], intr)
    string = str(i) + ' '
    for el in tvec:
        string += str(el[0]) + ' '
    for el in r.as_quat()[0]:
        string += str(el) + ' '
    write.write(string.strip() + '\n')
M = map.map3d
M1 = map.map2d
# print(M)
