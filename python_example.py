
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

import pyinterface
import os, sys

seq = 1
left_dir = "/home/aljosa/data/kitti_tracking/training/image_02/%04d"%seq
right_dir = "/home/aljosa/data/kitti_tracking/training/image_03/%04d"%seq

# KITTI params
focal = 721.537700
cu = 609.559300
cv = 172.854000
baseline = 0.532719

T_acc = np.identity(4)

p_acc = np.array([0.0, 0.0, 0.0, 1.0])
p_0 = np.array([0.0, 0.0, 0.0, 1.0])

poses = p_acc

for frame in range(0, 300):
	l1 = os.path.join(left_dir, "%06d.png"%frame)
	l2 = os.path.join(left_dir, "%06d.png"%(frame+1))

	r1 = os.path.join(right_dir, "%06d.png"%frame)
	r2 = os.path.join(right_dir, "%06d.png"%(frame+1))

	# Read images
	images = [mpimg.imread(l1), mpimg.imread(l2), mpimg.imread(r1), mpimg.imread(r2)]

	# Conv to grayscale
	images = [np.mean(x, -1) for x in images]

	# Run VO
	T = pyinterface.compute_vo(images[0], images[2], images[1], images[3], focal, cu, cv, baseline)

	# Update accumulated transf and compute current pose
	T_acc = T_acc.dot(np.linalg.inv(T))
	p_acc = T_acc.dot(p_0)
	poses = np.vstack((poses, p_acc))

# Show the poses
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter3D(poses[:, 0], poses[:, 1], poses[:, 2])
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
