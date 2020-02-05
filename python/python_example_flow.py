
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

import open3d as o3d

import pyinterface
import os, sys

seq = 1
left_dir = "/home/aljosa/data/kitti_tracking/training/image_02/%04d"%seq
right_dir = "/home/aljosa/data/kitti_tracking/training/image_03/%04d"%seq
#print (images[0].dtype)
# Show images
#for img in images:
#	plt.imshow(img)
#	plt.show()

# Call our VO fnc
focal = 721.537700
cu = 609.559300
cv = 172.854000
baseline = 0.532719

T_acc = np.identity(4)


p_acc = np.array([0.0, 0.0, 0.0, 1.0])
p_0 = np.array([0.0, 0.0, 0.0, 1.0])

poses = p_acc

pts_acc = None

vo = pyinterface.VOEstimator()
for frame in range(0, 350):
	print ("--- fr %d ---"%frame)
	l1 = os.path.join(left_dir, "%06d.png"%frame)
	r1 = os.path.join(right_dir, "%06d.png"%frame)

	# Read images
	images = [mpimg.imread(l1),  mpimg.imread(r1)]

	# Conv to grayscale
	images = [np.mean(x, -1) for x in images]

	if frame == 0:
		vo.init(images[0], images[1], focal, cu, cv, baseline, True)
	else:
		T = vo.compute_pose(images[0], images[1])

		F = vo.compute_flow(images[0], images[1], 0.1, 40.0)
		print (F.shape)

		# Update accumulated transf and compute current pose
		T_acc = T_acc.dot(np.linalg.inv(T))
		p_acc = T_acc.dot(p_0)
		poses = np.vstack((poses, p_acc))

		# Accumulate scene flow
		if pts_acc is None:
			pts_acc = F
		else:
			# pts_acc[:, 0:3] = T_acc.dot(pts_acc[0:3])
			# pts_acc[:, 3:6] = T_acc.dot(pts_acc[3:6])

			# print ("==== 2 ===== ")
			# print (F[:, 0:3].shape)
			# print (np.ones((F.shape[0], 1)).shape)

			# o3d_pc = o3d.geometry.PointCloud()
			# o3d_pc.points = o3d.utility.Vector3dVector(F[:, 0:3])
			# objs = []
			# objs.append(o3d_pc)
			# o3d.visualization.draw_geometries(objs)

			conmat = np.hstack((F[:, 0:3], np.ones((F.shape[0], 1))))
			pts_tmp = T_acc.dot(np.transpose(conmat))
			F[:, 0:3] = np.transpose(pts_tmp)[:, 0:3]

			#T_acc.dot(pts_acc[0:3])
			#T_acc.dot(pts_acc[3:6])
			pts_acc = np.vstack((pts_acc, F))

# # Show the poses
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.scatter3D(poses[:, 0], poses[:, 1], poses[:, 2])
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.show()

# # Show the accumulated points
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.scatter3D(pts_acc[:, 0], pts_acc[:, 1], pts_acc[:, 2], c="red")
# #ax.scatter3D(pts_acc[:, 3], pts_acc[:, 4], pts_acc[:, 5], c="green")
# plt.xlabel("X")
# plt.ylabel("Y")
# #plt.zlabel("Z")
# #ax.set_yticks(np.arange(0, 1.2, 0.2))
# #ax.set_xticks(np.asarray([25, 100, 200, 300, 500, 700]))
# # ax.set_ylim([-3.0, 3.0])
# # ax.set_xlim([-10.0, 10.0])	
# # ax.set_zlim([0.0, 50.0])	
# ax.view_init(elev=-72, azim=-91)
# #plt.axis('equal')
# plt.legend()
# plt.show()	

# Point cloud
o3d_pc = o3d.geometry.PointCloud()
o3d_pc.points = o3d.utility.Vector3dVector(pts_acc[:, 0:3])
objs = []
objs.append(o3d_pc)
o3d.visualization.draw_geometries(objs)