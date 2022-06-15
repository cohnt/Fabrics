#######################
# Fix import issues   #
import os             #
import sys            #
import traceback      #
sys.path.append(".")  #
#######################

import jax.numpy as np
from jax import jacfwd
import matplotlib.pyplot as plt
import time

from src.transform_tree import TransformTreeNode
import src.planar_arm_visualization as visualization

arm_link_lengths = np.append(np.zeros(1), np.ones(10))*0.4
joint_angles = np.zeros(len(arm_link_lengths) - 1)
joints_vels = np.zeros(shape=joint_angles.shape)
num_joints = len(joint_angles)
num_links = len(arm_link_lengths)
selected_joint = 0
keypress_angle_change = np.pi/128

axes_limits = ((-4.5,4.5),(-4.5,4.5))
highlighted_joint_color = "blue"
joint_disp_radius = 0.1
link_disp_width = 0.1
fk_translations = [np.array([l,0]) for l in arm_link_lengths]

goal_pos = None
num_ik_steps = 0
end_effector_pos = None
end_effector_link_idx = num_links - 1
end_effector_path = []
end_effector_path_downsample = 25

fabric_dt = 0.01
dist_thresh = 0.01
vel_thresh = 0.01
fabric_running = False

def make_rotation_matrix(theta):
	mat = np.array([
		[np.cos(theta), -np.sin(theta)],
		[np.sin(theta), np.cos(theta)]
	])
	return mat

def apply_transformation(rotation, translation, point):
	# rotation is a 2x2 rotation matrix
	# translation is a row vector
	# point is a row vector
	return np.matmul(point, rotation.T) + translation

def apply_fk(point, frame_idx, joint_angles):
	for i in reversed(range(0, frame_idx)):
		point = apply_transformation(make_rotation_matrix(joint_angles[i]), fk_translations[i], point)
	return point

def update_display(ax):
	global end_effector_pos, end_effector_path
	visualization.reset_display(ax, axes_limits)

	for i in range(0, num_links):
		point1 = apply_fk(np.array([0,0]), i, joint_angles)
		point2 = apply_fk(np.array([arm_link_lengths[i],0]), i, joint_angles)
		visualization.draw_link(ax, point1, point2, link_disp_width)
		
		if i >= 1:
			c = highlighted_joint_color if selected_joint == i-1 else "black"
			visualization.draw_joint(ax, point1, joint_disp_radius, joint_color=c)


		if goal_pos is not None:
			ax.scatter([goal_pos[0]], [goal_pos[1]], c="blue")

	if len(end_effector_path) > 0:
		arr = np.array(end_effector_path)[::end_effector_path_downsample]
		ax.plot(arr[:,0], arr[:,1], color="blue")

	plt.draw()

def handle_keypress(event):
	global selected_joint, joint_angles, joints_vels, fabric_running
	if event.key == "left":
		joint_angles = joint_angles.at[selected_joint].set(joint_angles[selected_joint] + keypress_angle_change)
		record_endeffector_pose(joint_angles)
		# print("Joint %d set to %f degrees" % (selected_joint, joint_angles[selected_joint] * 180 / np.pi))
		update_display(ax)
	elif event.key == "right":
		joint_angles = joint_angles.at[selected_joint].set(joint_angles[selected_joint] - keypress_angle_change)
		record_endeffector_pose(joint_angles)
		# print("Joint %d set to %f degrees" % (selected_joint, joint_angles[selected_joint] * 180 / np.pi))
		update_display(ax)
	elif event.key == "up":
		selected_joint = (selected_joint + 1) % num_joints
		# print("Selected joint %d" % selected_joint)
		update_display(ax)
	elif event.key == "down":
		selected_joint = (selected_joint - 1) % num_joints
		# print("Selected joint %d" % selected_joint)
		update_display(ax)
	elif event.key == "r":
		if goal_pos is None:
			print("Error: Choose a goal pose before running the fabric!")
			return
		elif fabric_running:
			print("Error: fabric is already running!")
			return
		else:
			fabric_running = True
			print("Running fabric until convergence or a keypress is received.")
			root = TransformTreeNode(parent=None, psi=None, fabric=None)
			reach_node = TransformTreeNode(parent=root, psi=reach_task_map, fabric=reach_fabric)
			while True:
				times = []
				for _ in range(100):
					t0 = time.time()
					joint_accel = root.solve(joint_angles, joints_vels)
					times.append(time.time() - t0)
					joint_angles = joint_angles + (joints_vels * fabric_dt)
					joints_vels = joints_vels + (joint_accel * fabric_dt)
					record_endeffector_pose(joint_angles)
					if np.linalg.norm(get_endeffector_pose(joint_angles) - goal_pos) < dist_thresh and np.linalg.norm(joints_vels) < vel_thresh: break
				t0 = time.time()
				update_display(ax)
				print("Drawing time: %f ms" % ((time.time() - t0)*1000))
				# print(np.linalg.norm(get_endeffector_pose(joint_angles) - goal_pos))
				# print(np.linalg.norm(joints_vels))
				print("Average fabric update rate: %f ms" % (np.mean(np.array(times))*1000))
				if np.linalg.norm(get_endeffector_pose(joint_angles) - goal_pos) < dist_thresh and np.linalg.norm(joints_vels) < vel_thresh:
					print("Goal reached!")
					fabric_running = False
					break
				if plt.waitforbuttonpress(0.1):
					print("Goal not reached!")
					fabric_running = False
					break

def handle_click(event):
	global goal_pos, num_ik_steps, end_effector_path
	if event.xdata is None or event.ydata is None:
		return
	else:
		goal_pos = np.array([event.xdata, event.ydata])
		end_effector_path = []
		print("New goal position: (%f, %f)" % (event.xdata, event.ydata))
		num_ik_steps = 0
		update_display(ax)

def get_endeffector_pose(joint_angles):
	end_effector_pos_local = np.array([arm_link_lengths[end_effector_link_idx],0])
	end_effector_pos = apply_fk(end_effector_pos_local, end_effector_link_idx, joint_angles)
	return end_effector_pos

def record_endeffector_pose(joint_angles):
	end_effector_pos = get_endeffector_pose(joint_angles)
	end_effector_path.append(end_effector_pos)

def reach_task_map(theta):
	#
	return get_endeffector_pose(theta) - goal_pos

def reach_fabric(x, x_dot):
	k = 5.
	alpha_psi = 10.
	beta = 10.5
	m_up = 2.0
	m_down = 0.2
	alpha_m = 0.75
	psi = lambda theta : k * (np.linalg.norm(theta) + (1 / alpha_psi) * np.log(1 + np.exp(-2 * alpha_psi * np.linalg.norm(theta))))
	dx = jacfwd(psi)(x)
	x_dot_dot = -1 * dx - beta * x_dot
	M = (m_up - m_down) * np.exp(-1 * (alpha_m * np.linalg.norm(x))**2) * np.eye(2) + m_down * np.eye(2)
	return (M, x_dot_dot)

fig, ax = visualization.make_display(axes_limits)
visualization.maximize_plt_fig(fig)
fig.canvas.mpl_connect("key_press_event", handle_keypress)
fig.canvas.mpl_connect("button_press_event", handle_click)
update_display(ax)

print("Instructions:")
print("\tLeft, right arrows control the selected joint.")
print("\tUp, down arrows change which joint is selected.")
print("\t\"r\" runs the optimization fabric.")
print("\t\"d\" toggles displaying the path the arm has taken.")
print("\t\"q\" ends the program.")
print("")
print("To run the fabric, you need to select a goal position for the arm. Do so by clicking anywhere on the screen.")
print("")

plt.show()