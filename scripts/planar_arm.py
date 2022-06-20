#######################
# Fix import issues   #
import os             #
import sys            #
import traceback      #
sys.path.append(".")  #
#######################

import jax.numpy as np
from jax import jacfwd, jacrev, grad
from jax.experimental.host_callback import id_print
from jax.lax import cond
import matplotlib.pyplot as plt
import time

from src.transform_tree import TransformTreeNode
import src.planar_arm_visualization as visualization
from src.util import log1pexp

from jax.config import config
config.update("jax_debug_nans", True)
# config.update("jax_enable_x64", True)

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

goal_pos = np.array([-2., 2.]) # Can manually specify an initial goal if desired
num_ik_steps = 0
end_effector_pos = None
end_effector_link_idx = num_links - 1
end_effector_path = []
end_effector_path_downsample = 25

obs_origins = np.array([
	[2, 1.5],
	[3, -2],
	[-1, 1]
], dtype=float)
obs_rs = [1., 0.75, 0.25]

fabric_dt = 0.01
dist_thresh = 0.01
vel_thresh = 0.01
fabric_running = False

joint_range_frac = 0.5 # 1. is -pi to pi
nominal_configuration = np.zeros(num_joints)
nominal_l2 = True # If True, penalize deviation from the nominal config with l2 norm, if False, l1 norm.

fig_num = 0

largest_angle = 0
largest_angle_idx = 0

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

def draw_obstacles(ax):
	for obs_origin, obs_r in zip(obs_origins, obs_rs):
		obs_element = plt.Circle(tuple(obs_origin), obs_r, color='black')
		ax.add_patch(obs_element)

def update_display(ax):
	global end_effector_pos, end_effector_path, fig_num
	visualization.reset_display(ax, axes_limits)

	draw_obstacles(ax)

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

	if fabric_running:
		plt.savefig("fig%04d.png" % fig_num)
		fig_num += 1
	# else:
	# 	plt.draw()

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
			root = create_fabric()
			while True:
				times = []
				for _ in range(100):
					# print(joint_angles)
					t0 = time.time()
					joint_accel = root.solve(joint_angles, joints_vels)
					times.append(time.time() - t0)
					joint_angles = joint_angles + (joints_vels * fabric_dt)
					global largest_angle, largest_angle_idx
					if np.max(np.abs(joint_angles[1:])) > largest_angle:
						largest_angle = np.max(np.abs(joint_angles[1:]))
						largest_angle_idx = np.argmax(np.abs(joint_angles[1:]))+1
					joints_vels = joints_vels + (joint_accel * fabric_dt)
					record_endeffector_pose(joint_angles)
					if np.linalg.norm(get_endeffector_pose(joint_angles) - goal_pos) < dist_thresh and np.linalg.norm(joints_vels) < vel_thresh: break
				t0 = time.time()
				update_display(ax)
				print("Drawing time: %f ms" % ((time.time() - t0)*1000))
				# print(np.linalg.norm(get_endeffector_pose(joint_angles) - goal_pos))
				# print(np.linalg.norm(joints_vels))
				print("Average fabric update rate: %f ms" % (np.mean(np.array(times))*1000))
				# print("Largest angle: %f \t Index: %d \t Boundary: %f" % (largest_angle, largest_angle_idx, joint_range_frac * np.pi))
				if np.linalg.norm(get_endeffector_pose(joint_angles) - goal_pos) < dist_thresh and np.linalg.norm(joints_vels) < vel_thresh:
					print("Goal reached!")
					fabric_running = False
					break
				if plt.waitforbuttonpress(1):
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
	psi = lambda theta : k * (np.linalg.norm(theta) + (1 / alpha_psi) * log1pexp(-2 * alpha_psi * np.linalg.norm(theta)))
	dx = jacfwd(psi)(x)
	x_dot_dot = -1 * dx - beta * x_dot
	M = (m_up - m_down) * np.exp(-1 * (alpha_m * np.linalg.norm(x))**2) * np.eye(2) + m_down * np.eye(2)
	return (M, x_dot_dot)

def upper_joint_limits_task_map_template(theta, joint_idx):
	upper_limit = joint_range_frac * np.pi
	# id_print(joint_idx)
	# id_print(upper_limit - theta[joint_idx])
	return np.array([upper_limit - theta[joint_idx]])

def lower_joint_limits_task_map_template(theta, joint_idx):
	lower_limit = -joint_range_frac * np.pi
	# id_print(joint_idx)
	# id_print(theta[joint_idx] - lower_limit)
	return np.array([theta[joint_idx] - lower_limit])

def joint_limits_fabric(x, x_dot):
	# id_print(x)
	a1, a2, a3, a4 = 0.4, 0.2, 20., 5.
	l = 0.25
	s = (x_dot < 0).astype(float)
	M = np.array([s * l / x])
	psi = lambda theta : (a1 / theta**2) + a2 * log1pexp(-a3 * (theta - a4))
	dx = jacrev(psi)(x).reshape(1) # For some reason, dx is a 1x1 array. This is needed to fix the shape.
	x_dot_dot = -s * np.linalg.norm(x_dot)**2 * dx
	return (M, x_dot_dot)

def nominal_configuration_task_map_template_l1(theta, joint_idx):
	#
	return np.array([theta[joint_idx] - nominal_configuration[joint_idx]])

def nominal_configuration_task_map_l2(theta):
	#
	return theta - nominal_configuration

def nominal_configuration_fabric(x, x_dot):
	lambda_dc = 0.0025
	k = 50.
	alpha_psi = 10.
	beta = 2.5
	eps = 0.0000001
	psi = lambda theta : k * (np.linalg.norm(theta) + (1 / alpha_psi) * log1pexp(-2 * alpha_psi * np.linalg.norm(theta)))
	dx = cond(np.linalg.norm(x) < eps,
	          lambda x : np.zeros(x.shape),
	          lambda x : jacfwd(psi)(x).flatten(),
	          x)
	x_dot_dot = -1 * dx - beta * x_dot
	M = lambda_dc * np.eye(x.shape[0])
	return (M, x_dot_dot)

def obstacle_avoidance_task_map_template(theta, joint_idx):
	link_endpoint_local = np.array([arm_link_lengths[joint_idx],0])
	link_endpoint = apply_fk(link_endpoint_local, joint_idx, theta)
	return np.array([np.linalg.norm(link_endpoint - obs_origin) / obs_r - 1.0 for obs_origin, obs_r in zip(obs_origins, obs_rs)])

def obstacle_avoidance_fabric(x, x_dot):
	k_beta = 0.01
	alpha_beta = 1.
	s = (x_dot < 0).astype(float)
	M = np.diag(k_beta * np.divide(s, x**2))
	psi = lambda theta : np.divide(alpha_beta, (2 * (theta**8)))
	dx = jacfwd(psi)(x)
	x_dot_dot = -s * x_dot**2 @ dx
	return (M, x_dot_dot)

def create_fabric():
	root = TransformTreeNode(parent=None, psi=None, fabric=None)
	TransformTreeNode(parent=root, psi=reach_task_map, fabric=reach_fabric)
	# We skip the very first joint, since we're assuming it can rotate through the full 360 degrees
	for joint_idx in range(1, num_joints):
		# Python is weird about binding values to lambdas. This works, because default initializations are
		# computed at creation time!
		# See: https://stackoverflow.com/questions/10452770/python-lambdas-binding-to-local-values
		upper_task_map = lambda theta, joint_idx=joint_idx : upper_joint_limits_task_map_template(theta, joint_idx)
		lower_task_map = lambda theta, joint_idx=joint_idx : lower_joint_limits_task_map_template(theta, joint_idx)
		TransformTreeNode(parent=root, psi=upper_task_map, fabric=joint_limits_fabric)
		TransformTreeNode(parent=root, psi=lower_task_map, fabric=joint_limits_fabric)
		if not nominal_l2:
			nominal_task_map = lambda theta, joint_idx=joint_idx : nominal_configuration_task_map_template_l1(theta, joint_idx)
			TransformTreeNode(parent=root, psi=nominal_task_map, fabric=nominal_configuration_fabric)
		if len(obs_origins) > 0:
			obstacle_avoidance_task_map = lambda theta, joint_idx=joint_idx : obstacle_avoidance_task_map_template(theta, joint_idx)
			TransformTreeNode(parent=root, psi=obstacle_avoidance_task_map, fabric=obstacle_avoidance_fabric)
	if nominal_l2:
		TransformTreeNode(parent=root, psi=nominal_configuration_task_map_l2, fabric=nominal_configuration_fabric)
	return root

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