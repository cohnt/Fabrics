#######################
# Fix import issues   #
import os             #
import sys            #
import traceback      #
sys.path.append(".")  #
#######################

import jax.numpy as np
from jax import jacfwd, grad, jit
import matplotlib.pyplot as plt

from src.transform_tree import TransformTreeNode

# Directly trying to replicate Alphonsus' point mass code in Python

# from jax.config import config
# config.update("jax_debug_nans", True)

root = TransformTreeNode(parent=None, psi=None, fabric=None)

def attractor_task_map(theta):
	return theta - goal

def attractor_fabric(x, x_dot):
	k = 10.
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

attractor_node = TransformTreeNode(parent=root, psi=attractor_task_map, fabric=attractor_fabric)

def repeller_task_map(theta):
	return np.array([np.linalg.norm(theta - obs_origin) / obs_r - 1.0])

def repeller_fabric(x, x_dot):
	k_beta = 20.
	alpha_beta = 1.
	s = (x_dot < 0).astype(float)
	M = np.diag(k_beta * np.divide(s, x**2))
	psi = lambda theta : np.divide(alpha_beta, (2 * (theta**8)))
	dx = jacfwd(psi)(x).reshape(1) # For some reason, dx is a 1x1 array. This is needed to fix the shape.
	x_dot_dot = -s * x_dot**2 * dx
	return (M, x_dot_dot)

repeller_node = TransformTreeNode(parent=root, psi=repeller_task_map, fabric=repeller_fabric)

inits = np.array([
	[5.5, -1],
	[5.5, -0.75],
	[5.5, -0.5],
	[5.5, -0.25],
	[5.5, -0.01],
	[5.5, 0.01],
	[5.5, 0.25],
	[5.5, 0.5],
	[5.5, 0.75],
	[5.5, 1]
])

dist_thresh = 0.01
vel_thresh = 0.01
goal = np.array([-5, 0], dtype=float)
obs_origin = np.array([0, 0], dtype=float)
obs_r = 1.
dt = 0.001

fig, ax = plt.subplots()
obs_element = plt.Circle(tuple(obs_origin), obs_r, color='black')
ax.add_patch(obs_element)

ax.set_aspect("equal")
plt.draw()
plt.pause(0.001)

matplotlib_downsample = 250

for x in inits:
	x_dot = np.array([0, 0], dtype=float)
	xs = []
	for i in range(100000):
		x_dot_dot = root.solve(x, x_dot)
		x = x + (x_dot * dt)
		x_dot = x_dot + (x_dot_dot * dt)
		xs.append(x)

		if np.linalg.norm(x - goal) < dist_thresh and np.linalg.norm(x_dot) < vel_thresh:
			break
	xs = np.asarray(xs)
	
	# ax.scatter(xs[::matplotlib_downsample,0], xs[::matplotlib_downsample,1], c=np.linspace(0, 1, len(xs[::matplotlib_downsample])), cmap=plt.get_cmap("viridis"))
	ax.plot(xs[::matplotlib_downsample,0], xs[::matplotlib_downsample,1])
	ax.set_aspect("equal")
	plt.draw()
	plt.pause(0.001)

ax.set_aspect("equal")
plt.show()