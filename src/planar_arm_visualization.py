import numpy as np
import matplotlib.pyplot as plt

def make_display(limits):
	# Creates a matplotlib figure and square axes, with the given
	# axes limits. limits should be a tuple ((xmin, xmax), (ymin, ymax)).
	# Returns a figure and axes object.

	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1)
	ax.set_aspect('equal', adjustable='box') # Keep an equal aspect ratio
	ax.set_xlim(limits[0])
	ax.set_ylim(limits[1])
	return fig, ax

def reset_display(ax, limits):
	# Resets a given ax object to blank.
	ax.cla()
	ax.set_aspect('equal', adjustable='box') # Keep an equal aspect ratio
	ax.set_xlim(limits[0])
	ax.set_ylim(limits[1])

def draw_link(ax, p1, p2, width, link_color="black"):
	# Draws a link from point p1 to point p2 as a rectangle with specife width.
	# p1 and p2 should be numpy arrays of length 2, like [x,y].
	# width should be a positive scalar.

	if (p1 == p2).all():
		return

	tangent_axis = p2 - p1
	tangent_axis = tangent_axis / np.linalg.norm(tangent_axis)
	normal_axis = np.array([-1 * tangent_axis[1], tangent_axis[0]])

	corners = np.zeros((4,2))
	corners[0] = p1 - (normal_axis * width / 2)
	corners[1] = p1 + (normal_axis * width / 2)
	corners[2] = p2 + (normal_axis * width / 2)
	corners[3] = p2 - (normal_axis * width / 2)

	# ax.scatter(corners[:,0], corners[:,1], color=link_color)
	ax.plot(corners[:,0], corners[:,1], color=link_color)
	ax.plot(corners[[0,-1],0], corners[[0,-1],1], color=link_color)

def draw_joint(ax, p, radius, joint_color="black"):
	# Draws a joint at point p, as a circle width specified radius.
	# p should be a numpy array of length 2, like [x,y].
	# radius should be a positive scalar.

	circle = plt.Circle(p, radius, color=joint_color, fill=False)
	ax.add_patch(circle)

def draw_coordinate_frame(ax, p, x_axis, axis_len):
	# Draws a coordinate frame at point p, with angle theta.
	# p should be a numpy array of length 2, like [x,y].
	# x_axis should be a vector collinear to the x axis.
	# axis_len is the length of the coordinate axes.

	if (x_axis == np.zeros_like(x_axis)).all():
		return

	x_axis = x_axis / np.linalg.norm(x_axis) * axis_len
	y_axis = np.array([-1 * x_axis[1], x_axis[0]])

	ax.arrow(p[0], p[1], x_axis[0], x_axis[1], color="red", head_width=0.1*axis_len)
	ax.arrow(p[0], p[1], y_axis[0], y_axis[1], color="green", head_width=0.1*axis_len)

def maximize_plt_fig(fig):
	# Reference:
	# https://stackoverflow.com/questions/12439588/how-to-maximize-a-plt-show-window-using-python/18824814#18824814
	# https://stackoverflow.com/questions/13065753/obtaining-the-figure-manager-via-the-oo-interface-in-matplotlib
	# Works for the Qt backend. Call before plt.show() or plt.draw().
	fig.canvas.manager.window.showMaximized()

if __name__ == "__main__":
	fig, ax = make_display(((-2,2),(-1,1)))
	draw_link(ax, np.array([0,0]), np.array([0,0.5]), 0.1)
	draw_joint(ax, np.array([0,0.5]), 0.1)
	draw_coordinate_frame(ax, np.array([0,0]), 0, 0.2)
	maximize_plt_fig(fig)
	plt.show()