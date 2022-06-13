import autograd.numpy as np
from autograd import grad

class TransformTreeNode():
	def __init__(self, parent, phi, J, f, M, space_dim):
		# parent is another TransformTreeNode, or None if this node is the root
		# phi is a function mapping from the configuration space of the parent node to this node
		# J is the Jacobian of phi. If set to None, it will be computed using autograd
		# f is the desired force map
		# M is the intertia matrix
		# space_dim is the dimension of the configuration space
		self.parent = parent
		self.children = []
		self.phi = phi
		self.J = J
		self.f = f
		self.M = M
		self.x = np.zeros((space_dim,1))
		self.x_dot = np.zeros((space_dim,1))

		if self.parent is not None:
			self.parent.register_child(self)

	def register_child(self, child):
		self.children.append(child)

	def forward_pass(self):
		pass

	def backward_pass(self):
		pass

	def pushforward(self):
		pass

	def pullback(self):
		pass

	def resolve(self):
		pass