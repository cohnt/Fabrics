import jax.numpy as np
from jax import jacfwd

class TransformTreeNode():
	def __init__(self, parent, psi, J, f, M, space_dim):
		# parent is another TransformTreeNode, or None if this node is the root
		# psi is a function mapping from the configuration space of the parent node to this node
		# J is the Jacobian of psi (if set to None, it will be computed with jax)
		# f is the desired force map
		# M is the intertia matrix
		# space_dim is the dimension of the configuration space
		self.parent = parent
		self.children = []
		self.psi = psi
		self.J = jacfwd(psi) if J is None else J
		self.f = f
		self.M = M
		self.x = np.zeros((space_dim,1))
		self.x_dot = np.zeros((space_dim,1))
		self.a = np.zeros((space_dim,1))

		if self.parent is not None:
			self.parent.register_child(self)

	def register_child(self, child):
		self.children.append(child)

	def forward_pass(self):
		self.pushforward()
		for child in self.children:
			child.forward_pass()

	def backward_pass(self):
		for child in self.children:
			child.backward_pass()
		self.pullback()

	def pushforward(self):
		for child in self.children:
			child.x = child.psi(self.x)
			child.x_dot = self.J(self.x) @ self.x_dot

	def pullback(self):
		self.f = np.zeros(self.x.shape[0])
		self.M = np.zeros(self.x.shape[0], self.x.shape[0])

		x = self.x
		x_dot = self.x_dot
		for child in self.children:
			self.f = self.f + child.J(x).T @ (child.f(x, x_dot) - (child.M(x, x_dot) @ jacfwd(child.J)(x) @ x_dot))
			self.M = self.M + child.J(x).T @ child.M(x, x_dot) @ child.J(x)

	def resolve(self):
		self.a = np.linalg.pinv(self.M) @ self.f