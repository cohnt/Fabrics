import jax.numpy as np
from jax import jacfwd, grad, jit, jvp
from functools import partial

class TransformTreeNode():
	def __init__(self, parent, psi, fabric, space_dim):
		# parent is another TransformTreeNode, or None if this node is the root
		# psi is a function mapping from the configuration space of the parent node to this node
		# fabric is a function which maps x and x_dot to M and x_dot_dot (or None if not a leaf node)
		# space_dim is the dimension of the configuration space
		self.parent = parent
		self.children = []
		self.psi = psi
		self.x = np.zeros(space_dim)
		self.x_dot = np.zeros(space_dim)
		self.M = np.zeros((space_dim, space_dim))
		self.x_dot_dot = np.zeros(space_dim)
		self.c = None
		self.J = None
		
		if fabric is None:
			self.is_leaf = False
			self.fabric = lambda : None
		else:
			self.is_leaf = True
			self.fabric = fabric

		if self.parent is not None:
			self.parent.register_child(self)

	def register_child(self, child):
		self.children.append(child)

	def forward_pass(self):
		if self.is_leaf:
			self.M, self.x_dot_dot = self.fabric(self.x, self.x_dot)
		else:
			self.pushforward()
			for child in self.children:
				child.forward_pass()

	def backward_pass(self):
		if self.is_leaf:
			self.c = jvp(lambda s : jvp(self.psi, (s,), (self.parent.x_dot,))[1], (self.parent.x,), (self.parent.x_dot,))[1]
			self.J = jacfwd(self.psi)(self.parent.x)
		else:
			for child in self.children:
				child.backward_pass()
			self.pullback()

	def pushforward(self):
		for child in self.children:
			child.x = child.psi(self.x)
			child.x_dot = jvp(child.psi, (self.x,), (self.x_dot,))[1]

	def pullback(self):
		self.M = np.sum(np.array([
			child.J.T @ child.M @ child.J for child in self.children
		]), axis=0)
		self.x_dot_dot = np.sum(np.array([
			child.J.T @ child.M @ (child.x_dot_dot - child.c) for child in self.children
		]), axis=0)

	def resolve(self):
		return np.linalg.pinv(self.M) @ self.x_dot_dot

	@partial(jit, static_argnums=(0,))
	def solve(self, x, x_dot):
		self.x = x
		self.x_dot = x_dot
		self.forward_pass()
		self.backward_pass()
		return self.resolve()
