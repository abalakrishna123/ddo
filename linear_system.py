import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.integrate import odeint

class Env(object):

	def step(self, u):
		raise NotImplementedError

	def reset(self, u):
		raise NotImplementedError

	@property
	def state(self):
		raise NotImplementedError

class SpaceSwitchedLinearSystemEnv(Env):
	"""
	Fully observable Discrete Time, Space-Switched Linear Dynamical System.
	Want to navigate to the origin.
	"""

	def __init__(self, num_switch, switch_int, horizon, xdim, udim, x0):
		self.num_switch = num_switch
		self.switch_int = switch_int
		self.xdim = xdim
		self.udim = udim
		self.As = [np.eye(xdim) * (0.1 * i+1) for i in range(num_switch)]
		self.B = np.random.randn(xdim, udim)
		self.P = np.eye(xdim)
		self.Q = np.eye(udim)
		self.T = horizon
		self.x = None
		self.t = None
		self.x0 = x0
		self._episode_cost = 0
		self.check_valid_system()
		print(self.P)
		print(self.Q)

	def check_valid_system(self):
		"""
		Checks whether the system has a valid configuration.
		"""
		assert np.all(np.linalg.eigvals(self.P) >= 0) and np.allclose(self.P, self.P.T, atol=1e-8)
		assert np.all(np.linalg.eigvals(self.Q) >= 0) and np.allclose(self.Q, self.Q.T, atol=1e-8)

	def reset(self):
		self._episode_cost = 0
		self.t = 0
		self.x = self.x0
		return self.x

	def step(self, u):
		assert self.t < self.T, "Horizon Reached."
		# look into this in more detail
		sys_idx = min( int(np.linalg.norm(self.x) // self.switch_int), self.num_switch - 1)
		print(sys_idx)
		xtp1 = self.As[sys_idx].dot(self.x) + self.B.dot(u)
		cost = xtp1.T.dot(self.P).dot(xtp1) + u.T.dot(self.Q).dot(u)
		self.x = xtp1
		self.t += 1
		self._episode_cost += cost
		return self.x, cost, self.t == self.T

	@property
	def state(self):
		return self.x

	@property
	def episode_cost(self):
		return self._episode_cost


class TimeSwitchedLinearSystemEnv(Env):
	"""
	Fully observable Discrete Time, Time-Switched Linear Dynamical System.
	Probably better to test with Time-Invariant Dynamics. Want to navigate to the origin.
	"""

	def __init__(self, num_switch, switch_int, xdim, udim, x0):
		self.num_switch = num_switch
		self.switch_int = switch_int
		self.xdim = xdim
		self.udim = udim
		self.As = [np.eye(xdim) * (0.1 * i+1) for i in range(num_switch)]
		self.B = np.random.randn(xdim, udim)
		self.P = np.eye(xdim)
		self.Q = np.eye(udim)
		self.T = self.num_switch * self.switch_int
		self.x0 = x0
		self.x = None
		self.t = None
		self._episode_cost = 0
		self.check_valid_system()

	def check_valid_system(self):
		"""
		Checks whether the system has a valid configuration.
		"""
		assert np.all(np.linalg.eigvals(self.P) >= 0) and np.allclose(self.P, self.P.T, atol=1e-8)
		assert np.all(np.linalg.eigvals(self.Q) >= 0) and np.allclose(self.Q, self.Q.T, atol=1e-8)

	def reset(self):
		self._episode_cost = 0
		self.t = 0
		self.x = self.x0
		return self.x

	def step(self, u):
		assert self.t < self.T, "Horizon Reached."
		sys_idx = self.t // self.switch_int
		xtp1 = self.As[sys_idx].dot(self.x) + self.B.dot(u)
		cost = xtp1.T.dot(self.P).dot(xtp1) + u.T.dot(self.Q).dot(u)
		self.x = xtp1
		self.t += 1
		self._episode_cost += cost
		return self.x, cost, self.t == self.T

	@property
	def state(self):
		return self.x

	@property
	def episode_cost(self):
		return self._episode_cost


if __name__ == '__main__':
	# env = TimeSwitchedLinearSystemEnv(2, 25, 1, 1, np.array([10]))
	env = SpaceSwitchedLinearSystemEnv(3, 25, 50, 1, 1, np.array([10]))
	xs = []
	x = env.reset()
	for i in range(50):
		print(x)
		# x, c, done = env.step(np.array([0.8]))
		x, c, done = env.step(np.random.randn(1))
		xs.append(x)
	print(xs)
	plt.plot(xs)
	plt.show()
