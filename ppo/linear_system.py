import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.integrate import odeint
import random
random.seed(42)

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

	def __init__(self, num_switch, switch_int, horizon, xdim, udim, random_start, x0):
		self.num_switch = num_switch
		self.switch_int = switch_int
		self.random_start = random_start
		self.xdim = xdim
		self.udim = udim
		# self.As = [np.eye(xdim) * (0.1 * i-1) for i in range(num_switch)]
		# self.As = [np.eye(xdim) * (0.1*i + 1) for i in range(num_switch-1)] + [np.eye(xdim) * (0.1*num_switch - 1)]
		self.As = [np.eye(xdim) * (0.1*i + 1) for i in range(num_switch-1)] + [np.eye(xdim) * (0.1*num_switch - 1)]
		print("As")
		print(self.As)
		self.B = (1/(xdim * udim) ) * ( (np.arange(xdim * udim) + 1).reshape((xdim, udim)) )
		# self.B = np.random.randn(xdim, udim)
		# print(self.B)
		self.P = np.eye(xdim)
		self.Q = np.eye(udim)
		self.T = horizon
		self.x = None
		self.t = None
		# Generate a random x0 with norm between -num_switch * switch_int and num_switch * switch_int
		# [0, 1] --> (val) * (high - low) + low 
		if self.random_start:
			rand_point = np.random.random(self.xdim)
			rand_norm = np.random.random() * (2 * self.num_switch * self.switch_int) - self.num_switch * self.switch_int
			self.x0 = rand_norm * (rand_point/np.linalg.norm(rand_point) )
		else:
			self.x0 = x0
		self._episode_cost = 0
		self.check_valid_system()
		# print(self.P)
		# print(self.Q)

	def check_valid_system(self):
		"""
		Checks whether the system has a valid configuration.
		"""
		assert np.all(np.linalg.eigvals(self.P) >= 0) and np.allclose(self.P, self.P.T, atol=1e-8)
		assert np.all(np.linalg.eigvals(self.Q) >= 0) and np.allclose(self.Q, self.Q.T, atol=1e-8)

	def reset(self):
		# print("Episode Cost: " + str(self.episode_cost) )
		self._episode_cost = 0
		self.t = 0
		if self.random_start:
			rand_point = np.random.random(self.xdim)
			rand_norm = np.random.random() * (2 * self.num_switch * self.switch_int) - self.num_switch * self.switch_int
			self.x = rand_norm * (rand_point/np.linalg.norm(rand_point) )
		else:
			self.x = self.x0
		return self.x

	def step(self, u):
		assert self.t < self.T, "Horizon Reached."
		# look into this in more detail
		sys_idx = min( int(np.linalg.norm(self.x) // self.switch_int), self.num_switch - 1)
		# print(sys_idx)
		xtp1 = self.As[sys_idx].dot(self.x) + self.B.dot(u)
		cost = xtp1.T.dot(self.P).dot(xtp1) + u.T.dot(self.Q).dot(u)
		self.x = xtp1
		self.t += 1
		self._episode_cost += cost
		# Return negative cost as reward, info=0
		return self.x, -cost, self.t == self.T, 0

	@property
	def action_space(self):
		return np.zeros((self.udim, 0))

	@property
	def observation_space(self):
		return np.zeros((self.xdim, 0))

	@property # Hack randomly chose 5 for now lol
	def action_high(self):
		return 5

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
		# Return negative cost as reward, info=0
		return self.x, -cost, self.t == self.T, 0

	@property
	def action_space(self):
		return np.zeros((self.udim, 0))

	@property
	def observation_space(self):
		return np.zeros((self.xdim, 0))

	@property # Hack randomly chose 5 for now
	def action_high(self):
		return 5

	@property
	def state(self):
		return self.x

	@property
	def episode_cost(self):
		return self._episode_cost


if __name__ == '__main__':
	# env = TimeSwitchedLinearSystemEnv(2, 25, 1, 1, np.array([10]))
	# env = SpaceSwitchedLinearSystemEnv(3, 25, 50, 1, 1, x0=np.array([10]))
	env = SpaceSwitchedLinearSystemEnv(3, 25, 50, 1, 1, False, np.array([100]))
	xs = []
	x = env.reset()
	for i in range(50):
		# x, c, done = env.step(np.array([0.8]))
		x, c, done, info = env.step(np.array([0]))
		xs.append(x)
		# print(c)
	plt.plot(xs)
	plt.show()
