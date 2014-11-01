import sys
import unittest

from numpy import asarray, any, zeros, abs, max
from numpy.random import dirichlet, binomial, randint, permutation
from scipy.stats import ks_2samp
from trlda.utils import random_select, sample_dirichlet, polygamma
from trlda.utils import load_users, load_users_as_dict
from tempfile import mkstemp

class ToolsTest(unittest.TestCase):
	def test_random_select(self):
		# select all elements
		self.assertTrue(set(random_select(8, 8)) == set(range(8)))

		idx0 = random_select(11, 121)
		idx1 = random_select(11, 121)

		# make sure function returned 11 numbers between 0 and 121
		for i in idx0:
			self.assertGreaterEqual(i, 0)
			self.assertLessEqual(i, 121)
		self.assertTrue(len(idx0), 11)

		# make sure results are not the same
		self.assertTrue(any(asarray(idx0) != asarray(idx1)))

		# n should be larger than k
		self.assertRaises(Exception, random_select, 10, 4)



	def test_polygamma(self):
		# example values of the polygamma function
		values = {
			(0, .1): -10.423754940411,
			(0, 1.): -0.5772156649015329,
			(0, 120.): 4.7833192891185,
			(1, .01): 10001.6212135283,
			(1, .1): 101.433299150792758817215450106,
			(1, .4): 7.275356590529597,
			(1, 11.): 0.09516633568168575,
			(2, 14.): -0.005479465690312488}

		for (n, x), y in values.items():
			self.assertAlmostEqual(y, polygamma(n, x))

		x = asarray([.01, .1])
		y = asarray([10001.6212135283, 101.433299150792758])

		self.assertLess(max(abs(polygamma(1, x).ravel() - y)), 1e-7)



	def test_sample_dirichlet(self):
		N = 100

		for K in [2, 5, 10]:
			for alpha in [.1, .5, 1., 4., 50.]:
				samples0 = dirichlet(zeros(K) + alpha, size=N).T
				samples1 = sample_dirichlet(K, N, alpha)

				p = ks_2samp(samples0.ravel(), samples1.ravel())[1]

				self.assertGreater(p, 1e-6)
				self.assertLess(max(abs(1. - samples1.sum(0))), 1e-6)



	def test_load_users(self):
		tmp_file = mkstemp()[1]

		M = 10
		B = 2
		N = 100
		p = .05
		uids = permutation(1000)

		# generate M random users
		users = {uid: [(i + 1, randint(1, 6)) for i in permutation(N)[:binomial(N, p)]]
			for uid in uids[:M]}

		with open(tmp_file, 'w') as handle:
			for uid in users:
				for item, rating in users[uid]:
					handle.write('{0} {1} {2}\n'.format(uid, item, rating))

		threshold = 3

		users_ = load_users_as_dict(tmp_file, threshold=threshold)

		missing_users = set(users.keys()).difference(users_.keys())

		# users should only be missing if all ratings are below threshold
		for uid in missing_users:
			for item, rating in users[uid]:
				self.assertLess(rating, threshold)

		# load users in batches
		users_ = {}
		for batch in load_users_as_dict(tmp_file, batch_size=B, threshold=0):
			users_.update(batch)

		# all users should be present since threshold is zero
		for uid in users:
			self.assertEqual(set(users_[uid]), set(users[uid]))



if __name__ == '__main__':
	unittest.main()
