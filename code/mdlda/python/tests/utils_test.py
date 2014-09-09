import sys
import unittest

from numpy import asarray, any, zeros, abs, max
from numpy.random import dirichlet
from scipy.stats import ks_2samp
from mdlda.utils import random_select, sample_dirichlet

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



	def test_sample_dirichlet(self):
		N = 100

		for K in [2, 5, 10]:
			for alpha in [.1, .5, 1., 4., 50.]:
				samples0 = dirichlet(zeros(K) + alpha, size=[N]).T
				samples1 = sample_dirichlet(K, N, alpha)


				p = ks_2samp(samples0.ravel(), samples1.ravel())[1]

				self.assertGreater(p, 1e-6)
				self.assertLess(max(abs(1. - samples1.sum(0))), 1e-6)



if __name__ == '__main__':
	unittest.main()
