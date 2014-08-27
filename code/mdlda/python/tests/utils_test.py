import sys
import unittest

from numpy import asarray, any
from mdlda.utils import random_select

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



if __name__ == '__main__':
	unittest.main()
