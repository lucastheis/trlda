import unittest

from time import time
from pickle import load, dump
from tempfile import mkstemp
from random import choice, randint
from string import ascii_letters
from numpy import corrcoef, random, abs, max, asarray, round, zeros_like
from trlda.models import BatchLDA
from trlda.utils import sample_dirichlet

class Tests(unittest.TestCase):
	def test_basics(self):
		W = 102
		D = 1010
		K = 11
		alpha = .27
		eta = 3.1

		model = BatchLDA(num_words=W, num_topics=K, alpha=alpha, eta=eta)

		self.assertEqual(K, model.num_topics)
		self.assertEqual(K, model.alpha.size)
		self.assertEqual(W, model.num_words)
		self.assertEqual(alpha, model.alpha.ravel()[randint(0, K - 1)])
		self.assertEqual(eta, model.eta)

		with self.assertRaises(RuntimeError):
			model.alpha = random.rand(K + 1)

		alpha = random.rand(K, 1)
		model.alpha = alpha
		self.assertLess(max(abs(model.alpha.ravel() - alpha.ravel())), 1e-20)



	def test_empirical_bayes_alpha(self):
		model = BatchLDA(
			num_words=4,
			num_topics=2,
			alpha=[.2, .05],
			eta=.2)

		model.lambdas = [
			[100, 100, 1e-16, 1e-16],
			[1e-16, 1e-16, 100, 100]]

		documents = model.sample(num_documents=100, length=20)

		# set alpha to wrong values
		model.alpha = [4., 4.]

		model.update_parameters(documents,
			max_epochs=10,
			max_iter_inference=200,
			update_lambda=False,
			update_alpha=True,
			emp_bayes_threshold=0.)

		# make sure empirical Bayes went in the right direction
		self.assertGreater(model.alpha[0], model.alpha[1])
		self.assertLess(model.alpha[0], 4.)
		self.assertLess(model.alpha[1], 4.)



	def test_empirical_bayes_eta(self):
		for eta, initial_eta in [(.045, .2), (.41, .2)]:
			model = BatchLDA(
				num_words=100,
				num_topics=10,
				alpha=[.1, .1],
				eta=initial_eta)

			# this will sample a beta with the given eta
			model.lambdas = zeros_like(model.lambdas) + eta
			documents = model.sample(500, 10)

			model.update_parameters(documents, 
				max_epochs=10,
				update_eta=True,
				emp_bayes_threshold=0.)

			# optimization should at least walk in the right direction and don't explode
			self.assertLess(abs(model.eta - eta), abs(model.eta - initial_eta))



	def test_pickle(self):
		model0 = BatchLDA(
			num_words=300,
			num_topics=50,
			alpha=random.rand(),
			eta=random.rand())

		tmp_file = mkstemp()[1]

		# save model
		with open(tmp_file, 'w') as handle:
			dump({'model': model0}, handle)

		# load model
		with open(tmp_file) as handle:
			model1 = load(handle)['model']

		# make sure parameters haven't changed
		self.assertEqual(model0.num_words, model1.num_words)
		self.assertEqual(model0.num_topics, model1.num_topics)
		self.assertLess(max(abs(model0.lambdas - model1.lambdas)), 1e-20)
		self.assertLess(max(abs(model0.alpha - model1.alpha)), 1e-20)
		self.assertLess(abs(model0.eta - model1.eta), 1e-20)



if __name__ == '__main__':
	unittest.main()
