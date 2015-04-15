import unittest

from time import time
from pickle import load, dump
from tempfile import mkstemp
from random import choice, randint
from string import ascii_letters
from numpy import corrcoef, random, abs, max, asarray, round, zeros_like
from trlda.models import OnlineLDA
from trlda.utils import sample_dirichlet
from onlineldavb import OnlineLDA as ReferenceLDA

class Tests(unittest.TestCase):
	def test_basics(self):
		W = 102
		D = 1010
		K = 11
		alpha = .27
		eta = 3.1

		model = OnlineLDA(num_words=W, num_topics=K, num_documents=D, alpha=alpha, eta=eta)

		self.assertEqual(K, model.num_topics)
		self.assertEqual(K, model.alpha.size)
		self.assertEqual(D, model.num_documents)
		self.assertEqual(W, model.num_words)
		self.assertEqual(alpha, model.alpha.ravel()[randint(0, K - 1)])
		self.assertEqual(eta, model.eta)

		with self.assertRaises(RuntimeError):
			model.alpha = random.rand(K + 1)

		alpha = random.rand(K, 1)
		model.alpha = alpha
		self.assertLess(max(abs(model.alpha.ravel() - alpha.ravel())), 1e-20)



	def test_vi(self):
		W = 100
		K = 20
		D = 10
		N = 100

		# generate random vocabulary
		vocab = [''.join(choice(ascii_letters) for _ in range(5 + random.randint(10))) for _ in range(W)]
		model0 = ReferenceLDA(vocab, K, D, 0.1, 0.3, 1024., 0.9)

		model1 = OnlineLDA(num_words=W, num_topics=K, num_documents=D)
		model1.alpha = model0._alpha
		model1.lambdas = model0._lambda

		# generate D random documents of length up to N
		docs1 = []
		for _ in range(D):
			docs1.append([
				(w, random.randint(10)) for w in random.permutation(W)[:1 + random.randint(N)]])
		docs0 = [zip(*doc) for doc in docs1]

		# use the same initialization of gamma
		initial_gamma = random.gamma(100., 1./100., [K, D])

		gamma0, sstats0 = model0.do_e_step(docs0, max_steps=50, gamma=initial_gamma.T)
		gamma1, sstats1 = model1.do_e_step(docs1, max_iter=50, latents=initial_gamma)

		# make sure e-Step gives the same results
		self.assertGreater(corrcoef(gamma0.T.ravel(), gamma1.ravel())[0, 1], 0.99)
		self.assertGreater(corrcoef(sstats0.ravel(), sstats1.ravel())[0, 1], 0.99)



	def test_lower_bound(self):
		W = 100
		K = 22
		D = 30
		N = 60

		# generate random vocabulary
		vocab = [''.join(choice(ascii_letters) for _ in range(5 + random.randint(10))) for _ in range(W)]
		model0 = ReferenceLDA(vocab, K, D, 0.1, 0.3, 1024., 0.9)

		model1 = OnlineLDA(num_words=W, num_topics=K, num_documents=D)
		model1.alpha = model0._alpha
		model1.lambdas = model0._lambda

		# generate D/2 random documents of average length N
		docs1 = model1.sample(D // 2, N)
		docs0 = [zip(*doc) for doc in docs1]

		# estimate lower bound
		elbo1 = model1.lower_bound(docs1)
		elbo0 = model0.approx_bound(docs0)

		# estimate should deviate less than 1% from reference estimate
		self.assertLess(abs(elbo1 - elbo0) / abs(elbo0), 0.01)



	def test_gibbs(self):
		model = OnlineLDA(num_words=100, num_topics=10, num_documents=1000)

		# random documents
		docs = []
		for _ in range(model.num_documents):
			docs.append([
				(w, random.randint(10)) for w in random.permutation(model.num_words)[:1 + random.randint(100)]])

		# make sure it doesn't segfault
		theta, sstats = model.update_variables(docs, inference_method='gibbs')



	def test_m_step(self):
		model1 = OnlineLDA(num_words=100, num_topics=10, num_documents=1000)

		# this used to cause a floating point exception
		model1.update_parameters([])



	def test_empirical_bayes_alpha(self):
		model = OnlineLDA(
			num_words=4,
			num_topics=2,
			num_documents=1000,
			alpha=[.2, .01],
			eta=.2)

		model.lambdas = [
			[100, 100, 1e-16, 1e-16],
			[1e-16, 1e-16, 100, 100]]

		documents = model.sample(100, 10)

		# set alpha to wrong values
		model.alpha = [4., 4.]

		for i in range(100):
			model.update_parameters(documents, rho=.1, max_iter_tr=0, update_lambda=False, update_alpha=True)

		# make sure empirical Bayes went in the right direction
		self.assertGreater(model.alpha[0], model.alpha[1])
		self.assertLess(model.alpha[0], 4.)
		self.assertLess(model.alpha[1], 4.)



	def test_empirical_bayes_eta(self):
		for eta, initial_eta in [(.045, .2), (.41, .2)]:
			model = OnlineLDA(
				num_words=100,
				num_topics=10,
				num_documents=500,
				alpha=[.1, .1],
				eta=initial_eta)

			# this will sample a beta with the given eta
			model.lambdas = zeros_like(model.lambdas) + eta
			documents = model.sample(500, 10)

			for i in range(50):
				model.update_parameters(documents, rho=.1, update_eta=True)

			# optimization should at least walk in the right direction and don't explode
			self.assertLess(abs(model.eta - eta), abs(model.eta - initial_eta))



	def test_pickle(self):
		model0 = OnlineLDA(
			num_words=300,
			num_topics=50,
			num_documents=11110,
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
		self.assertEqual(model0.num_documents, model1.num_documents)
		self.assertLess(max(abs(model0.lambdas - model1.lambdas)), 1e-20)
		self.assertLess(max(abs(model0.alpha - model1.alpha)), 1e-20)
		self.assertLess(abs(model0.eta - model1.eta), 1e-20)



	def test_speed(self):
		model1 = OnlineLDA(
			num_words=1000,
			num_topics=100,
			num_documents=10000,
			alpha=.1,
			eta=.3)

		# random vocabulary
		vocab = [''.join(choice(ascii_letters) for _ in range(5 + random.randint(10)))
			for _ in range(model1.num_words)]

		model0 = ReferenceLDA(vocab,
			D=model1.num_documents,
			K=model1.num_topics,
			alpha=model1.alpha[0, 0],
			eta=model1.eta,
			kappa=.9,
			tau0=1024)

		# generate D random documents of length up to N
		D = 110
		N = 600
		docs1 = []
		for _ in range(D):
			wordids = random.permutation(model1.num_words)[:1 + random.randint(N)]
			docs1.append([
				(w, random.randint(10)) for w in wordids])
		docs0 = [zip(*doc) for doc in docs1]

		initial_gamma = random.gamma(100., 1./100., [model1.num_topics, D])

		start = time()
		gamma0, _ = model0.do_e_step(docs0, max_steps=100, gamma=initial_gamma.T)
		time0 = time() - start

		start = time()
		gamma1, _ = model1.do_e_step(docs1, max_iter=100, latents=initial_gamma)
		time1 = time() - start

		# make sure that C++ implementation is actually faster than Python version
		self.assertLess(time1, time0,
			msg='Inference step took longer ({0:.2f} s) than reference implementation ({1:.2f})'.format(time1, time0))



if __name__ == '__main__':
	unittest.main()
