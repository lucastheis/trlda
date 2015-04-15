#include <utility>
using std::pair;
using std::make_pair;

#include "Eigen/Core"
using Eigen::Array;
using Eigen::Dynamic;
using Eigen::ArrayXi;
using Eigen::ArrayXd;
using Eigen::ArrayXXd;

#include "lda.h"
#include "utils.h"

TRLDA::LDA::Parameters::Parameters(
	InferenceMethod inferenceMethod,
	double threshold,
	int maxIterInference,
	int maxIterTR,
	double tau,
	double kappa,
	double rho,
	bool adaptive,
	int numSamples,
	int burnIn,
	bool initializeGamma,
	bool updateLambda,
	bool updateAlpha,
	bool updateEta,
	double minAlpha,
	double minEta,
	int maxEpochs,
	int maxIterAlpha,
	int maxIterEta,
	double empBayesThreshold,
	int verbosity) :
	inferenceMethod(inferenceMethod),
	threshold(threshold),
	maxIterInference(maxIterInference),
	maxIterTR(maxIterTR),
	tau(tau),
	kappa(kappa),
	rho(rho),
	adaptive(adaptive),
	numSamples(numSamples),
	burnIn(burnIn),
	initGamma(initializeGamma),
	updateLambda(updateLambda),
	updateAlpha(updateAlpha),
	updateEta(updateEta),
	minAlpha(minAlpha),
	minEta(minEta),
	maxEpochs(maxEpochs),
	maxIterAlpha(maxIterAlpha),
	maxIterEta(maxIterEta),
	empBayesThreshold(empBayesThreshold),
	verbosity(verbosity)
{
}



TRLDA::LDA::LDA(
	int numWords,
	int numTopics,
	double alpha,
	double eta) :
		mEta(eta)
{
	mAlpha = ArrayXd::Constant(numTopics, alpha);
	mLambda = sampleGamma(numTopics, numWords, 100) / 100.;
}



TRLDA::LDA::LDA(
	int numWords,
	ArrayXd alpha,
	double eta) :
		mAlpha(alpha),
		mEta(eta)
{
	mLambda = sampleGamma(alpha.size(), numWords, 100) / 100.;
}



TRLDA::LDA::Documents TRLDA::LDA::sample(int numDocuments, double length) {
	Documents documents;

	// sample document lengths
	ArrayXi lengths = samplePoisson(numDocuments, 1, length);

	// sample beta
	ArrayXXd beta(numTopics(), numWords());
	for(int k = 0; k < numTopics(); ++k)
		beta.row(k) = sampleDirichlet(mLambda.row(k).transpose()).transpose();

	for(int n = 0; n < numDocuments; ++n) {
		// sample theta
		ArrayXd theta = sampleDirichlet(mAlpha);

		// sample words
		Document document;
		for(int i = 0; i < lengths[n]; ++i) {
			int k = sampleHistogram(theta);
			int wordID = sampleHistogram(beta.row(k));
			document.push_back(make_pair(wordID, 1));
		}

		documents.push_back(document);
	}

	return documents;
}



pair<ArrayXXd, ArrayXXd> TRLDA::LDA::updateVariables(
		const Documents& documents,
		const Parameters& parameters) const
{
	switch(parameters.inferenceMethod) {
		case GIBBS:
			return updateVariables(
				documents,
				sampleDirichlet(numTopics(), documents.size()),
				parameters);

		case VI:
		default:
			// initialize with random gamma
			return updateVariables(
				documents,
				sampleGamma(numTopics(), documents.size(), 100) / 100.,
				parameters);
	}
}



pair<ArrayXXd, ArrayXXd> TRLDA::LDA::updateVariables(
		const Documents& documents,
		const ArrayXXd& latents,
		const Parameters& parameters) const
{
	switch(parameters.inferenceMethod) {
		case GIBBS:
			return updateVariablesGibbs(documents, latents, parameters);

		case VI:
		default:
			return updateVariablesVI(documents, latents, parameters);

	}
}



pair<ArrayXXd, ArrayXXd> TRLDA::LDA::updateVariablesVI(
		const Documents& documents,
		const ArrayXXd& initialGamma,
		const Parameters& parameters) const
{
	if(initialGamma.rows() != numTopics() || initialGamma.cols() != documents.size())
		throw Exception("Initial gamma has wrong dimensionality.");

	ArrayXXd gamma = initialGamma;
	ArrayXXd sstats = ArrayXXd::Zero(numTopics(), numWords());

	// compute $\exp E[ \beta | \lambda ]$ and $A \exp E[ \theta \mid \gamma ]$
	ArrayXd psiSum = digamma(mLambda.rowwise().sum());
	MatrixXd expPsiLambda = (digamma(mLambda).colwise() - psiSum).exp();
	MatrixXd expPsiGamma = digamma(gamma).exp();

	#pragma omp parallel for
	for(int i = 0; i < documents.size(); ++i) {
		// select columns (words) needed for this document
		MatrixXd expPsiLambdaDoc(numTopics(), documents[i].size());
		for(int j = 0; j < documents[i].size(); ++j)
			expPsiLambdaDoc.col(j) = expPsiLambda.col(documents[i][j].first);

		ArrayXd phiNorm = (expPsiGamma.col(i).transpose() * expPsiLambdaDoc).array() + 1e-100;

		for(int k = 0; k < parameters.maxIterInference; ++k) {
			ArrayXd lastGamma = gamma.col(i);

			// recompute gamma, represent phi implicitly
			gamma.col(i).setZero();
			for(int j = 0; j < documents[i].size(); ++j) {
				const int& wordCount = documents[i][j].second;
				gamma.col(i) += wordCount / phiNorm[j] * expPsiLambdaDoc.col(j).array();
			}
			gamma.col(i) *= expPsiGamma.array().col(i);
			gamma.col(i) += mAlpha;

			expPsiGamma.col(i) = digamma(gamma.col(i)).exp();

			phiNorm = (expPsiGamma.col(i).transpose() * expPsiLambdaDoc).array() + 1e-100;

			// test for convergence
			if((lastGamma - gamma.col(i)).abs().mean() < parameters.threshold)
				break;
		}

		// update sufficient statistics
		for(int j = 0; j < documents[i].size(); ++j) {
			const int& wordID = documents[i][j].first;
			const int& wordCount = documents[i][j].second;

			#pragma omp critical
			sstats.col(wordID) += wordCount / phiNorm[j] * expPsiGamma.array().col(i);
		}
	}

	// finish computing sufficient statistics
	sstats *= expPsiLambda.array();

	return make_pair(gamma, sstats);
}



pair<ArrayXXd, ArrayXXd> TRLDA::LDA::updateVariablesGibbs(
		const Documents& documents,
		const ArrayXXd& initialTheta,
		const Parameters& parameters) const
{
	if(initialTheta.rows() != numTopics() || initialTheta.cols() != documents.size())
		throw Exception("Initial theta has wrong dimensionality.");

	ArrayXXd sstats = ArrayXXd::Zero(numTopics(), numWords());
	ArrayXXd theta = initialTheta;
	double unit = 1. / parameters.numSamples;

	// compute $\exp E[ \beta | \lambda ]$
	ArrayXd psiSum = digamma(mLambda.rowwise().sum());
	ArrayXXd expPsiLambda = (digamma(mLambda).colwise() - psiSum).exp();

	#pragma omp parallel for
	for(int i = 0; i < documents.size(); ++i) {
		// container for topics associated with this document
		vector<vector<int> > topics(documents[i].size());

		// counts the occurrences of topics in this document (plus alpha)
		ArrayXd counts = mAlpha;

		// initialize topics (blocked Gibbs sampling)
		for(int j = 0; j < documents[i].size(); ++j) {
			const int& wordid = documents[i][j].first;
			const int& wordcount = documents[i][j].second;

			// unnormalized distribution over topics conditioned on theta
			ArrayXd dist = expPsiLambda.col(wordid) * theta.col(j);

			// for each occurrence of the word
			for(int k = 0; k < wordcount; ++k) {
				// sample a topic
				topics[j].push_back(sampleHistogram(dist));
				counts[topics[j][k]] += 1.;
			}
		}

		for(int s = 0; s < parameters.numSamples + parameters.burnIn; ++s) {
			// update each topic once (collapsed Gibbs sampling)
			for(int j = 0; j < documents[i].size(); ++j) {
				const int& wordid = documents[i][j].first;
				const int& wordcount = documents[i][j].second;

				for(int k = 0; k < wordcount; ++k) {
					counts[topics[j][k]] -= 1.;
					topics[j][k] = sampleHistogram(expPsiLambda.col(wordid) * counts);
					counts[topics[j][k]] += 1.;
				}
			}

			if(s >= parameters.burnIn)
				// collect sufficient statistics
				for(int j = 0; j < documents[i].size(); ++j) {
					const int& wordid = documents[i][j].first;
					const int& wordcount = documents[i][j].second;

					for(int k = 0; k < wordcount; ++k)
						sstats(topics[j][k], wordid) += unit;
				}
		}

		// resample theta
		theta.col(i) = sampleDirichlet(counts);
	}

	return make_pair(theta, sstats);
}



double TRLDA::LDA::lowerBound(
	const Documents& documents,
	const Parameters& parameters,
	int numDocuments) const
{
	// correction factor for using a subset of total documents
	double factor = numDocuments >= 0 ?
		numDocuments / static_cast<double>(documents.size()) : 1;

	// inference over latent variables
	pair<ArrayXXd, ArrayXXd> results = updateVariables(documents, parameters);
	ArrayXXd& gamma = results.first;
	ArrayXXd& sstats = results.second;

	// some helper variables
	ArrayXXd psiLambda = digamma(mLambda);
	ArrayXd lambdaSum = mLambda.rowwise().sum();
	ArrayXd psiLambdaSum = digamma(lambdaSum);

	// terms due to p(w | z, \beta), p(\beta), and q(\beta)
	double EqLogPwPb = ((mEta + factor * sstats - mLambda) * (psiLambda.colwise() - psiLambdaSum)).sum();

	// terms due to p(z) and q(z)
	double EqLogPz = 0.;

	// terms due to p(\theta) and q(\theta)
	double EqLogPthta = 0.;

	for(int d = 0; d < documents.size(); ++d) {
		double gammaSum = gamma.col(d).sum();
		ArrayXd psiGamma = digamma(gamma.col(d));
		ArrayXXd phi(numTopics(), documents[d].size());
		Array<int, 1, Dynamic> counts(documents[d].size());

		// recompute phi
		for(int i = 0; i < documents[d].size(); ++i) {
			counts[i] = documents[d][i].second;
			phi.col(i) = psiLambda.row(documents[d][i].first);
		}
		phi.colwise() -= psiLambdaSum;
		phi.colwise() += psiGamma;
		phi.rowwise() -= logSumExp(phi);
		phi = phi.exp();

		ArrayXd EqLogTheta = psiGamma - digamma(gammaSum);

		Array<double, 1, Dynamic> tmp = EqLogTheta.matrix().transpose() * phi.matrix();
		tmp -= (phi * phi.log()).colwise().sum();

		// sum over words
		EqLogPz += (counts.cast<double>() * tmp).sum();

		EqLogPthta += ((mAlpha - gamma.col(d)) * EqLogTheta).sum();
		EqLogPthta -= lngamma(gammaSum);
		EqLogPthta += lngamma(gamma.col(d)).sum();
	}

	// Dirichlet normalization constants
	EqLogPthta += (lngamma(mAlpha.sum()) - lngamma(mAlpha).sum()) * documents.size();
	EqLogPwPb += numTopics() * lngamma(numWords() * mEta) - lngamma(lambdaSum).sum();
	EqLogPwPb -= numTopics() * numWords() * lngamma(mEta) - lngamma(mLambda).sum();

	return EqLogPwPb + factor * EqLogPz + factor * EqLogPthta;
}
