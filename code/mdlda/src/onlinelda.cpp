#include <utility>
using std::pair;
using std::make_pair;

#include "Eigen/Core"
using Eigen::ArrayXd;
using Eigen::ArrayXXd;

#include "onlinelda.h"
#include "utils.h"

#include <iostream>
using std::cout;
using std::endl;

#include <cmath>
using std::pow;

MDLDA::OnlineLDA::Parameters::Parameters(
	InferenceMethod inferenceMethod,
	double threshold,
	int maxIterInference,
	int maxIterMD,
	double tau,
	double kappa,
	double rho) :
	inferenceMethod(inferenceMethod),
	threshold(threshold),
	maxIterInference(maxIterInference),
	maxIterMD(maxIterMD),
	tau(tau),
	kappa(kappa),
	rho(rho)
{
}



MDLDA::OnlineLDA::OnlineLDA(
	int numWords,
	int numTopics,
	int numDocuments,
	double alpha,
	double eta) :
		mNumDocuments(numDocuments),
		mAlpha(alpha),
		mEta(eta),
		mUpdateCounter(0)
{
	mLambda = sampleGamma(numTopics, numWords, 100) / 100.;
}



pair<ArrayXXd, ArrayXXd> MDLDA::OnlineLDA::updateVariables(
		const Documents& documents,
		const Parameters& parameters) const
{
	// initialize with random gamma
	return updateVariables(
		documents,
		sampleGamma(numTopics(), documents.size(), 100) / 100.,
		parameters);
}



pair<ArrayXXd, ArrayXXd> MDLDA::OnlineLDA::updateVariables(
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
		// select columns needed for this document
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



bool MDLDA::OnlineLDA::updateParameters(const Documents& documents, const Parameters& parameters) {
	if(documents.size() == 0)
		// nothing to be done
		return true;

	// choose a learning rate
	double rho = parameters.rho;
	if(rho < 0.)
		rho = pow(parameters.tau + mUpdateCounter, -parameters.kappa);

	if(parameters.maxIterMD > 0) {
		ArrayXXd lambdaPrime = mLambda;

		// sufficient statistics if $\phi_{dwk}$ is 1/K
		ArrayXd wordcounts = ArrayXd::Zero(numWords());
		for(int i = 0; i < documents.size(); ++i)
			for(int j = 0; j < documents[i].size(); ++j)
				wordcounts[documents[i][j].first] += documents[i][j].second;

		// initial update to lambda to avoid local optima
		mLambda = ((1. - rho) * lambdaPrime).rowwise()
			+ rho * (mEta + mNumDocuments / documents.size() / numTopics() * wordcounts.transpose());

		pair<ArrayXXd, ArrayXXd> results;

		// mirror descent iterations
		for(int i = 0; i < parameters.maxIterMD; ++i) {
			// compute sufficient statistics (E-step)
			if(i > 0)
				// initialize with gamma of previous iteration
				results = updateVariables(documents, results.first);
			else
				results = updateVariables(documents);
			ArrayXXd& sstats = results.second;

			// update parameters (M-step)
			mLambda = (1. - rho) * lambdaPrime
				+ rho * (mEta + mNumDocuments / documents.size() * sstats);
		}
	} else {
		// compute sufficient statistics (E-step)
		pair<ArrayXXd, ArrayXXd> results = updateVariables(documents);
		ArrayXXd& sstats = results.second;

		// update parameters (M-step)
		mLambda = (1. - rho) * mLambda
			+ rho * (mEta + mNumDocuments / documents.size() * sstats);
	}

	mUpdateCounter++;

	return true;
}
