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

MDLDA::OnlineLDA::OnlineLDA(
	int numWords,
	int numTopics,
	int numDocuments,
	double alpha,
	double eta,
	double tau,
	double kappa) :
		mNumDocuments(numDocuments),
		mAlpha(alpha),
		mEta(eta),
		mTau(tau),
		mKappa(kappa),
		mUpdateCounter(0)
{
	mLambda = sampleGamma(numTopics, numWords, 100) / 100.;
}



pair<ArrayXXd, ArrayXXd> MDLDA::OnlineLDA::updateVariables(
		const Documents& documents,
		int maxIter,
		double threshold) const
{
	return updateVariables(
		documents,
		sampleGamma(numTopics(), documents.size(), 100) / 100.,
		maxIter,
		threshold);
}



pair<ArrayXXd, ArrayXXd> MDLDA::OnlineLDA::updateVariables(
		const Documents& documents,
		const ArrayXXd& initialGamma,
		int maxIter,
		double threshold) const
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

		for(int k = 0; k < maxIter; ++k) {
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

			if((lastGamma - gamma.col(i)).abs().mean() < threshold)
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



bool MDLDA::OnlineLDA::updateParameters(const Documents& documents, int maxIter, double rho) {
	if(rho < 0.)
		rho = pow(mTau + mUpdateCounter, -mKappa);

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
	for(int i = 0; i < maxIter; ++i) {
		if(i > 0)
			// initialize with gamma of previous iteration
			results = updateVariables(documents, results.first);
		else
			results = updateVariables(documents);

		ArrayXXd& gamma = results.first;
		ArrayXXd& sstats = results.second;

		mLambda = (1. - rho) * lambdaPrime
			+ rho * (mEta + mNumDocuments / documents.size() * sstats);
	}

	mUpdateCounter++;

	return true;
}
