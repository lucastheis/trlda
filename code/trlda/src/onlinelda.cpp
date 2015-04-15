#include <utility>
using std::pair;
using std::make_pair;

#include <cmath>
using std::pow;

#include "Eigen/Core"
using Eigen::Array;
using Eigen::Dynamic;
using Eigen::ArrayXi;
using Eigen::ArrayXd;
using Eigen::ArrayXXd;

#include "onlinelda.h"
#include "utils.h"

TRLDA::OnlineLDA::OnlineLDA(
	int numWords,
	int numTopics,
	int numDocuments,
	double alpha,
	double eta) :
		LDA(numWords, numTopics, alpha, eta),
		mNumDocuments(numDocuments),
		mUpdateCounter(0)
{
	mAdaTau = 1000.;
	mAdaRho = 1. / mAdaTau;
	mAdaSqNorm = 1.;
	mAdaGradient = ArrayXXd::Zero(numTopics, numWords);
}



TRLDA::OnlineLDA::OnlineLDA(
	int numWords,
	int numDocuments,
	ArrayXd alpha,
	double eta) :
		LDA(numWords, alpha, eta),
		mNumDocuments(numDocuments),
		mUpdateCounter(0)
{
	mAdaTau = 1000.;
	mAdaRho = 1. / mAdaTau;
	mAdaSqNorm = 1.;
	mAdaGradient = ArrayXXd::Zero(alpha.size(), numWords);
}



double TRLDA::OnlineLDA::updateParameters(const Documents& documents, const Parameters& parameters) {
	if(documents.size() == 0)
		// nothing to be done
		return true;

	// choose a learning rate
	double rho = parameters.rho;
	if(rho < 0.) {
		if(parameters.adaptive) {
			rho = mAdaRho;
		} else {
			rho = pow(parameters.tau + mUpdateCounter, -parameters.kappa);
		}
	}

	ArrayXXd lambdaPrime = mLambda;
	ArrayXXd lambdaHat;

	pair<ArrayXXd, ArrayXXd> results;


	//// UPDATE LAMBDA

	if(parameters.updateLambda) {
		if(parameters.maxIterTR > 0) {
			// sufficient statistics as if $\phi_{dwk}$ was 1/K
			ArrayXd wordcounts = ArrayXd::Zero(numWords());
			for(int i = 0; i < documents.size(); ++i)
				for(int j = 0; j < documents[i].size(); ++j)
					wordcounts[documents[i][j].first] += documents[i][j].second;

			// initial update to lambda to avoid local optima
			mLambda = ((1. - rho) * lambdaPrime).rowwise()
				+ rho * (mEta + static_cast<double>(mNumDocuments) / documents.size() / numTopics() * wordcounts.transpose());

			// trust region update iterations
			for(int i = 0; i < parameters.maxIterTR; ++i) {
				// compute sufficient statistics (E-step)
				if(i > 0 && parameters.initGamma)
					// initialize with gamma of previous iteration
					results = updateVariables(documents, results.first, parameters);
				else
					results = updateVariables(documents, parameters);
				ArrayXXd& sstats = results.second;

				// update parameters (M-step)
				lambdaHat = mEta + static_cast<double>(mNumDocuments) / documents.size() * sstats;
				mLambda = (1. - rho) * lambdaPrime + rho * lambdaHat;
			}
		} else {
			// compute sufficient statistics (E-step)
			results = updateVariables(documents, parameters);
			ArrayXXd& sstats = results.second;

			// update parameters (M-step)
			lambdaHat = mEta + static_cast<double>(mNumDocuments) / documents.size() * sstats;
			mLambda = (1. - rho) * lambdaPrime + rho * lambdaHat;
		}
	}


	//// UPDATE ALPHA

	if(parameters.updateAlpha) {
		if(!parameters.updateLambda)
			// estimate distribution over theta
			results = updateVariables(documents, parameters);

		// empirical Bayes update of alpha
		ArrayXXd& gamma = results.first;

		ArrayXXd psiGamma = digamma(gamma);
		Array<double, 1, Dynamic> psiGammaSum = digamma(gamma.colwise().sum());

		// gradient of lower bound with respect to alpha
		ArrayXd g = (psiGamma.rowwise() - psiGammaSum).rowwise().sum()
			- documents.size() * (digamma(mAlpha) - digamma(mAlpha.sum()));

		// components that make up Hessian
		ArrayXd h = -static_cast<double>(documents.size()) * polygamma(1, mAlpha);
		double z = documents.size() * polygamma(1, mAlpha.sum());
		double c = (g / h).sum() / (1. / z + (1. / h).sum());

		// perform stochastic natural gradient/Newton step
		mAlpha = mAlpha - rho * (g - c) / h;

		for(int i = 0; i < mAlpha.size(); ++i)
			if(mAlpha[i] < parameters.minAlpha)
				mAlpha[i] = parameters.minAlpha;
	}


	//// UPDATE ETA

	if(parameters.updateEta) {
		// empirical Bayes update of eta
		int K = numTopics();
		int N = numWords();

		// gradient of lower bound with respect to eta
		double g = digamma(mLambda).sum() - N * digamma(mLambda.rowwise().sum()).sum()
			- K * N * (digamma(mEta) - digamma(N * mEta));
		double h = K * N * (polygamma(1, N * mEta) - polygamma(1, mEta));

		// perform stochastic natural gradient/Newton step
		mEta = mEta - rho * g / h;

		if(mEta < parameters.minEta)
			mEta = parameters.minEta;
	}


	//// ADJUST LEARNING RATES

	if(parameters.updateLambda && parameters.adaptive) {
		ArrayXXd lambdaUpdate = lambdaHat - lambdaPrime;

		// compute running average of gradients and adjust learning rate
		mAdaGradient = (1. - 1. / mAdaTau) * mAdaGradient + 1. / mAdaTau * lambdaUpdate;
		mAdaSqNorm   = (1. - 1. / mAdaTau) * mAdaSqNorm   + 1. / mAdaTau * lambdaUpdate.matrix().squaredNorm();
		mAdaRho = mAdaGradient.matrix().squaredNorm() / mAdaSqNorm;
		mAdaTau = mAdaTau * (1. - mAdaRho) + 1.;
	}

	mUpdateCounter++;

	return rho;
}



double TRLDA::OnlineLDA::lowerBound(
	const Documents& documents,
	const Parameters& parameters,
	int numDocuments) const
{
	return LDA::lowerBound(documents, parameters,
		numDocuments >= 0 ? numDocuments : mNumDocuments);
}
