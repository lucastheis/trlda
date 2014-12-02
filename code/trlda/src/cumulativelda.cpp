#include <utility>
using std::pair;
using std::make_pair;

#include <cmath>
using std::pow;

#include <iostream>
using std::cout;
using std::endl;

#include "Eigen/Core"
using Eigen::Array;
using Eigen::Dynamic;
using Eigen::ArrayXi;
using Eigen::ArrayXd;
using Eigen::ArrayXXd;

#include "cumulativelda.h"
#include "utils.h"

TRLDA::CumulativeLDA::CumulativeLDA(
	int numWords,
	int numTopics,
	double alpha,
	double eta) :
		LDA(numWords, numTopics, alpha, eta)
{
	mLambda.setConstant(eta);
}



TRLDA::CumulativeLDA::CumulativeLDA(
	int numWords,
	ArrayXd alpha,
	double eta) :
		LDA(numWords, alpha, eta)
{
	mLambda.setConstant(eta);
}



double TRLDA::CumulativeLDA::updateParameters(const Documents& documents, const Parameters& parameters) {
	if(documents.size() == 0)
		// nothing to be done
		return true;


	//// UPDATE LAMBDA

	ArrayXXd lambdaPrime = mLambda;

	if(parameters.updateLambda) {
		for(int epoch = 0; epoch < parameters.maxEpochs; ++epoch) {
			// compute sufficient statistics (E-step)
			pair<ArrayXXd, ArrayXXd> results = updateVariables(documents, parameters);
			ArrayXXd& sstats = results.second;

			// update parameters (M-step)
			mLambda = lambdaPrime + sstats;
		}
	}


	//// UPDATE ALPHA

	if(parameters.updateAlpha) {
		// estimate distribution over theta
		pair<ArrayXXd, ArrayXXd> results = updateVariables(documents, parameters);
		ArrayXXd& gamma = results.first;

		ArrayXXd psiGamma = digamma(gamma);
		Array<double, 1, Dynamic> psiGammaSum = digamma(gamma.colwise().sum());

		mPsiGammaDiff += (psiGamma.rowwise() - psiGammaSum).rowwise().sum();

		if(parameters.verbosity > 1)
			cout << "Optimizing alpha..." << endl;

		// lower bound module constants
		double L = documents.size() * (lngamma(mAlpha.sum()) - lngamma(mAlpha).sum())
			+ (mPsiGammaDiff * (mAlpha - 1.)).sum();
		double Lprime = L;

		for(int i = 0; i < parameters.maxIterAlpha; ++i) {
			if(parameters.verbosity > 1)
				cout << "\tCurrent function value: " << L << endl;

			// gradient of lower bound with respect to alpha
			ArrayXd g = mPsiGammaDiff - documents.size() * (digamma(mAlpha) - digamma(mAlpha.sum()));

			// components that make up Hessian
			ArrayXd h = -static_cast<double>(documents.size()) * polygamma(1, mAlpha);
			double z = documents.size() * polygamma(1, mAlpha.sum());
			double c = (g / h).sum() / (1. / z + (1. / h).sum());

			double rho = .2;

			// line search along natural gradient/Newton step
			for(int j = 0; j < 20; ++j) {
				ArrayXd alpha = mAlpha - rho * (g - c) / h;

				// check for too small alphas
				bool smallAlpha = false;
				for(int i = 0; i < alpha.size(); ++i)
					if(alpha[i] < parameters.minAlpha) {
						smallAlpha = true;
						break;
					}
				if(smallAlpha) {
					rho /= 2.;
					continue;
				}

				Lprime = documents.size() * (lngamma(alpha.sum()) - lngamma(alpha).sum())
					+ (mPsiGammaDiff * (alpha - 1.)).sum();

				if(L <= Lprime) {
					if(parameters.verbosity > 1) {
						cout << "\tStep width: " << rho << endl;
						cout << "\tGradient magnitude: " << g.matrix().norm() << endl;
					}

					// lower bound has increased
					mAlpha = alpha;
					break;
				}

				// try again with smaller learning rate
				rho /= 2.;
			}

			if(Lprime - L < parameters.empBayesThreshold)
				// improvement was very small
				break;

			// update current function value
			L = Lprime;

		}
	}

	return 1.;
}
