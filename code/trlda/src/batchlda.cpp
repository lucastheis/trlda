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

#include "batchlda.h"
#include "utils.h"

TRLDA::BatchLDA::BatchLDA(
	int numWords,
	int numTopics,
	double alpha,
	double eta) :
		LDA(numWords, numTopics, alpha, eta)
{
}



TRLDA::BatchLDA::BatchLDA(
	int numWords,
	ArrayXd alpha,
	double eta) :
		LDA(numWords, alpha, eta)
{
}



double TRLDA::BatchLDA::updateParameters(const Documents& documents, const Parameters& parameters) {
	if(documents.size() == 0)
		// nothing to be done
		return 1.;

	for(int epoch = 0; epoch < parameters.maxEpochs; ++epoch) {
		pair<ArrayXXd, ArrayXXd> results;


		//// UPDATE LAMBDA

		if(parameters.updateLambda) {
			// compute sufficient statistics (E-step)
			results = updateVariables(documents, parameters);
			ArrayXXd& sstats = results.second;

			// update parameters (M-step)
			mLambda = mEta + sstats;
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
			ArrayXd psiGammaDiff = (psiGamma.rowwise() - psiGammaSum).rowwise().sum();

			if(parameters.verbosity > 1)
				cout << "Optimizing alpha..." << endl;

			// lower bound module constants
			double L = documents.size() * (lngamma(mAlpha.sum()) - lngamma(mAlpha).sum())
				+ (psiGammaDiff * (mAlpha - 1.)).sum();
			double Lprime = L;

			for(int i = 0; i < parameters.maxIterAlpha; ++i) {
				if(parameters.verbosity > 1)
					cout << "\tCurrent function value: " << L << endl;

				// gradient of lower bound with respect to alpha
				ArrayXd g = psiGammaDiff - documents.size() * (digamma(mAlpha) - digamma(mAlpha.sum()));

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
						+ (psiGammaDiff * (alpha - 1.)).sum();

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


		//// UPDATE ETA

		if(parameters.updateEta) {
			// empirical Bayes update of eta
			int K = numTopics();
			int N = numWords();

			// constant independent of eta
			double c = digamma(mLambda).sum() - N * digamma(mLambda.rowwise().sum()).sum();

			if(parameters.verbosity > 1)
				cout << "Optimizing eta..." << endl;

			// lower bound modulo constants
			double L = (mEta - 1) * c + K * lngamma(N * mEta) - K * N * lngamma(mEta);
			double Lprime = L;

			for(int i = 0; i < parameters.maxIterEta; ++i) {
				if(parameters.verbosity > 1)
					cout << "\tCurrent function value: " << L << endl;

				// gradient of lower bound with respect to eta
				double g = c - K * N * (digamma(mEta) - digamma(N * mEta));
				double h = K * N * (polygamma(1, N * mEta) - polygamma(1, mEta));

				double rho = .5;

				// line search along natural gradient/Newton step
				for(int j = 0; j < 20; ++j) {
					double eta = mEta - rho * g / h;

					if(eta < parameters.minEta) {
						rho /= 2.;
						continue;
					}

					Lprime = (eta - 1) * c + K * lngamma(N * eta) - K * N * lngamma(eta);

					if(L <= Lprime) {
						if(parameters.verbosity > 1) {
							cout << "\tStep width: " << rho << endl;
							cout << "\tGradient: " << g << endl;
						}

						// lower bound has increased
						mEta = eta;
						break;
					}

					// try again with smaller rho
					rho /= 2.;
				}

				if(Lprime - L < parameters.empBayesThreshold)
					// improvement was very small
					break;

				// current function value
				L = Lprime;
			}
		}
	}

	return 1.;
}
