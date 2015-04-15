#ifndef TRLDA_LDA_H
#define TRLDA_LDA_H

#include <utility>
#include <vector>

#include "Eigen/Core"
#include "distribution.h"
#include "exception.h"

namespace TRLDA {
	using std::pair;
	using std::vector;

	using Eigen::ArrayXd;
	using Eigen::ArrayXi;
	using Eigen::ArrayXXd;

	class LDA : public Distribution {
		public:
			typedef pair<int, int> Word;
			typedef vector<Word> Document;
			typedef vector<Document> Documents;

			enum InferenceMethod {
				VI, GIBBS, MAP
			};

			/**
			 * Training parameters.
			 */
			struct Parameters {
				public:
					InferenceMethod inferenceMethod;
					double threshold;
					int maxIterInference;
					int maxIterTR;
					double tau;
					double kappa;
					double rho;
					bool adaptive;
					int numSamples;
					int burnIn;
					bool initGamma;
					bool updateLambda;
					bool updateAlpha;
					bool updateEta;
					double minAlpha;
					double minEta;
					int maxEpochs;
					int maxIterAlpha;
					int maxIterEta;
					double empBayesThreshold;
					int verbosity;

					Parameters(
						InferenceMethod inferenceMethod = VI,
						double threshold = 0.001,
						int maxIterInference = 100,
						int maxIterMd = 20,
						double tau = 1024.,
						double kappa = .9,
						double rho = -1.,
						bool adaptive = false,
						int numSamples = 1,
						int burnIn = 2,
						bool initGamma = true,
						bool updateLambda = true,
						bool updateAlpha = false,
						bool updateEta = false,
						double minAlpha = 1e-6,
						double minEta = 1e-6,
						int maxEpochs = 100,
						int maxIterAlpha = 10,
						int maxIterEta = 20,
						double empBayesThreshold = 1e-8,
						int verbosity = 0);
			};

			LDA(
				int numWords,
				int numTopics,
				double alpha = .1,
				double eta = .3);
			LDA(
				int numWords,
				ArrayXd alpha,
				double eta = .3);

			inline int numTopics() const;
			inline int numWords() const;

			inline ArrayXd alpha() const;
			inline void setAlpha(double alpha);
			inline void setAlpha(const ArrayXd& alpha);

			inline double eta() const;
			inline void setEta(double eta);

			inline ArrayXXd lambda() const;
			inline void setLambda(const ArrayXXd& lambda);

			virtual Documents sample(int numDocuments, double length);

			virtual pair<ArrayXXd, ArrayXXd> updateVariables(
				const Documents& documents,
				const Parameters& parameters = Parameters()) const;
			virtual pair<ArrayXXd, ArrayXXd> updateVariables(
				const Documents& documents,
				const ArrayXXd& latents,
				const Parameters& parameters = Parameters()) const;

			virtual pair<ArrayXXd, ArrayXXd> updateVariablesVI(
				const Documents& documents,
				const ArrayXXd& initialGamma,
				const Parameters& parameters = Parameters()) const;
			virtual pair<ArrayXXd, ArrayXXd> updateVariablesGibbs(
				const Documents& documents,
				const ArrayXXd& initialTheta,
				const Parameters& parameters = Parameters()) const;

			virtual double updateParameters(
				const Documents& documents,
				const Parameters& parameters) = 0;

			virtual double lowerBound(
				const Documents& documents,
				const Parameters& parameters = Parameters(),
				int numDocuments = -1) const;

		protected:
			ArrayXd mAlpha;
			double mEta;
			ArrayXXd mLambda;
	};
}



inline Eigen::ArrayXd TRLDA::LDA::alpha() const {
	return mAlpha;
}



inline void TRLDA::LDA::setAlpha(double alpha) {
	if(alpha < 0.)
		throw Exception("Alpha should not be negative.");
	mAlpha.setConstant(alpha);
}



inline void TRLDA::LDA::setAlpha(const ArrayXd& alpha) {
	if(alpha.size() != numTopics())
		throw Exception("Alpha has wrong dimensionality.");
	for(int i = 0; i < alpha.size(); ++i)
		if(alpha[i] < 0.)
			throw Exception("Alpha should not be negative.");
	mAlpha = alpha;
}



inline double TRLDA::LDA::eta() const {
	return mEta;
}



inline void TRLDA::LDA::setEta(double eta) {
	if(eta < 0.)
		throw Exception("Eta should not be negative.");
	mEta = eta;
}



inline Eigen::ArrayXXd TRLDA::LDA::lambda() const {
	return mLambda;
}



inline void TRLDA::LDA::setLambda(const ArrayXXd& lambda) {
	if(lambda.rows() != numTopics() || lambda.cols() != numWords())
		throw Exception("Lambda has wrong dimensionality.");
	mLambda = lambda;
}



inline int TRLDA::LDA::numTopics() const {
	return mLambda.rows();
}



inline int TRLDA::LDA::numWords() const {
	return mLambda.cols();
}

#endif
