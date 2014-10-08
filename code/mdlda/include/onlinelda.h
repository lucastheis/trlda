#ifndef MDLDA_LDA_H
#define MDLDA_LDA_H

#include <utility>
#include <vector>

#include "Eigen/Core"
#include "distribution.h"
#include "exception.h"

namespace MDLDA {
	using std::pair;
	using std::vector;

	using Eigen::ArrayXi;
	using Eigen::ArrayXXd;

	class OnlineLDA : public Distribution {
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
					int maxIterMD;
					double tau;
					double kappa;
					double rho;
					bool adaptive;
					int numSamples;
					int burnIn;
					bool initGamma;

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
						bool initGamma = true);
			};

			OnlineLDA(
				int numWords,
				int numTopics,
				int numDocuments,
				double alpha = .1,
				double eta = .3);

			inline int numTopics() const;
			inline int numWords() const;
			inline int numDocuments() const;
			inline void setNumDocuments(int numDocuments);

			inline int updateCount() const;
			inline void setUpdateCount(int updateCount);

			inline double alpha() const;
			inline void setAlpha(double alpha);

			inline double eta() const;
			inline void setEta(double eta);

			inline ArrayXXd lambda() const;
			inline void setLambda(const ArrayXXd& lambda);

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
				const Parameters& parameters = Parameters());

		private:
			int mNumDocuments;
			double mAlpha;
			double mEta;
			ArrayXXd mLambda;
			int mUpdateCounter;

			// adaptive learning rate parameters (Ranganath et al., 2013)
			double mAdaRho;
			double mAdaTau;
			double mAdaSqNorm;
			ArrayXXd mAdaGradient;
	};
}



inline int MDLDA::OnlineLDA::numDocuments() const {
	return mNumDocuments;
}



inline void MDLDA::OnlineLDA::setNumDocuments(int numDocuments) {
	if(numDocuments < 0)
		throw Exception("The number of documents should not be negative.");
	mNumDocuments = numDocuments;
}



inline int MDLDA::OnlineLDA::updateCount() const {
	return mUpdateCounter;
}



inline void MDLDA::OnlineLDA::setUpdateCount(int updateCount) {
	if(updateCount < 0)
		throw Exception("The update count not be negative.");
	mUpdateCounter = updateCount;
}



inline double MDLDA::OnlineLDA::alpha() const {
	return mAlpha;
}



inline void MDLDA::OnlineLDA::setAlpha(double alpha) {
	if(alpha < 0.)
		throw Exception("Alpha should not be negative.");
	mAlpha = alpha;
}



inline double MDLDA::OnlineLDA::eta() const {
	return mEta;
}



inline void MDLDA::OnlineLDA::setEta(double eta) {
	if(eta < 0.)
		throw Exception("Eta should not be negative.");
	mEta = eta;
}



inline Eigen::ArrayXXd MDLDA::OnlineLDA::lambda() const {
	return mLambda;
}



inline void MDLDA::OnlineLDA::setLambda(const ArrayXXd& lambda) {
	if(lambda.rows() != numTopics() || lambda.cols() != numWords())
		throw Exception("Lambda has wrong dimensionality.");
	mLambda = lambda;
}



inline int MDLDA::OnlineLDA::numTopics() const {
	return mLambda.rows();
}



inline int MDLDA::OnlineLDA::numWords() const {
	return mLambda.cols();
}

#endif
