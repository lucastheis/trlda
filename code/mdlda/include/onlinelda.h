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

	using Eigen::ArrayXXd;

	class OnlineLDA : public Distribution {
		public:
			typedef pair<int, int> Word;
			typedef vector<Word> Document;
			typedef vector<Document> Documents;

			OnlineLDA(
				int numWords,
				int numTopics,
				int numDocuments,
				double alpha = .1,
				double eta = .3,
				double tau = 1024.,
				double kappa = .9);

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

			inline double tau() const;
			inline void setTau(double tau);

			inline double kappa() const;
			inline void setKappa(double kappa);

			inline ArrayXXd lambda() const;
			inline void setLambda(const ArrayXXd& lambda);

			virtual pair<ArrayXXd, ArrayXXd> updateVariables(
				const Documents& documents,
				int maxIter = 100,
				double threhsold = 0.001) const;
			virtual pair<ArrayXXd, ArrayXXd> updateVariables(
				const Documents& documents,
				const ArrayXXd& initialGamma,
				int maxIter = 100,
				double threshold = 0.001) const;

			virtual bool updateParameters(
				const Documents& documents,
				int maxIter = 20,
				double rho = -1.);

		private:
			int mNumDocuments;
			double mAlpha;
			double mEta;
			double mTau;
			double mKappa;
			ArrayXXd mLambda;
			int mUpdateCounter;
	};
}



inline int MDLDA::OnlineLDA::numDocuments() const {
	return mAlpha;
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



inline double MDLDA::OnlineLDA::tau() const {
	return mTau;
}



inline void MDLDA::OnlineLDA::setTau(double tau) {
	if(tau < 0.)
		throw Exception("Tau should not be negative.");
	mTau = tau;
}



inline double MDLDA::OnlineLDA::kappa() const {
	return mKappa;
}



inline void MDLDA::OnlineLDA::setKappa(double kappa) {
	if(kappa < 0.)
		throw Exception("Kappa should not be negative.");
	mKappa = kappa;
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
