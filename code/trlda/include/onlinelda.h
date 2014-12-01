#ifndef TRLDA_ONLINELDA_H
#define TRLDA_ONLINELDA_H

#include "lda.h"

namespace TRLDA {
	class OnlineLDA : public LDA {
		public:
			OnlineLDA(
				int numWords,
				int numTopics,
				int numDocuments,
				double alpha = .1,
				double eta = .3);
			OnlineLDA(
				int numWords,
				int numDocuments,
				ArrayXd alpha,
				double eta = .3);

			inline int numDocuments() const;
			inline void setNumDocuments(int numDocuments);

			inline int updateCount() const;
			inline void setUpdateCount(int updateCount);

			virtual double updateParameters(
				const Documents& documents,
				const Parameters& parameters = Parameters());

		private:
			int mNumDocuments;
			int mUpdateCounter;

			// adaptive learning rate parameters (Ranganath et al., 2013)
			double mAdaRho;
			double mAdaTau;
			double mAdaSqNorm;
			ArrayXXd mAdaGradient;
	};
}



inline int TRLDA::OnlineLDA::numDocuments() const {
	return mNumDocuments;
}



inline void TRLDA::OnlineLDA::setNumDocuments(int numDocuments) {
	if(numDocuments < 0)
		throw Exception("The number of documents should not be negative.");
	mNumDocuments = numDocuments;
}



inline int TRLDA::OnlineLDA::updateCount() const {
	return mUpdateCounter;
}



inline void TRLDA::OnlineLDA::setUpdateCount(int updateCount) {
	if(updateCount < 0)
		throw Exception("The update count should not be negative.");
	mUpdateCounter = updateCount;
}

#endif
