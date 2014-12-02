#ifndef TRLDA_CUMULATIVELDA_H
#define TRLDA_CUMULATIVELDA_H

#include "lda.h"

namespace TRLDA {
	class CumulativeLDA : public LDA {
		public:
			CumulativeLDA(
				int numWords,
				int numTopics,
				double alpha = .1,
				double eta = .3);
			CumulativeLDA(
				int numWords,
				ArrayXd alpha,
				double eta = .3);

			virtual double updateParameters(
				const Documents& documents,
				const Parameters& parameters = Parameters());

		protected:
			ArrayXd mPsiGammaDiff;
	};
}

#endif
