#ifndef TRLDA_BATCHLDA_H
#define TRLDA_BATCHLDA_H

#include "lda.h"

namespace TRLDA {
	class BatchLDA : public LDA {
		public:
			BatchLDA(
				int numWords,
				int numTopics,
				double alpha = .1,
				double eta = .3);
			BatchLDA(
				int numWords,
				ArrayXd alpha,
				double eta = .3);

			virtual double updateParameters(
				const Documents& documents,
				const Parameters& parameters = Parameters());
	};
}

#endif
