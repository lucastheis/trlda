#ifndef TRLDA_EXCEPTION_H
#define TRLDA_EXCEPTION_H

namespace TRLDA {
	class Exception {
		public:
			inline Exception(const char* message = "");

			inline const char* message();

		protected:
			const char* mMessage;
	};
}


inline TRLDA::Exception::Exception(const char* message) : mMessage(message) {
}



inline const char* TRLDA::Exception::message() {
	return mMessage;
}

#endif
