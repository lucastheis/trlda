#ifndef MDLDA_EXCEPTION_H
#define MDLDA_EXCEPTION_H

namespace MDLDA {
	class Exception {
		public:
			inline Exception(const char* message = "");

			inline const char* message();

		protected:
			const char* mMessage;
	};
}


inline MDLDA::Exception::Exception(const char* message) : mMessage(message) {
}



inline const char* MDLDA::Exception::message() {
	return mMessage;
}

#endif
