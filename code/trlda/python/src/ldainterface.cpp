#include "onlineldainterface.h"

#include <new>
using std::bad_alloc;

#include <vector>
using std::vector;

#include <utility>
using std::pair;

#include <iostream>
using std::cout;
using std::endl;

#include "trlda/utils"
using TRLDA::Exception;

#include <sstream>
using std::stringstream;

#include <iomanip>
using std::setprecision;

#include <string>
using std::string;

#include "pyutils.h"

const char* LDA_doc = "Abstract base class.\n";

int LDA_init(LDAObject* self, PyObject* args, PyObject* kwds) {
	PyErr_SetString(PyExc_NotImplementedError, "This is an abstract class.");
	return -1;
}


PyObject* LDA_num_topics(LDAObject* self, void*) {
	return PyInt_FromLong(self->lda->numTopics());
}



PyObject* LDA_num_words(LDAObject* self, void*) {
	return PyInt_FromLong(self->lda->numWords());
}



PyObject* LDA_lambda(LDAObject* self, void*) {
	PyObject* array = PyArray_FromMatrixXd(self->lda->lambda());

	// make array immutable
	reinterpret_cast<PyArrayObject*>(array)->flags &= ~NPY_WRITEABLE;

	return array;
}



int LDA_set_lambda(LDAObject* self, PyObject* value, void*) {
	value = PyArray_FROM_OTF(value, NPY_DOUBLE, NPY_IN_ARRAY);

	if(!value) {
		PyErr_SetString(PyExc_TypeError, "Lambda should be of type `ndarray`.");
		return -1;
	}

	try {
		self->lda->setLambda(PyArray_ToMatrixXd(value));
	} catch(Exception exception) {
		Py_DECREF(value);
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return -1;
	}

	Py_DECREF(value);

	return 0;
}



PyObject* LDA_alpha(LDAObject* self, void*) {
	return PyArray_FromMatrixXd(self->lda->alpha());
}



int LDA_set_alpha(LDAObject* self, PyObject* alpha, void*) {
	try {
		if(PyFloat_Check(alpha)) {
			self->lda->setAlpha(PyFloat_AsDouble(alpha));
		} else if(PyInt_Check(alpha)) {
			self->lda->setAlpha(PyInt_AsLong(alpha));
		} else {
			alpha = PyArray_FROM_OTF(alpha, NPY_DOUBLE, NPY_IN_ARRAY);

			if(!alpha) {
				PyErr_SetString(PyExc_TypeError, "Alpha should be of type `ndarray`.");
				return -1;
			}

			MatrixXd alpha_ = PyArray_ToMatrixXd(alpha);

			if(alpha_.rows() == 1)
				alpha_ = alpha_.transpose();
			if(alpha_.cols() != 1) {
				PyErr_SetString(PyExc_TypeError, "Alpha should be one-dimensional.");
				return -1;
			}

			self->lda->setAlpha(alpha_);
		}
	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return -1;
	}

	return 0;
}



PyObject* LDA_eta(LDAObject* self, void*) {
	return PyFloat_FromDouble(self->lda->eta());
}



int LDA_set_eta(LDAObject* self, PyObject* value, void*) {
	double eta = PyFloat_AsDouble(value);

	if(PyErr_Occurred())
		return -1;

	try {
		self->lda->setEta(eta);
	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return -1;
	}

	return 0;
}



int PyList_ToDocuments(PyObject* docs, void* documents_) {
	LDA::Documents& documents = *reinterpret_cast<LDA::Documents*>(documents_);

	if(!PyList_Check(docs)) {
		PyErr_SetString(PyExc_TypeError, "Documents must be stored in a list.");
		return 0;
	}

	try {
		// create container for documents
		documents = LDA::Documents(PyList_Size(docs));

		// convert documents
		for(int i = 0; i < documents.size(); ++i) {
			PyObject* doc = PyList_GetItem(docs, i);

			// make sure document is a list
			if(!PyList_Check(doc)) {
				PyErr_SetString(PyExc_TypeError, "Each document must be a list of tuples.");
				return 0;
			}

			// create container for words
			documents[i] = LDA::Document(PyList_Size(doc));

			// load words
			for(int j = 0; j < documents[i].size(); ++j)
				if(!PyArg_ParseTuple(PyList_GetItem(doc, j), "ii",
					&documents[i][j].first,
					&documents[i][j].second))
					return 0;
		}
	} catch(bad_alloc&) {
		PyErr_SetString(PyExc_TypeError, "Not enough memory.");
		return 0;
	}

	return 1;
}



PyObject* PyList_FromDocuments(const LDA::Documents& documents) {
	PyObject* documents_ = PyList_New(0);

	for(int n = 0; n < documents.size(); ++n) {
		PyObject* document = PyList_New(0);

		for(int i = 0; i < documents[n].size(); ++i) {
			const int& wordID = documents[n][i].first;
			const int& wordCount = documents[n][i].second;

			PyObject* tuple = Py_BuildValue("(ii)", wordID, wordCount);
			PyList_Append(document, tuple);
			Py_DECREF(tuple);
		}

		PyList_Append(documents_, document);
		Py_DECREF(document);
	}

	return documents_;
}



const char* LDA_sample_doc =
	"";

PyObject* LDA_sample(
	LDAObject* self,
	PyObject* args,
	PyObject* kwds)
{
	const char* kwlist[] = {"num_documents", "length", 0};

	int num_documents;
	int length;

	// parse arguments
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "ii", const_cast<char**>(kwlist),
			&num_documents, &length))
		return 0;

	try {
		// return list of documents
		return PyList_FromDocuments(self->lda->sample(num_documents, length));
	} catch(Exception& exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}

	return 0;
}



const char* LDA_update_variables_doc =
	"";

PyObject* LDA_update_variables(
	LDAObject* self,
	PyObject* args,
	PyObject* kwds)
{
	const char* kwlist[] = {"docs", "latents", "inference_method", "max_iter", "threshold", "num_samples", "burn_in", 0};

	LDA::Documents documents;
	LDA::Parameters parameters;
	PyObject* latents = 0;
	const char* inference_method = 0;

	// parse arguments
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "O&|Osidii", const_cast<char**>(kwlist),
			&PyList_ToDocuments, &documents,
			&latents,
			&inference_method,
			&parameters.maxIterInference,
			&parameters.threshold,
			&parameters.numSamples,
			&parameters.burnIn))
		return 0;

	if(latents) {
		// make sure latents is a NumPy array
		latents = PyArray_FROM_OTF(latents, NPY_DOUBLE, NPY_IN_ARRAY);
		if(!latents) {
			PyErr_SetString(PyExc_TypeError, "`latents` should be of type `ndarray`.");
			return 0;
		}
	}

	if(inference_method) {
		switch(inference_method[0]) {
			case 'g':
			case 'G':
				parameters.inferenceMethod = LDA::GIBBS;
				break;

			case 'v':
			case 'V':
				parameters.inferenceMethod = LDA::VI;
				break;

			default:
				PyErr_SetString(PyExc_TypeError, "`inference_method` should be either 'gibbs' or 'vi'.");
				return 0;
		}
	}

	try {
		pair<ArrayXXd, ArrayXXd> results;

		if(latents)
			results = self->lda->updateVariables(
				documents,
				PyArray_ToMatrixXd(latents),
				parameters);
		else
			results = self->lda->updateVariables(documents, parameters);

		PyObject* rlatents = PyArray_FromMatrixXd(results.first);
		PyObject* sstats = PyArray_FromMatrixXd(results.second);
		PyObject* result = Py_BuildValue("(OO)", rlatents, sstats);

		Py_DECREF(rlatents);
		Py_DECREF(sstats);

		return result;

	} catch(Exception& exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		Py_XDECREF(latents);
		return 0;
	}

	Py_XDECREF(latents);

	return 0;
}



const char* LDA_lower_bound_doc =
	"Estimate lower bound for the given documents.";

PyObject* LDA_lower_bound(
	LDAObject* self,
	PyObject* args,
	PyObject* kwds)
{
	const char* kwlist[] = {"docs", "inference_method", "max_iter", "num_samples", "burn_in", 0};

	LDA::Documents documents;
	LDA::Parameters parameters;
	const char* inference_method = 0;

	// parse arguments
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "O&|Osiii", const_cast<char**>(kwlist),
			&PyList_ToDocuments, &documents,
			&inference_method,
			&parameters.maxIterInference,
			&parameters.numSamples,
			&parameters.burnIn))
		return 0;

	if(inference_method) {
		switch(inference_method[0]) {
			case 'g':
			case 'G':
				parameters.inferenceMethod = LDA::GIBBS;
				break;

			case 'v':
			case 'V':
				parameters.inferenceMethod = LDA::VI;
				break;

			default:
				PyErr_SetString(PyExc_TypeError, "`inference_method` should be either 'gibbs' or 'vi'.");
				return 0;
		}
	}

	try {
		return PyFloat_FromDouble(
			self->lda->lowerBound(documents, parameters));
	} catch(Exception& exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}

	return 0;
}



PyObject* LDA_str(PyObject* self_) {
	LDAObject* self = reinterpret_cast<LDAObject*>(self_);

	int numTopics = self->lda->numTopics();
	double eta = self->lda->eta();
	ArrayXXd alpha = self->lda->alpha();
	double alphaMax = alpha.maxCoeff();
	double alphaMin = alpha.minCoeff();

	stringstream strstr;

	strstr << "Number of topics: " << numTopics << "\n";
	strstr << setprecision(4) << "Eta: " << eta << "\n";
	strstr << setprecision(4) << "Alpha: " << alphaMin << ", " << alphaMax << " (min, max)\n";

	const string& str = strstr.str();
	return PyString_FromString(str.c_str());
}
