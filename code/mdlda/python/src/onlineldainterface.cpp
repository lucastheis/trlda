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

#include "mdlda/utils"
using MDLDA::Exception;

#include "pyutils.h"

const char* OnlineLDA_doc =
	"An implementation of mirror descent for latent dirichlet allocation (LDA).";

int OnlineLDA_init(OnlineLDAObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {
		"num_words",
		"num_topics",
		"num_documents",
		"alpha",
		"eta",
		"kappa_",
		"tau_", 0};

	int num_words;
	int num_topics;
	int num_documents;
	double alpha = .1;
	double eta = .3;

	// needed to support opening of older versions of pickled OnlineLDA objects
	double kappa_ = 0.;
	double tau_ = 0.;

	if(!PyArg_ParseTupleAndKeywords(args, kwds, "iii|dddd", const_cast<char**>(kwlist),
			&num_words, &num_topics, &num_documents, &alpha, &eta, &kappa_, &tau_))
		return -1;

	try {
		self->lda = new OnlineLDA(num_words, num_topics, num_documents, alpha, eta);
	} catch(Exception& exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
	}

	return 0;
}



PyObject* OnlineLDA_num_topics(OnlineLDAObject* self, void*) {
	return PyInt_FromLong(self->lda->numTopics());
}



PyObject* OnlineLDA_num_words(OnlineLDAObject* self, void*) {
	return PyInt_FromLong(self->lda->numWords());
}



PyObject* OnlineLDA_num_documents(OnlineLDAObject* self, void*) {
	return PyFloat_FromDouble(self->lda->numDocuments());
}



int OnlineLDA_set_num_documents(OnlineLDAObject* self, PyObject* value, void*) {
	int num_documents = PyInt_AsLong(value);

	if(PyErr_Occurred())
		return -1;

	try {
		self->lda->setNumDocuments(num_documents);
	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return -1;
	}

	return 0;
}




PyObject* OnlineLDA_lambda(OnlineLDAObject* self, void*) {
	PyObject* array = PyArray_FromMatrixXd(self->lda->lambda());

	// make array immutable
	reinterpret_cast<PyArrayObject*>(array)->flags &= ~NPY_WRITEABLE;

	return array;
}



int OnlineLDA_set_lambda(OnlineLDAObject* self, PyObject* value, void*) {
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



PyObject* OnlineLDA_alpha(OnlineLDAObject* self, void*) {
	return PyFloat_FromDouble(self->lda->alpha());
}



int OnlineLDA_set_alpha(OnlineLDAObject* self, PyObject* value, void*) {
	double alpha = PyFloat_AsDouble(value);

	if(PyErr_Occurred())
		return -1;

	try {
		self->lda->setAlpha(alpha);
	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return -1;
	}

	return 0;
}



PyObject* OnlineLDA_eta(OnlineLDAObject* self, void*) {
	return PyFloat_FromDouble(self->lda->eta());
}



int OnlineLDA_set_eta(OnlineLDAObject* self, PyObject* value, void*) {
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
	OnlineLDA::Documents& documents = *reinterpret_cast<OnlineLDA::Documents*>(documents_);

	if(!PyList_Check(docs)) {
		PyErr_SetString(PyExc_TypeError, "Documents must be stored in a list.");
		return 0;
	}

	try {
		// create container for documents
		documents = OnlineLDA::Documents(PyList_Size(docs));

		// convert documents
		for(int i = 0; i < documents.size(); ++i) {
			PyObject* doc = PyList_GetItem(docs, i);

			// make sure document is a list
			if(!PyList_Check(doc)) {
				PyErr_SetString(PyExc_TypeError, "Each document must be a list of tuples.");
				return 0;
			}

			// create container for words
			documents[i] = OnlineLDA::Document(PyList_Size(doc));

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



const char* OnlineLDA_update_variables_doc =
	"";

PyObject* OnlineLDA_update_variables(
	OnlineLDAObject* self,
	PyObject* args,
	PyObject* kwds)
{
	const char* kwlist[] = {"docs", "max_iter", "gamma", 0};

	OnlineLDA::Documents documents;
	OnlineLDA::Parameters parameters;
	PyObject* gamma = 0;

	// parse arguments
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "O&|iO", const_cast<char**>(kwlist),
			&PyList_ToDocuments, &documents, &parameters.maxIterInference, &gamma))
		return 0;

	if(gamma) {
		// make sure gamma is a NumPy array
		gamma = PyArray_FROM_OTF(gamma, NPY_DOUBLE, NPY_IN_ARRAY);
		if(!gamma) {
			PyErr_SetString(PyExc_TypeError, "`gamma` should be of type `ndarray`.");
			return 0;
		}
	}

	try {
		pair<ArrayXXd, ArrayXXd> results;

		if(gamma)
			results = self->lda->updateVariables(
				documents,
				PyArray_ToMatrixXd(gamma),
				parameters);
		else
			results = self->lda->updateVariables(documents, parameters);

		PyObject* rgamma = PyArray_FromMatrixXd(results.first);
		PyObject* sstats = PyArray_FromMatrixXd(results.second);
		PyObject* result = Py_BuildValue("(OO)", rgamma, sstats);

		Py_DECREF(rgamma);
		Py_DECREF(sstats);

		return result;

	} catch(Exception& exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		Py_XDECREF(gamma);
		return 0;
	}

	Py_XDECREF(gamma);

	return 0;
}



const char* OnlineLDA_update_parameters_doc =
	"";

PyObject* OnlineLDA_update_parameters(
	OnlineLDAObject* self,
	PyObject* args,
	PyObject* kwds)
{
	const char* kwlist[] = {"docs", "max_iter", "kappa", "tau", "rho", "adaptive", 0};

	OnlineLDA::Documents documents;
	OnlineLDA::Parameters parameters;

	// parse arguments
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "O&|idddb", const_cast<char**>(kwlist),
			&PyList_ToDocuments, &documents,
			&parameters.maxIterMD,
			&parameters.kappa,
			&parameters.tau,
			&parameters.rho,
			&parameters.adaptive))
		return 0;

	try {
		// return learning rate used
		return PyFloat_FromDouble(self->lda->updateParameters(documents, parameters));
	} catch(Exception& exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}

	return 0;
}



const char* OnlineLDA_reduce_doc =
	"__reduce__(self)\n"
	"\n"
	"Method used by Pickle.";

PyObject* OnlineLDA_reduce(OnlineLDAObject* self, PyObject*) {
	// constructor arguments
	PyObject* args = Py_BuildValue("(iiidd)",
		self->lda->numWords(),
		self->lda->numTopics(),
		self->lda->numDocuments(),
		self->lda->alpha(),
		self->lda->eta());

	PyObject* lambda = OnlineLDA_lambda(self, 0);
	PyObject* state = Py_BuildValue("(Oi)", lambda, self->lda->updateCount());
	PyObject* result = Py_BuildValue("(OOO)", Py_TYPE(self), args, state);

	Py_DECREF(lambda);
	Py_DECREF(args);
	Py_DECREF(state);

	return result;
}



const char* OnlineLDA_setstate_doc =
	"__setstate__(self)\n"
	"\n"
	"Method used by Pickle.";

PyObject* OnlineLDA_setstate(OnlineLDAObject* self, PyObject* state) {
	PyObject* lambda;
	int updateCount;

	if(!PyArg_ParseTuple(state, "(Oi)", &lambda, &updateCount))
		return 0;

	try {
		OnlineLDA_set_lambda(self, lambda, 0);
		self->lda->setUpdateCount(updateCount);
	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}

	Py_INCREF(Py_None);
	return Py_None;
}
