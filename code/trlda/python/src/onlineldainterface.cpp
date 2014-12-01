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

#include "pyutils.h"

const char* OnlineLDA_doc =
	"An implementation of an online trust region method for latent dirichlet allocation (LDA).";

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
	PyObject* alpha = 0;
	double eta = .3;

	// needed to support opening of old versions of pickled OnlineLDA objects
	double kappa_ = 0.;
	double tau_ = 0.;

	if(!PyArg_ParseTupleAndKeywords(args, kwds, "iii|Oddd", const_cast<char**>(kwlist),
			&num_words, &num_topics, &num_documents, &alpha, &eta, &kappa_, &tau_))
		return -1;

	try {
		if(alpha == 0) {
			self->lda = new OnlineLDA(num_words, num_topics, num_documents, .1, eta);
		} else if(PyFloat_Check(alpha)) {
			self->lda = new OnlineLDA(num_words, num_topics, num_documents, PyFloat_AsDouble(alpha), eta);
		} else if(PyInt_Check(alpha)) {
			self->lda = new OnlineLDA(num_words, num_topics, num_documents, PyInt_AsLong(alpha), eta);
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

			self->lda = new OnlineLDA(num_words, num_documents, alpha_, eta);
		}
	} catch(Exception& exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
	}

	return 0;
}



PyObject* OnlineLDA_num_documents(OnlineLDAObject* self, void*) {
	return PyInt_FromLong(self->lda->numDocuments());
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




const char* OnlineLDA_update_parameters_doc =
	"";

PyObject* OnlineLDA_update_parameters(
	OnlineLDAObject* self,
	PyObject* args,
	PyObject* kwds)
{
	const char* kwlist[] = {
		"docs",
		"max_iter_tr",
		"max_iter_inference",
		"kappa",
		"tau",
		"rho",
		"adaptive",
		"init_gamma",
		"update_lambda",
		"update_alpha",
		"update_eta",
		"min_alpha",
		"min_eta",
		"verbosity", 0};

	OnlineLDA::Documents documents;
	OnlineLDA::Parameters parameters;

	// parse arguments
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "O&|iidddbbbbbddi", const_cast<char**>(kwlist),
			&PyList_ToDocuments, &documents,
			&parameters.maxIterTR,
			&parameters.maxIterInference,
			&parameters.kappa,
			&parameters.tau,
			&parameters.rho,
			&parameters.adaptive,
			&parameters.initGamma,
			&parameters.updateLambda,
			&parameters.updateAlpha,
			&parameters.updateEta,
			&parameters.minAlpha,
			&parameters.minEta,
			&parameters.verbosity))
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
	PyObject* alpha = PyArray_FromMatrixXd(self->lda->alpha());

	// constructor arguments
	PyObject* args = Py_BuildValue("(iiiOd)",
		self->lda->numWords(),
		self->lda->numTopics(),
		self->lda->numDocuments(),
		alpha,
		self->lda->eta());

	Py_DECREF(alpha);

	PyObject* lambda = LDA_lambda(reinterpret_cast<LDAObject*>(self), 0);
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
		LDA_set_lambda(reinterpret_cast<LDAObject*>(self), lambda, 0);
		self->lda->setUpdateCount(updateCount);
	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}

	Py_INCREF(Py_None);
	return Py_None;
}
