#include "batchldainterface.h"

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

const char* BatchLDA_doc =
	"An implementation of an online trust region method for latent dirichlet allocation (LDA).";

int BatchLDA_init(BatchLDAObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {
		"num_words",
		"num_topics",
		"alpha",
		"eta", 0};

	int num_words;
	int num_topics;
	PyObject* alpha = 0;
	double eta = .3;

	if(!PyArg_ParseTupleAndKeywords(args, kwds, "ii|Od", const_cast<char**>(kwlist),
			&num_words, &num_topics, &alpha, &eta))
		return -1;

	try {
		if(alpha == 0) {
			self->lda = new BatchLDA(num_words, num_topics, .1, eta);
		} else if(PyFloat_Check(alpha)) {
			self->lda = new BatchLDA(num_words, num_topics, PyFloat_AsDouble(alpha), eta);
		} else if(PyInt_Check(alpha)) {
			self->lda = new BatchLDA(num_words, num_topics, PyInt_AsLong(alpha), eta);
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

			self->lda = new BatchLDA(num_words, alpha_, eta);
		}
	} catch(Exception& exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
	}

	return 0;
}



const char* BatchLDA_update_parameters_doc =
	"";

PyObject* BatchLDA_update_parameters(
	BatchLDAObject* self,
	PyObject* args,
	PyObject* kwds)
{
	const char* kwlist[] = {
		"docs",
		"max_epochs",
		"max_iter_inference",
		"max_iter_alpha",
		"max_iter_eta",
		"update_lambda",
		"update_alpha",
		"update_eta",
		"min_alpha",
		"min_eta", 0};

	BatchLDA::Documents documents;
	BatchLDA::Parameters parameters;

	// parse arguments
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "O&|iiiibbbdd", const_cast<char**>(kwlist),
			&PyList_ToDocuments, &documents,
			&parameters.maxEpochs,
			&parameters.maxIterInference,
			&parameters.maxIterAlpha,
			&parameters.maxIterEta,
			&parameters.updateLambda,
			&parameters.updateAlpha,
			&parameters.updateEta,
			&parameters.minAlpha,
			&parameters.minEta))
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



const char* BatchLDA_reduce_doc =
	"__reduce__(self)\n"
	"\n"
	"Method used by Pickle.";

PyObject* BatchLDA_reduce(BatchLDAObject* self, PyObject*) {
	PyObject* alpha = PyArray_FromMatrixXd(self->lda->alpha());

	// constructor arguments
	PyObject* args = Py_BuildValue("(iiOd)",
		self->lda->numWords(),
		self->lda->numTopics(),
		alpha,
		self->lda->eta());

	Py_DECREF(alpha);

	PyObject* lambda = LDA_lambda(reinterpret_cast<LDAObject*>(self), 0);
	PyObject* state = Py_BuildValue("(O)", lambda);
	PyObject* result = Py_BuildValue("(OOO)", Py_TYPE(self), args, state);

	Py_DECREF(lambda);
	Py_DECREF(args);
	Py_DECREF(state);

	return result;
}



const char* BatchLDA_setstate_doc =
	"__setstate__(self)\n"
	"\n"
	"Method used by Pickle.";

PyObject* BatchLDA_setstate(BatchLDAObject* self, PyObject* state) {
	PyObject* lambda;
	int updateCount;

	if(!PyArg_ParseTuple(state, "(O)", &lambda))
		return 0;

	try {
		LDA_set_lambda(reinterpret_cast<LDAObject*>(self), lambda, 0);
	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}

	Py_INCREF(Py_None);
	return Py_None;
}
