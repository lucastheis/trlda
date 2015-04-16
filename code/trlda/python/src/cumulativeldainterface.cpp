#include "cumulativeldainterface.h"

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

const char* CumulativeLDA_doc =
	"An implementation of SDA for LDA (see Broderick et al., 2013).\n"
	"\n"
	"Example:\n"
	"\n"
	"	>>> model = CumulativeLDA(num_words=7000, num_topics=100, alpha=.1, eta=.3)\n"
	"	>>> \n"
	"	>>> for documents in load_documents('data_train.mat', 1000):\n"
	"	>>>    model.update_parameters(documents, max_epochs=100)\n"
	"\n"
	"In contrast to L{OnlineLDA}, each document should be processed only once by "
	"L{update_parameters()}.\n"
	"\n"
	"@undocumented: __init__, __new__, __reduce__, __setstate__\n";

int CumulativeLDA_init(CumulativeLDAObject* self, PyObject* args, PyObject* kwds) {
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
			self->lda = new CumulativeLDA(num_words, num_topics, .1, eta);
		} else if(PyFloat_Check(alpha)) {
			self->lda = new CumulativeLDA(num_words, num_topics, PyFloat_AsDouble(alpha), eta);
		} else if(PyInt_Check(alpha)) {
			self->lda = new CumulativeLDA(num_words, num_topics, PyInt_AsLong(alpha), eta);
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

			self->lda = new CumulativeLDA(num_words, alpha_, eta);
		}
	} catch(Exception& exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
	}

	return 0;
}



const char* CumulativeLDA_update_parameters_doc =
	"update_parameters(docs, max_epochs=100, max_iter_inference=100, **kwargs)\n"
	"\n"
	"Updates beliefs over parameters.\n"
	"\n"
	"@type  docs: C{list}\n"
	"@param docs: a batch of documents\n"
	"\n"
	"@type  max_epochs: C{int}\n"
	"@param max_epochs: number of repeated updates to parameters and hyperparameters\n"
	"\n"
	"@type  max_iter_inference: C{int}\n"
	"@param max_iter_inference: number of variational inference steps per iteration\n"
	"\n"
	"@type  max_iter_alpha: C{int}\n"
	"@param max_iter_alpha: number of Newton steps applied to $\\boldsymbo{\\alpha}$ per iteration\n"
	"\n"
	"@type  update_lambda: C{bool}\n"
	"@param update_lambda: if C{False}, don't update beliefs over topics, $\\boldsymbol{\\lambda}$ (default: True)\n"
	"\n"
	"@type  update_alpha: C{bool}\n"
	"@param update_alpha: if True, update $\\boldsymbol{\\alpha}$ via empirical Bayes (default: False)\n"
	"\n"
	"@type  min_alpha: C{float}\n"
	"@param min_alpha: constrain the $\\alpha_k$ to be at least this large (default: 1e-6)\n"
	"\n"
	"@type  emp_bayes_threshold: C{float}\n"
	"@param emp_bayes_threshold: used to stop empirical Bayes updates when parameters don't change (default: 1e-8)\n"
	"\n"
	"@type  verbosity: C{int}\n"
	"@param verbosity: controls how many messages are printed";

PyObject* CumulativeLDA_update_parameters(
	CumulativeLDAObject* self,
	PyObject* args,
	PyObject* kwds)
{
	const char* kwlist[] = {
		"docs",
		"max_epochs",
		"max_iter_inference",
		"max_iter_alpha",
		"update_lambda",
		"update_alpha",
		"min_alpha",
		"emp_bayes_threshold",
		"inference_threshold",
		"verbosity", 0};

	CumulativeLDA::Documents documents;
	CumulativeLDA::Parameters parameters;

	// parse arguments
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "O&|iiibbdddi", const_cast<char**>(kwlist),
			&PyList_ToDocuments, &documents,
			&parameters.maxEpochs,
			&parameters.maxIterInference,
			&parameters.maxIterAlpha,
			&parameters.updateLambda,
			&parameters.updateAlpha,
			&parameters.minAlpha,
			&parameters.empBayesThreshold,
			&parameters.threshold,
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



const char* CumulativeLDA_reduce_doc =
	"__reduce__(self)\n"
	"\n"
	"Method used by Pickle.";

PyObject* CumulativeLDA_reduce(CumulativeLDAObject* self, PyObject*) {
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



const char* CumulativeLDA_setstate_doc =
	"__setstate__(self)\n"
	"\n"
	"Method used by Pickle.";

PyObject* CumulativeLDA_setstate(CumulativeLDAObject* self, PyObject* state) {
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
