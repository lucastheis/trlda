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
	"An implementation of an online trust region method for latent Dirichlet allocation.\n"
	"\n"
	"\t>>> model = OnlineLDA(\n"
	"\t\tnum_words=7000,\n"
	"\t\tnum_topics=100,\n"
	"\t\tnum_documents=10000,\n"
	"\t\talpha=.1,\n"
	"\t\teta=.3)\n"
	"\n"
	"C{alpha} can be a scalar or an array with one entry for each topic.\n"
	"\n"
	"@undocumented: __new__, __init__, __reduce__, __setstate__";

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



PyObject* OnlineLDA_update_count(OnlineLDAObject* self, void*) {
	return PyInt_FromLong(self->lda->updateCount());
}



int OnlineLDA_set_update_count(OnlineLDAObject* self, PyObject* value, void*) {
	int update_count = PyInt_AsLong(value);

	if(PyErr_Occurred())
		return -1;

	try {
		self->lda->setUpdateCount(update_count);
	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return -1;
	}

	return 0;
}




const char* OnlineLDA_update_parameters_doc =
	"update_parameters(docs, max_iter_tr=10, max_iter_inference=20, kappa=.7, tau=100, **kwargs)\n"
	"\n"
	"Updates beliefs over parameters.\n"
	"\n"
	"Set C{max_iter_tr} to zero to perform the standard natural gradient step of stochastic variational "
	"inference (in this case increase C{max_iter_inference}).\n"
	"\n"
	"By default, the learning rate is automatically set to\n"
	"\n"
	"$$\\rho_t = (\\tau + t)^{-\\kappa},$$\n"
	"\n"
	"where $t$ is the number of calls to this function.\n"
	"\n"
	"@type  docs: C{list}\n"
	"@param docs: a batch of documents\n"
	"\n"
	"@type  max_iter_tr: C{int}\n"
	"@param max_iter_tr: number of steps in trust-region optimization\n"
	"\n"
	"@type  max_iter_inference: C{int}\n"
	"@param max_iter_inference: number of variational inference steps per trust-region step\n"
	"\n"
	"@type  kappa: C{float}\n"
	"@param kappa: controls the learning rate decay\n"
	"\n"
	"@type  tau: C{float}\n"
	"@param tau: decreases intial learning rates\n"
	"\n"
	"@type  rho: C{float}\n"
	"@param rho: can be used to manually set the learning rate\n"
	"\n"
	"@type  adaptive: C{bool}\n"
	"@param adaptive: automatically adapt the learning rate (see Ranganath et al., 2013)\n"
	"\n"
	"@type  init_gamma: C{bool}\n"
	"@param init_gamma: initialize beliefs over $\\boldsymbol{\\theta}$ with beliefs of previous trust-region step (default: True)\n"
	"\n"
	"@type  update_lambda: C{bool}\n"
	"@param update_lambda: if C{False}, don't update beliefs over topics, $\\boldsymbol{\\lambda}$ (default: True)\n"
	"\n"
	"@type  update_alpha: C{bool}\n"
	"@param update_alpha: if True, update $\\boldsymbol{\\alpha}$ via empirical Bayes (default: False)\n"
	"\n"
	"@type  update_eta: C{bool}\n"
	"@param update_eta: if True, update $\\eta$ via empirical Bayes (default: False)\n"
	"\n"
	"@type  min_alpha: C{float}\n"
	"@param min_alpha: constrain the $\\alpha_k$ to be at least this large (default: 1e-6)\n"
	"\n"
	"@type  min_eta: C{float}\n"
	"@param min_eta: constrain $\\eta$ to be at least this large (default: 1e-6)\n"
	"\n"
	"@type  verbosity: C{int}\n"
	"@param verbosity: controls how many messages are printed\n"
	"\n"
	"@rtype: C{float}\n"
	"@return: the learning rate used in this update\n"
	"\n"
	"@seealso: L{update_count}";

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
	parameters.maxIterInference = 20;

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
