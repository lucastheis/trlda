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

const char* LDA_doc = 
	"Abstract base class.\n"
	"\n"
	"@undocumented: __init__, __new__, __str__\n";

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
	"sample(self, num_documents, length)\n"
	"\n"
	"Samples a specified number of documents from the model.\n"
	"\n"
	"Topics ($\\boldsymbol{\\beta}$) are first sampled from the current Dirichlet beliefs over topics. "
	"This is done only once per call to C{sample} and all documents are sampled conditioned on "
	"these topics. The length of the documents is sampled from a Poisson distribution "
	"where the rate (average length) is given by C{length}. Documents of "
	"length zero are possible.\n"
	"\n"
	"Words are represented as tuples of a word ID and a word count. All generated word counts will "
	"be 1, but words can occur multiple times in a document, e.g., C{[(12, 1), (4, 1), (12, 1)]}.\n"
	"\n"
	"@type  num_documents: C{int}\n"
	"@param num_documents: number of documents to sample\n"
	"\n"
	"@type  length: C{int}\n"
	"@param length: average length of the sampled documents\n"
	"\n"
	"@rtype: C{list}\n"
	"@return: a list of documents, where each document is a list of tuples";

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
	"update_variables(docs, latents=None, inference_method='VI', max_iter=100, threshold=0.001, num_samples=1, burn_in=2)\n"
	"\n"
	"Computes beliefs over topic assignments ($z_{di}$) for the given documents.\n"
	"\n"
	"The beliefs may be estimated via mean-field variational inference ('VI') "
	"or collapsed Gibbs sampling ('GIBBS'). The method returns a tuple of a "
	"$K$-dimensional column vector and a $W \\times K$-dimensional matrix of sufficient "
	"statistics. In the case of variational inference, the vector represents the Dirichlet "
	"beliefs over the distribution of topics ($\\boldsymbol{\\theta}$) while for Gibbs sampling it "
	"represents a sample of $\\boldsymbol{\\theta}$ conditioned on the sampled topic assignments $\\mathbf{z}$. "
	"This can be used to initialize the algorithm in a later call to C{update_variables} "
	"via C{latents}. The matrix of sufficient statistics indicates the expected number of "
	"occurrences of words with topics in the given set of documents.\n"
	"\n"
	"Each document should be represented as a list of words, where each word is a tuple\n"
	"of a word ID and a word count.\n"
	"\n"
	"@type  docs: C{list}\n"
	"@param docs: a set of documents for which to perform inference\n"
	"\n"
	"@type  latents: C{ndarray}\n"
	"@param latents: can be used to initialize beliefs over $\\boldsymbol{\\theta}$\n"
	"\n"
	"@type  inference_method: C{str}\n"
	"@param inference_method: either 'VI' or 'GIBBS'\n"
	"\n"
	"@type  max_iter: C{int}\n"
	"@param max_iter: maximum number of belief updates in variational inference\n"
	"\n"
	"@type  threshold: C{float}\n"
	"@param threshold: if the average change in beliefs over $\\boldsymbol{\\theta}$ is smaller than this, stop iterations\n"
	"\n"
	"@type  num_samples: C{int}\n"
	"@param num_samples: number of samples used to estimate expected word/topic occurences\n"
	"\n"
	"@type  burn_in: C{int}\n"
	"@param burn_in: number of MCMC updates performed before starting to collect samples\n"
	"\n"
	"@rtype: C{tuple}\n"
	"@return: a tuple of beliefs over $\\boldsymbol{\\theta}$$ and sufficient statistics";

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
				PyErr_SetString(PyExc_TypeError, "`inference_method` should be either 'GIBBS' or 'VI'.");
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
	"lower_bound(docs, num_documents=-1, inference_method='VI', max_iter=100, num_samples=1, burn_in=2)\n"
	"\n"
	"Estimate lower bound, $\\mathcal{L}(\\boldsymbol{\\lambda})$, for the given set of documents.\n"
	"\n"
	"@type  docs: C{list}\n"
	"@param docs: a set of documents for which to perform inference\n"
	"\n"
	"@type  num_documents: C{int}\n"
	"@param num_documents: can be used to target a lower bound with a different number of documents\n"
	"\n"
	"@type  inference_method: C{str}\n"
	"@param inference_method: either 'VI' or 'GIBBS'\n"
	"\n"
	"@type  max_iter: C{int}\n"
	"@param max_iter: maximum number of belief updates in variational inference\n"
	"\n"
	"@type  num_samples: C{int}\n"
	"@param num_samples: number of samples used to estimate expected word/topic occurences\n"
	"\n"
	"@type  burn_in: C{int}\n"
	"@param burn_in: number of sampling steps performed before starting to collect samples\n"
	"\n"
	"@rtype: C{float}\n"
	"@return: estimate of the lower bound";

PyObject* LDA_lower_bound(
	LDAObject* self,
	PyObject* args,
	PyObject* kwds)
{
	const char* kwlist[] = {"docs", "num_documents", "inference_method", "max_iter", "num_samples", "burn_in", 0};

	LDA::Documents documents;
	LDA::Parameters parameters;
	int num_documents = -1;
	const char* inference_method = 0;

	// parse arguments
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "O&|isiii", const_cast<char**>(kwlist),
			&PyList_ToDocuments, &documents,
			&num_documents,
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
				PyErr_SetString(PyExc_TypeError, "`inference_method` should be either 'GIBBS' or 'VI'.");
				return 0;
		}
	}

	try {
		return PyFloat_FromDouble(
			self->lda->lowerBound(documents, parameters, num_documents));
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
