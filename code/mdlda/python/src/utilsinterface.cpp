#include "utilsinterface.h"

#include "mdlda/utils"
using MDLDA::Exception;
using MDLDA::randomSelect;
using MDLDA::sampleDirichlet;

#include <set>
using std::set;

#include "pyutils.h"

const char* random_select_doc =
	"random_select(k, n)\n"
	"\n"
	"Randomly selects $k$ out of $n$ elements.\n"
	"\n"
	"@type  k: C{int}\n"
	"@param k: the number of elements to pick\n"
	"\n"
	"@type  n: C{int}\n"
	"@param n: the number of elements to pick from\n"
	"\n"
	"@rtype: C{list}\n"
	"@return: a list of $k$ indices";

PyObject* random_select(PyObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {"k", "n", 0};

	int n;
	int k;

	if(!PyArg_ParseTupleAndKeywords(args, kwds, "ii", const_cast<char**>(kwlist), &k, &n))
		return 0;

	try {
		set<int> indices = randomSelect(k, n);

		PyObject* list = PyList_New(indices.size());
		
		int i = 0;

		for(set<int>::iterator iter = indices.begin(); iter != indices.end(); ++iter, ++i)
			PyList_SetItem(list, i, PyInt_FromLong(*iter));

		return list;
	} catch(Exception& exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}
	
	return 0;
}



const char* sample_dirichlet_doc =
	"";

PyObject* sample_dirichlet(PyObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {"m", "n", "alpha", 0};

	int m;
	int n;
	double alpha;

	if(!PyArg_ParseTupleAndKeywords(args, kwds, "iid", const_cast<char**>(kwlist),
		&m, &n, &alpha))
		return 0;

	try {
		return PyArray_FromMatrixXd(sampleDirichlet(m, n, alpha));
	} catch(Exception& exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}
	
	return 0;
}
