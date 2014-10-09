#define PY_ARRAY_UNIQUE_SYMBOL MDLDA_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <arrayobject.h>
#include <stdlib.h>
#include <sys/time.h>
#include "distributioninterface.h"
#include "onlineldainterface.h"
#include "utilsinterface.h"
#include "Eigen/Core"

static PyGetSetDef Distribution_getset[] = {
	{0}
};

static PyMethodDef Distribution_methods[] = {
	{0}
};

PyTypeObject Distribution_type = {
	PyObject_HEAD_INIT(0)
	0,                                    /*ob_size*/
	"mdlda.models.Distribution",          /*tp_name*/
	sizeof(DistributionObject),           /*tp_basicsize*/
	0,                                    /*tp_itemsize*/
	(destructor)Distribution_dealloc,     /*tp_dealloc*/
	0,                                    /*tp_print*/
	0,                                    /*tp_getattr*/
	0,                                    /*tp_setattr*/
	0,                                    /*tp_compare*/
	0,                                    /*tp_repr*/
	0,                                    /*tp_as_number*/
	0,                                    /*tp_as_sequence*/
	0,                                    /*tp_as_mapping*/
	0,                                    /*tp_hash */
	0,                                    /*tp_call*/
	0,                                    /*tp_str*/
	0,                                    /*tp_getattro*/
	0,                                    /*tp_setattro*/
	0,                                    /*tp_as_buffer*/
	Py_TPFLAGS_DEFAULT,                   /*tp_flags*/
	Distribution_doc,                     /*tp_doc*/
	0,                                    /*tp_traverse*/
	0,                                    /*tp_clear*/
	0,                                    /*tp_richcompare*/
	0,                                    /*tp_weaklistoffset*/
	0,                                    /*tp_iter*/
	0,                                    /*tp_iternext*/
	Distribution_methods,                 /*tp_methods*/
	0,                                    /*tp_members*/
	Distribution_getset,                  /*tp_getset*/
	0,                                    /*tp_base*/
	0,                                    /*tp_dict*/
	0,                                    /*tp_descr_get*/
	0,                                    /*tp_descr_set*/
	0,                                    /*tp_dictoffset*/
	(initproc)Distribution_init,          /*tp_init*/
	0,                                    /*tp_alloc*/
	Distribution_new,                     /*tp_new*/
};

static PyGetSetDef OnlineLDA_getset[] = {
	{"num_topics",
		(getter)OnlineLDA_num_topics,
		0,
		"Number of topics."},
	{"num_words",
		(getter)OnlineLDA_num_words,
		0,
		"Number of words."},
	{"num_documents",
		(getter)OnlineLDA_num_documents,
		(setter)OnlineLDA_set_num_documents,
		"Number of documents of the (hypothetical) full dataset."},
	{"lambdas",
		(getter)OnlineLDA_lambda,
		(setter)OnlineLDA_set_lambda,
		"Parameters governing beliefs over topics, $\\beta_{ki}$."},
	{"_lambda",
		(getter)OnlineLDA_lambda,
		(setter)OnlineLDA_set_lambda,
		"Alias for C{lambdas}."},
	{"alpha",
		(getter)OnlineLDA_alpha,
		(setter)OnlineLDA_set_alpha,
		"Controls Dirichlet prior over topic weights, $\\theta_k$."},
	{"eta",
		(getter)OnlineLDA_eta,
		(setter)OnlineLDA_set_eta,
		"Controls Dirichlet prior over topics, $\\beta_{ki}$."},
	{0}
};

static PyMethodDef OnlineLDA_methods[] = {
	{"update_parameters",
		(PyCFunction)OnlineLDA_update_parameters,
		METH_VARARGS | METH_KEYWORDS,
		OnlineLDA_update_parameters_doc},
	{"update_variables",
		(PyCFunction)OnlineLDA_update_variables,
		METH_VARARGS | METH_KEYWORDS,
		OnlineLDA_update_variables_doc},
	{"do_e_step",
		(PyCFunction)OnlineLDA_update_variables,
		METH_VARARGS | METH_KEYWORDS,
		"Alias for C{update_variables}."},
	{"__reduce__", (PyCFunction)OnlineLDA_reduce, METH_NOARGS, 0},
	{"__setstate__", (PyCFunction)OnlineLDA_setstate, METH_VARARGS, 0},
	{0}
};

PyTypeObject OnlineLDA_type = {
	PyObject_HEAD_INIT(0)
	0,                                /*ob_size*/
	"mdlda.models.OnlineLDA",         /*tp_name*/
	sizeof(OnlineLDAObject),          /*tp_basicsize*/
	0,                                /*tp_itemsize*/
	(destructor)Distribution_dealloc, /*tp_dealloc*/
	0,                                /*tp_print*/
	0,                                /*tp_getattr*/
	0,                                /*tp_setattr*/
	0,                                /*tp_compare*/
	0,                                /*tp_repr*/
	0,                                /*tp_as_number*/
	0,                                /*tp_as_sequence*/
	0,                                /*tp_as_mapping*/
	0,                                /*tp_hash */
	0,                                /*tp_call*/
	0,                                /*tp_str*/
	0,                                /*tp_getattro*/
	0,                                /*tp_setattro*/
	0,                                /*tp_as_buffer*/
	Py_TPFLAGS_DEFAULT,               /*tp_flags*/
	OnlineLDA_doc,                    /*tp_doc*/
	0,                                /*tp_traverse*/
	0,                                /*tp_clear*/
	0,                                /*tp_richcompare*/
	0,                                /*tp_weaklistoffset*/
	0,                                /*tp_iter*/
	0,                                /*tp_iternext*/
	OnlineLDA_methods,                /*tp_methods*/
	0,                                /*tp_members*/
	OnlineLDA_getset,                 /*tp_getset*/
	&Distribution_type,               /*tp_base*/
	0,                                /*tp_dict*/
	0,                                /*tp_descr_get*/
	0,                                /*tp_descr_set*/
	0,                                /*tp_dictoffset*/
	(initproc)OnlineLDA_init,         /*tp_init*/
	0,                                /*tp_alloc*/
	Distribution_new,                 /*tp_new*/
};

PyObject* seed(PyObject* self, PyObject* args, PyObject* kwds) {
	int seed;

	if(!PyArg_ParseTuple(args, "i", &seed))
		return 0;

	srand(seed);

	Py_INCREF(Py_None);
	return Py_None;
}

static const char* mdlda_doc =
	"An implementation of mirror descent for latent dirichlet allocation (LDA).";

static PyMethodDef mdlda_methods[] = {
	{"seed", (PyCFunction)seed, METH_VARARGS, 0},
	{"random_select", (PyCFunction)random_select, METH_VARARGS | METH_KEYWORDS, random_select_doc},
	{"sample_dirichlet", (PyCFunction)sample_dirichlet, METH_VARARGS | METH_KEYWORDS, sample_dirichlet_doc},
	{"polygamma", (PyCFunction)polygamma, METH_VARARGS | METH_KEYWORDS, 0},
	{0}
};

PyMODINIT_FUNC init_mdlda() {
	// set random seed
	timeval time;
	gettimeofday(&time, 0);
	srand(time.tv_usec * time.tv_sec);

	// initialize NumPy
	import_array();

	// initialize Eigen
	Eigen::initParallel();

	// create module object
	PyObject* module = Py_InitModule3("_mdlda", mdlda_methods, mdlda_doc);

	// initialize types
	if(PyType_Ready(&OnlineLDA_type) < 0)
		return;

	Py_INCREF(&Distribution_type);
	Py_INCREF(&OnlineLDA_type);

	// add types to module
	PyModule_AddObject(module, "Distribution", reinterpret_cast<PyObject*>(&Distribution_type));
	PyModule_AddObject(module, "OnlineLDA", reinterpret_cast<PyObject*>(&OnlineLDA_type));
}
