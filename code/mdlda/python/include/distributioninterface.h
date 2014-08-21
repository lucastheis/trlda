#ifndef DISTRIBUTIONINTERFACE_H
#define DISTRIBUTIONINTERFACE_H

#define PY_ARRAY_UNIQUE_SYMBOL MDLDA_ARRAY_API
#define NO_IMPORT_ARRAY
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <arrayobject.h>

#include "mdlda/models"
using MDLDA::Distribution;

struct DistributionObject {
	PyObject_HEAD
	Distribution* dist;
};

extern const char* Distribution_doc;

PyObject* Distribution_new(PyTypeObject*, PyObject*, PyObject*);
int Distribution_init(DistributionObject*, PyObject*, PyObject*);
void Distribution_dealloc(DistributionObject*);

#endif
