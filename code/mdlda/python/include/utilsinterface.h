#ifndef UTILSINTERFACE_H
#define UTILSINTERFACE_H

#define PY_ARRAY_UNIQUE_SYMBOL MDLDA_ARRAY_API
#define NO_IMPORT_ARRAY
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <arrayobject.h>
#include "pyutils.h"

extern const char* random_select_doc;
extern const char* sample_dirichlet_doc;

PyObject* random_select(PyObject*, PyObject*, PyObject*);
PyObject* sample_dirichlet(PyObject*, PyObject*, PyObject*);

#endif
