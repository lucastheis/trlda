#ifndef BATCHLDAINTERFACE_H
#define BATCHLDAINTERFACE_H

#define PY_ARRAY_UNIQUE_SYMBOL TRLDA_ARRAY_API
#define NO_IMPORT_ARRAY

#include <Python.h>
#include <arrayobject.h>
#include "ldainterface.h"

#include "trlda/models"
using TRLDA::BatchLDA;

extern const char* BatchLDA_doc;
extern const char* BatchLDA_update_parameters_doc;

struct BatchLDAObject {
	PyObject_HEAD
	BatchLDA* lda;
};

int BatchLDA_init(BatchLDAObject*, PyObject*, PyObject*);

PyObject* BatchLDA_update_parameters(BatchLDAObject*, PyObject*, PyObject*);

PyObject* BatchLDA_reduce(BatchLDAObject*, PyObject*);
PyObject* BatchLDA_setstate(BatchLDAObject*, PyObject*);

#endif
