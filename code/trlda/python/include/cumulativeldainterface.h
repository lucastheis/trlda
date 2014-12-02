#ifndef CUMULATIVELDAINTERFACE_H
#define CUMULATIVELDAINTERFACE_H

#define PY_ARRAY_UNIQUE_SYMBOL TRLDA_ARRAY_API
#define NO_IMPORT_ARRAY

#include <Python.h>
#include <arrayobject.h>
#include "ldainterface.h"

#include "trlda/models"
using TRLDA::CumulativeLDA;

extern const char* CumulativeLDA_doc;
extern const char* CumulativeLDA_update_parameters_doc;

struct CumulativeLDAObject {
	PyObject_HEAD
	CumulativeLDA* lda;
};

int CumulativeLDA_init(CumulativeLDAObject*, PyObject*, PyObject*);

PyObject* CumulativeLDA_update_parameters(CumulativeLDAObject*, PyObject*, PyObject*);

PyObject* CumulativeLDA_reduce(CumulativeLDAObject*, PyObject*);
PyObject* CumulativeLDA_setstate(CumulativeLDAObject*, PyObject*);

#endif
