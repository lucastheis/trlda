#ifndef ONLINELDAINTERFACE_H
#define ONLINELDAINTERFACE_H

#define PY_ARRAY_UNIQUE_SYMBOL TRLDA_ARRAY_API
#define NO_IMPORT_ARRAY

#include <Python.h>
#include <arrayobject.h>
#include "ldainterface.h"

#include "trlda/models"
using TRLDA::OnlineLDA;

extern const char* OnlineLDA_doc;
extern const char* OnlineLDA_update_parameters_doc;

struct OnlineLDAObject {
	PyObject_HEAD
	OnlineLDA* lda;
};

int OnlineLDA_init(OnlineLDAObject*, PyObject*, PyObject*);

PyObject* OnlineLDA_num_documents(OnlineLDAObject*, void*);
int OnlineLDA_set_num_documents(OnlineLDAObject*, PyObject*, void*);

PyObject* OnlineLDA_update_parameters(OnlineLDAObject*, PyObject*, PyObject*);

PyObject* OnlineLDA_reduce(OnlineLDAObject*, PyObject*);
PyObject* OnlineLDA_setstate(OnlineLDAObject*, PyObject*);

#endif
