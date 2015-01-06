#ifndef LDAINTERFACE_H
#define LDAINTERFACE_H

#define PY_ARRAY_UNIQUE_SYMBOL TRLDA_ARRAY_API
#define NO_IMPORT_ARRAY

#include <Python.h>
#include <arrayobject.h>

#include "trlda/models"
using TRLDA::LDA;

extern const char* LDA_doc;
extern const char* LDA_update_variables_doc;
extern const char* LDA_sample_doc;
extern const char* LDA_lower_bound_doc;

struct LDAObject {
	PyObject_HEAD
	LDA* lda;
};

int LDA_init(LDAObject*, PyObject*, PyObject*);

PyObject* LDA_num_topics(LDAObject*, void*);
PyObject* LDA_num_words(LDAObject*, void*);
PyObject* LDA_num_documents(LDAObject*, void*);
int LDA_set_num_documents(LDAObject*, PyObject*, void*);

PyObject* LDA_lambda(LDAObject*, void*);
int LDA_set_lambda(LDAObject*, PyObject*, void*);

PyObject* LDA_alpha(LDAObject*, void*);
int LDA_set_alpha(LDAObject*, PyObject*, void*);

PyObject* LDA_eta(LDAObject*, void*);
int LDA_set_eta(LDAObject*, PyObject*, void*);

PyObject* LDA_sample(LDAObject*, PyObject*, PyObject*);

PyObject* LDA_update_parameters(LDAObject*, PyObject*, PyObject*);
PyObject* LDA_update_variables(LDAObject*, PyObject*, PyObject*);

PyObject* LDA_lower_bound(LDAObject*, PyObject*, PyObject*);

int PyList_ToDocuments(PyObject* docs, void* documents);
PyObject* PyList_FromDocuments(const LDA::Documents& documents);

PyObject* LDA_str(PyObject*);

#endif
