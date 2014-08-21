#!/usr/bin/env python

import os
import sys
import numpy

sys.path.append('./code')
sys.path.append('./code/mdlda')

from distutils.core import setup, Extension
from distutils.ccompiler import CCompiler
from utils import parallelCCompiler
from python import __version__

if 'darwin' in sys.platform:
	libraries = []
	extra_compile_args = []
else:
	libraries = ['gomp']
	extra_compile_args = ['-fopenmp']

modules = [
	Extension('_mdlda',
		language='c++',
		sources=[
				'code/mdlda/src/onlinelda.cpp',
				'code/mdlda/src/distribution.cpp',
				'code/mdlda/src/digamma.cpp',
				'code/mdlda/src/utils.cpp',
				'code/mdlda/python/src/module.cpp',
				'code/mdlda/python/src/onlineldainterface.cpp',
				'code/mdlda/python/src/distributioninterface.cpp',
				'code/mdlda/python/src/pyutils.cpp',
			],
		include_dirs=[
				'code/',
				'code/mdlda/include',
				'code/mdlda/python/include',
				os.path.join(numpy.__path__[0], 'core/include/numpy')
			],
		libraries=[] + libraries,
		extra_compile_args=[
				'-Wno-write-strings',
			] + extra_compile_args)
	]

# enable parallel compilation
CCompiler.compile = parallelCCompiler

setup(
	name='mdlda',
	version=__version__,
	author='Lucas Theis',
	author_email='theis@adobe.com',
	ext_modules=modules,
	description='An implementation of mirror descent for latent dirichlet allocation (LDA).',
	package_dir={'mdlda': 'code/mdlda/python'},
	packages=[
			'mdlda',
			'mdlda.models',
			'mdlda.utils'
		])
