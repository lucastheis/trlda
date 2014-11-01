#!/usr/bin/env python

import os
import sys
import numpy

sys.path.append('./code')
sys.path.append('./code/trlda')

from distutils.core import setup, Extension
from distutils.ccompiler import CCompiler
from utils import parallelCCompiler
from python import __version__

if 'darwin' in sys.platform:
	libraries = []
	extra_compile_args = ['-std=c++0x', '-stdlib=libc++']
else:
	libraries = ['gomp']
	extra_compile_args = ['-std=c++0x', '-fopenmp']

modules = [
	Extension('_trlda',
		language='c++',
		sources=[
				'code/trlda/src/onlinelda.cpp',
				'code/trlda/src/distribution.cpp',
				'code/trlda/src/digamma.cpp',
				'code/trlda/src/utils.cpp',
				'code/trlda/src/zeta.cpp',
				'code/trlda/python/src/module.cpp',
				'code/trlda/python/src/distributioninterface.cpp',
				'code/trlda/python/src/onlineldainterface.cpp',
				'code/trlda/python/src/utilsinterface.cpp',
				'code/trlda/python/src/pyutils.cpp',
			],
		include_dirs=[
				'code/',
				'code/trlda/include',
				'code/trlda/python/include',
				os.path.join(numpy.__path__[0], 'core/include/numpy')
			],
		libraries=[] + libraries,
		extra_compile_args=[
				'-Wno-write-strings',
				'-Wno-sign-compare',
				'-Wno-unused-variable',
				'-Wno-#warnings',
			] + extra_compile_args)
	]

# enable parallel compilation
CCompiler.compile = parallelCCompiler

setup(
	name='trlda',
	version=__version__,
	author='Lucas Theis',
	author_email='lucas@theis.io',
	ext_modules=modules,
	description='An implementation of an online trust region method for latent dirichlet allocation (LDA).',
	package_dir={'trlda': 'code/trlda/python'},
	packages=[
			'trlda',
			'trlda.models',
			'trlda.utils'
		])
