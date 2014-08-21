/**
 * Copyright © 2001, 2002 Enthought, Inc.
 * All rights reserved.
 *
 * Copyright © 2003-2013 SciPy Developers.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *
 * - Redistributions of source code must retain the above copyright notice, this list of conditions and
 *   the following disclaimer.
 *
 * - Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation and/or other materials
 *   provided with the distribution.
 *
 * - Neither the name of Enthought nor the names of the SciPy Developers may be used to endorse or promote
 *   products derived from this software without specific prior written
 *   permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS
 * OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
 * IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF
 * THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <limits>
using std::numeric_limits;

#include <cmath>
using std::log;
using std::floor;

#include "utils.h"

#define UNK 1

#ifdef UNK
static double A[] = {
	8.33333333333333333333E-2,
	-2.10927960927960927961E-2,
	7.57575757575757575758E-3,
	-4.16666666666666666667E-3,
	3.96825396825396825397E-3,
	-8.33333333333333333333E-3,
	8.33333333333333333333E-2
};
#endif

#ifdef DEC
static unsigned short A[] = {
	0037252, 0125252, 0125252, 0125253,
	0136654, 0145314, 0126312, 0146255,
	0036370, 0037017, 0101740, 0174076,
	0136210, 0104210, 0104210, 0104211,
	0036202, 0004040, 0101010, 0020202,
	0136410, 0104210, 0104210, 0104211,
	0037252, 0125252, 0125252, 0125253
};
#endif

#ifdef IBMPC
static unsigned short A[] = {
	0x5555, 0x5555, 0x5555, 0x3fb5,
	0x5996, 0x9599, 0x9959, 0xbf95,
	0x1f08, 0xf07c, 0x07c1, 0x3f7f,
	0x1111, 0x1111, 0x1111, 0xbf71,
	0x0410, 0x1041, 0x4104, 0x3f70,
	0x1111, 0x1111, 0x1111, 0xbf81,
	0x5555, 0x5555, 0x5555, 0x3fb5
};
#endif

#ifdef MIEEE
static unsigned short A[] = {
	0x3fb5, 0x5555, 0x5555, 0x5555,
	0xbf95, 0x9959, 0x9599, 0x5996,
	0x3f7f, 0x07c1, 0xf07c, 0x1f08,
	0xbf71, 0x1111, 0x1111, 0x1111,
	0x3f70, 0x4104, 0x1041, 0x0410,
	0xbf81, 0x1111, 0x1111, 0x1111,
	0x3fb5, 0x5555, 0x5555, 0x5555
};
#endif

/*
 * Cephes Math Library Release 2.1: December, 1988
 * Copyright 1984, 1987, 1988 by Stephen L. Moshier
 * Direct inquiries to 30 Frost Street, Cambridge, MA 02140
 */
double polevl(double x, double coef[], int N) {
	double ans;
	int i;
	double *p;

	p = coef;
	ans = *p++;
	i = N;

	do
		ans = ans * x + *p++;
	while(--i);

	return ans;
}

/*
 * Cephes Math Library Release 2.8: June, 2000
 * Copyright 1984, 1987, 1992, 2000 by Stephen L. Moshier
 */
double MDLDA::digamma(double x) {
	double p, q, nz, s, w, y, z;
	int i, n, negative;

	negative = 0;
	nz = 0.0;

	if(x <= 0.0) {
		negative = 1;
		q = x;
		p = floor(q);

		if(p == q)
			return numeric_limits<double>::infinity();

		// Remove the zeros of tan(MDLDA_PI x) by subtracting the nearest integer from x
		nz = q - p;

		if(nz != 0.5) {
			if (nz > 0.5) {
				p += 1.0;
				nz = q - p;
			}
			nz = MDLDA_PI / tan(MDLDA_PI * nz);
		} else {
			nz = 0.0;
		}
		x = 1.0 - x;
	}

	// check for positive integer up to 10
	if(x <= 10.0 && x == floor(x)) {
		y = 0.0;
		n = x;
		for (i = 1; i < n; i++) {
			w = i;
			y += 1.0 / w;
		}
		y -= MDLDA_EULER;
		goto done;
	}

	s = x;
	w = 0.0;
	while(s < 10.0) {
		w += 1.0 / s;
		s += 1.0;
	}

	if(s < 1.0e17) {
		z = 1.0 / (s * s);
		y = z * polevl(z, A, 6);
	} else
		y = 0.0;

	y = log(s) - (0.5 / s) - y - w;

	done:
		if(negative)
			y -= nz;

	return (y);
}
