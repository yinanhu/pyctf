// bdiir() - zero phase-shift infinite impulse response digital filter
//
//      Author: Stephen E. Robinson
//                      Neuromagnetism Laboratory
//                      Dept. of Neurology
//                      Henry Ford Hospital
//                      Detroit, MI
//                      Copyright (c) 2007
//

#include <stdlib.h>
#include "filters.h"

int bdiir(double *In,           // input data array
	  double *Out,          // output data array
	  int T,                // number of points in data array
	  IIRSPEC *Filter       // iir filter structure pointer
) {
    static double *Tmp;         // temporary time-series
    double offset;              // value of 0th data point
    register double x;          // input sample value;
    register double y;          // output sample value;
    register int t;             // time-index
    register int i;             // coefficient-index
    register int j;             // offset index
    static int mem = 0;

    // if req'd, allocate or modify intermediate array
    if (mem != T) {
	if (mem != 0)
	    free((void *)Tmp);
	if ((Tmp = (double *)malloc((size_t) (T * sizeof(double)))) == NULL)
	    return -1;
	mem = T;
    }
    // filter is not enabled, move data to output
    if (Filter->enable == 0) {
	for (t = 0; t < T; t++)
	    Out[t] = In[t];
	return 0;
    }
    Filter->den[0] = 0.;        // set 1st term of denominator to zero
    offset = In[0];             // get starting value to offset series
    for (t = 0; t < T; t++)
	for (i = 0, Tmp[t] = 0.; i < Filter->NC; i++) {
	    j = t - i;
	    if (j < 0) {
		x = y = 0.;
	    } else {
		x = In[j] - offset;
		y = Tmp[j];
	    }
	    Tmp[t] += Filter->num[i] * x - Filter->den[i] * y;
	}
    offset = Tmp[T - 1];        // get ending value to offset series
    for (t = (T - 1); t >= 0; t--)
	for (i = 0, Out[t] = 0.; i < Filter->NC; i++) {
	    j = t + i;
	    if (j >= T) {
		x = y = 0.;
	    } else {
		x = Tmp[j] - offset;
		y = Out[j];
	    }
	    Out[t] += Filter->num[i] * x - Filter->den[i] * y;
	}
    return 0;
}
