// Butterworth() -- generate a Butterworth filter function
//  return gain coefficients & effective bandwidth. Coefficients
//  for positive frequencies are returned in 0 to N/2, while
//  those for negagive frequencies are returned in N-2+1 to N-1.
//
//  Author: Stephen E. Robinson
//          MEG Core Facility
//          NIMH
//

#include <stdio.h>
#include <math.h>
#include <filters.h>

void Butterworth(
    double      *Gain,          // Gain[N] -- the filter function
    int         N,              // number of points in the filter window
    double      HighPassFreq,   // highpass cutoff frequency (Hz)
    double      LowPassFreq,    // lowpass cutoff frequency (Hz)
    double      SampleRate,     // sample rate (Hz)
    int         FilterType,     // highpass, lowpass, bandpass
    int         Order,          // the filter order
    double      *bw             // bw = sum gain*binwidth
) {
    double      Freq;           // frequency (fraction sample rate)
    double      Fh;             // highpass (fraction sample rate)
    double      Fl;             // lowpass (fraction sample rate)
    double      Fw;             // filter width (fraction sample rate)
    double      Fc;             // filter center frequency (fraction sample rate)
    double      ratio;          // frequency ratio
    double      exponent;       // 2*order
    double      binwidth;       // SampleRate / N
    double      tmp;            // what else?
    int         NumBins;        // N / 2
    int         i;              // i-index (positive freqs)
    int         j;              // j-index (negative freqs)

    // set up constants
    exponent = 2. * (double)Order;
    NumBins = N / 2;
    binwidth = SampleRate / (double)N;

    // select filter function
    switch(FilterType) {

    case HIGHPASS:      // highpass filter transfer function
	Fh = HighPassFreq;
	for(i=0, j=N-1, *bw=0.; i<NumBins; i++, j--) {
	    Freq = (double)i * binwidth;
	    ratio = Fh / Freq;
	    tmp = sqrt(1. / (1. + pow(ratio, exponent)));
	    Gain[i] = tmp;
	    Gain[j] = tmp;
	    *bw += tmp * binwidth;
	}
	break;

    case LOWPASS:       // lowpass filter transfer function
	Fl = LowPassFreq;
	for(i=0, j=N-1, *bw=0.; i<NumBins; i++, j--) {
	    Freq = (double)i * binwidth;
	    ratio = Freq / Fl;
	    tmp = sqrt(1. / (1. + pow(ratio, exponent)));
	    Gain[i] = tmp;
	    Gain[j] = tmp;
	    *bw += tmp * binwidth;
	}
	break;

    case BANDPASS:      // bandpass filter transfer function
	Fw = (LowPassFreq - HighPassFreq);
	Fc = (LowPassFreq + HighPassFreq) / 2.;
	for(i=0, j=N-1, *bw=0.; i<NumBins; i++, j--) {
	    Freq = (double)i * binwidth;
	    ratio = 2. * (Freq - Fc) / Fw;
	    tmp = sqrt(1. / (1. + pow(ratio, exponent)));
	    Gain[i] = tmp;
	    Gain[j] = tmp;
	    *bw += tmp * binwidth;
	}
	break;

    case BANDREJECT:    // bandstop filter transfer function
	Fw = (LowPassFreq - HighPassFreq);
	Fc = (LowPassFreq + HighPassFreq) / 2.;
	for(i=0, j=N-1, *bw=0.; i<NumBins; i++, j--) {
	    Freq = (double)i * binwidth;
	    ratio = 0.5 * Fw / (Freq - Fc);
	    tmp = sqrt(1. / (1. + pow(ratio, exponent)));
	    Gain[i] = tmp;
	    Gain[j] = tmp;
	    *bw += tmp * binwidth;
	}
	break;
#if 0
    case ALLPASS:                   // allpass
	for(i=0, *bw=0.; i<N; i++) {
	    Gain[i] = 1.;
	    *bw += binwidth;
	}
	break;
#endif
    default:
	printf("unknown Butterworth filter type\n");
	break;
    }
}
