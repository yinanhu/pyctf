#ifndef H_FILTERS
#define H_FILTERS

#define MAXORDER    20
#define ENABLE      1
#define DISABLE     0

/* filter type definitions */
#define LOWPASS     0
#define HIGHPASS    1
#define BANDPASS    2
#define BANDREJECT  3

typedef struct iirspec {
    int type;               /* filter type */
    int enable;             /* 1=enable filter, 0=disable filter */
    double fh;              /* high edge filter frequency (Hz) */
    double fl;              /* low edge filter frequency (Hz) */
    double rate;            /* sample rate (Hz) */
    double bwn;             /* noise bandwidth (Hz) */
    int order;              /* filter order */
    int NC;                 /* number of coefficient pairs */
    double num[MAXORDER];   /* numerator iir coefficients */
    double den[MAXORDER];   /* denominator iir coefficients */
} IIRSPEC;

extern int mkiir(IIRSPEC *);
extern int bdiir(double *, double *, int, IIRSPEC *);
extern double response(IIRSPEC *, double, double);
extern void Butterworth(double *, int, double, double, double, int, int, double *);
extern void FFTfilter(double *, double *, double *, int);

#endif  // H_FILTERS
