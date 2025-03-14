//
// Created by past12am on 3/13/25.
//

#ifndef GSLCOMPLEXOPERATORS_HPP
#define GSLCOMPLEXOPERATORS_HPP

#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>

inline gsl_complex operator+(const gsl_complex a, const gsl_complex b)
{
    return gsl_complex_add(a, b);
}

inline gsl_complex operator-(const gsl_complex a, const gsl_complex b)
{
    return gsl_complex_sub(a, b);
}

inline gsl_complex operator*(const gsl_complex a, const gsl_complex b)
{
    return gsl_complex_mul(a, b);
}

inline gsl_complex operator/(gsl_complex a, gsl_complex b)
{
    return gsl_complex_div(a, b);
}




inline gsl_complex operator+(const gsl_complex a, const double b)
{
    return gsl_complex_add_real(a, b);
}

inline gsl_complex operator-(const gsl_complex a, const double b)
{
    return gsl_complex_sub_real(a, b);
}

inline gsl_complex operator*(const gsl_complex a, const double b)
{
    return gsl_complex_mul_real(a, b);
}

inline gsl_complex operator/(const gsl_complex a, const double b)
{
    return gsl_complex_div_real(a, b);
}



inline gsl_complex operator+(const double a, const gsl_complex b)
{
    return gsl_complex_add_real(b, a);
}

inline gsl_complex operator*(const double a, const gsl_complex b)
{
    return gsl_complex_mul_real(b, a);
}

inline gsl_complex operator/(const double a, const gsl_complex b)
{
    return a * gsl_complex_inverse(b);
}


#endif //GSLCOMPLEXOPERATORS_HPP
