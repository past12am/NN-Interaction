//
// Created by past12am on 3/2/23.
//

#include "../../../include/numerics/integration/GaussChebyshev.hpp"

#include <math.h>
#include <numbers>

#include <functional>
#include <gsl/gsl_complex_math.h>

double GaussChebyshev::integrate_f_times_sqrt(std::function<double(double)>& f)
{
    double val = 0;
    double a = std::numbers::pi/(n + 1.0);
    for (int i = 0; i < n; i++)
    {
        val += (a * pow(sin(a * i), 2)) * f(cos(a * i));
    }
    return val;
}

GaussChebyshev::GaussChebyshev(int n) : n(n)
{

}

gsl_complex GaussChebyshev::integrate_complex_f_times_sqrt(std::function<gsl_complex(double)> &f)
{
    gsl_complex val = gsl_complex_rect(0, 0);
    double a = std::numbers::pi/(n + 1.0);
    for (int i = 0; i < n; i++)
    {
        val = gsl_complex_add(val, gsl_complex_mul_real(f(cos(a * i)), a * pow(sin(a * i), 2)));
    }
    return val;
}
