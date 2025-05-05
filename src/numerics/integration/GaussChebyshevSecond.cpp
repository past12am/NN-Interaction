//
// Created by past12am on 3/2/23.
//

#include "../../../include/numerics/integration/GaussChebyshevSecond.hpp"

#include <math.h>
#include <numbers>

#include <functional>
#include <gsl/gsl_complex_math.h>

#include "../../../include/gslhacks/GSLComplexOperators.hpp"


std::tuple<double*, double*> GaussChebyshevSecond::generageWeights(int n)
{
    double* x_arr = new double[n];
    double* w_arr = new double[n];

    for (int i = 0; i < n; i++)
    {
        double ip1pi_over_np1 = (i + 1) * std::numbers::pi / ((double) (n + 1));
        double sinval = sin(ip1pi_over_np1);

        w_arr[i] = sinval * sinval;
        x_arr[i] = cos(ip1pi_over_np1);
    }

    return std::tuple<double*, double*>{w_arr, x_arr};
}

GaussChebyshevSecond::GaussChebyshevSecond(int n) : n(n)
{
    std::tuple<double*, double*> weights = generageWeights(n);

    w_arr = std::get<0>(weights);
    x_arr = std::get<1>(weights);
}

double GaussChebyshevSecond::integrate_f_times_sqrt(std::function<double(double)>& f)
{
    double val = 0.0;
    for (int i = 0; i < n; i++)
    {
        val = val + w_arr[i] * f(x_arr[i]);
    }

    return val * (std::numbers::pi / ((double) (n+1)));
}

gsl_complex GaussChebyshevSecond::integrate_complex_f_times_sqrt(std::function<gsl_complex(double)> &f)
{
    gsl_complex val = gsl_complex_rect(0, 0);
    for (int i = 0; i < n; i++)
    {
        val = val + w_arr[i] * f(x_arr[i]);
    }

    return val * (std::numbers::pi / ((double) (n+1)));
}
