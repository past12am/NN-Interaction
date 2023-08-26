//
// Created by past12am on 3/2/23.
//

#include "../../../include/numerics/integration/GaussLegendre.hpp"
#include "../../../include/numerics/roots/NewtonRootFinder.hpp"
#include "../../../include/numerics/polynomials/LegendrePolynomials.hpp"

#include <math.h>
#include <numbers>

#include <stdio.h>
#include <iostream>
#include <gsl/gsl_complex_math.h>

std::tuple<double*, double*> GaussLegendre::generageWeights(int n)
{
    double* x_arr = new double[n];
    double* w_arr = new double[n];

    // Find n Roots of the n-th Legendre Polynomial
    for (int i = 0; i < (n + 1)/2; i++)
    {
        double init_x_estimate = cos(std::numbers::pi * (4.0 * i + 3)/(4.0 * n + 2));
        std::tuple<double, double, double> root_fval_dfval = NewtonRootFinder::estimateRoot(LegendrePolynomials::LegendreP_n, init_x_estimate, 3E-14, n);

        double df_val = std::get<2>(root_fval_dfval);

        x_arr[i] = std::get<0>(root_fval_dfval);
        x_arr[n - 1 - i] = -x_arr[i];

        w_arr[i] = 2.0/((1.0 - pow(x_arr[i], 2.0)) * pow(df_val, 2.0));
        w_arr[n - 1 - i] = w_arr[i];
    }

    return std::tuple<double*, double*>{w_arr, x_arr};
}

double GaussLegendre::integrate(std::function<double(double)>& f, double a, double b)
{
    /*
    std::cout << "w: ";
    for (int i = 0; i < n; i++)
    {
        std::cout << w_arr[i] << " # ";
    }
    std::cout << std::endl;

    std::cout << "x: ";
    for (int i = 0; i < n; i++)
    {
        std::cout << x_arr[i] << " # ";
    }
    std::cout << std::endl;
    */

    double val = 0.0;
    for (int i = 0; i < n; i++)
    {
        double cur_val = w_arr[i] * (b - a)/2.0 * f((b - a)/2.0 * x_arr[i] + (b + a)/2.0);
        val += cur_val;
    }

    return val;
}

GaussLegendre::GaussLegendre(int n) : n(n)
{
    std::tuple<double*, double*> weights = generageWeights(n);

    w_arr = std::get<0>(weights);
    x_arr = std::get<1>(weights);
}

GaussLegendre::~GaussLegendre()
{
    delete []w_arr;
    delete []x_arr;
}

gsl_complex GaussLegendre::integrateComplex(std::function<gsl_complex(double)> &f, double a, double b)
{
    gsl_complex val = gsl_complex_rect(0, 0);
    for (int i = 0; i < n; i++)
    {
        gsl_complex cur_val = gsl_complex_mul_real(f((b - a)/2.0 * x_arr[i] + (b + a)/2.0), w_arr[i] * (b - a)/2.0);
        val = gsl_complex_add(val, cur_val);
    }

    return val;
}
