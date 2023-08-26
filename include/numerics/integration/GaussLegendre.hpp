//
// Created by past12am on 3/2/23.
//

#ifndef QUARKDSE_GAUSSLEGENDRE_HPP
#define QUARKDSE_GAUSSLEGENDRE_HPP

#include <cstddef>
#include <tuple>

#include <functional>
#include <complex>
#include <gsl/gsl_complex.h>

class GaussLegendre
{
    private:
        int n;

        double* w_arr;
        double* x_arr;

        static std::tuple<double*, double*> generageWeights(int n);

    public:
        GaussLegendre(int n);
        virtual ~GaussLegendre();

        double integrate(std::function<double(double)>& f, double a, double b);
        gsl_complex integrateComplex(std::function<gsl_complex(double)>& f, double a, double b);
};

#endif //QUARKDSE_GAUSSLEGENDRE_HPP
