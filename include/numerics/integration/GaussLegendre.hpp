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
    protected:
        int n;

        double* w_arr;
        double* x_arr;

        static std::tuple<double*, double*> generageWeights(int n);

        double calc_z_log_spacing(double x, double A, double B);

    public:
        GaussLegendre(int n);
        virtual ~GaussLegendre();

        double integrate(std::function<double(double)>& f, double a, double b);
        gsl_complex integrateComplex(std::function<gsl_complex(double)>& f, double a, double b);

        gsl_complex integrateComplexLogSpacing(std::function<gsl_complex(double)>& f, double a, double b);
};

#endif //QUARKDSE_GAUSSLEGENDRE_HPP
