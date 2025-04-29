//
// Created by past12am on 3/2/23.
//

#ifndef QUARKDSE_GAUSSLEGENDREDEFORMED_HPP
#define QUARKDSE_GAUSSLEGENDREDEFORMED_HPP

#include <cstddef>
#include <tuple>

#include <functional>
#include <complex>
#include <gsl/gsl_complex.h>

#include "GaussLegendre.hpp"

class GaussLegendreDeformed : GaussLegendre
{
    public:
        GaussLegendreDeformed(int n);
        ~GaussLegendreDeformed();

    gsl_complex integrateComplexDeformed(std::function<gsl_complex(gsl_complex)>& f,
        const std::function<gsl_complex(double, double, double, double)>& gamma,
        const std::function<gsl_complex(double, double, double, double)>& deriv_gamma,
        double a, double b, double X, double epsilon, double eta);

    gsl_complex integrateComplexDeformedAndStoreKernel(std::function<gsl_complex(gsl_complex)>& f,
        const std::function<gsl_complex(double, double, double, double)>& gamma,
        const std::function<gsl_complex(double, double, double, double)>& deriv_gamma,
        double a, double b, double X, double Z, double epsilon, double eta, char* file_prefix);

    gsl_complex integrateComplexDeformedLogSpacing(std::function<gsl_complex(gsl_complex)>& f,
        const std::function<gsl_complex(double, double, double, double)>& gamma,
        const std::function<gsl_complex(double, double, double, double)>& deriv_gamma,
        double a, double b, double X, double epsilon, double eta);
};

#endif //QUARKDSE_GAUSSLEGENDREDEFORMED_HPP
