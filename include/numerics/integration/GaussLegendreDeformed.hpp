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

        gsl_complex integrateComplexDeformed(std::function<gsl_complex(gsl_complex)>& f, double a, double b);
};

#endif //QUARKDSE_GAUSSLEGENDREDEFORMED_HPP
