//
// Created by past12am on 3/2/23.
//

#include "../../../include/numerics/integration/GaussLegendreDeformed.hpp"
#include "../../../include/numerics/roots/NewtonRootFinder.hpp"
#include "../../../include/numerics/polynomials/LegendrePolynomials.hpp"

#include <math.h>
#include <numbers>

#include <stdio.h>
#include <iostream>
#include <gsl/gsl_complex_math.h>
GaussLegendreDeformed::GaussLegendreDeformed(int n) : GaussLegendre(n)
{

}

GaussLegendreDeformed::~GaussLegendreDeformed()
{

}

gsl_complex GaussLegendreDeformed::integrateComplexDeformed(std::function<gsl_complex(gsl_complex)>& f, double a,
                                                            double b)
{
    // TODO
    return GSL_COMPLEX_ZERO;
}
