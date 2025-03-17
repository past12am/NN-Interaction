//
// Created by past12am on 3/2/23.
//

#include "../../../include/numerics/integration/GaussLegendreDeformed.hpp"

#include <math.h>
#include <numbers>

#include <stdio.h>
#include <iostream>
#include <gsl/gsl_complex_math.h>

#include "../../../include/gslhacks/GSLComplexOperators.hpp"

GaussLegendreDeformed::GaussLegendreDeformed(int n) : GaussLegendre(n)
{

}

GaussLegendreDeformed::~GaussLegendreDeformed()
{

}

gsl_complex GaussLegendreDeformed::integrateComplexDeformed(std::function<gsl_complex(gsl_complex)>& f,
        const std::function<gsl_complex(double, double, double, double)>& gamma, const std::function<gsl_complex(double, double, double, double)>& deriv_gamma,
        double a, double b, double X, double epsilon, double eta)
{
        gsl_complex val = gsl_complex_rect(0, 0);

        for (int i = 0; i < n; i++)
        {
                double t = (b - a) / 2.0 * x_arr[i] + (b + a) / 2.0;
                gsl_complex gamma_res = gamma(t, X, epsilon, eta);
                gsl_complex deriv_gamma_res = deriv_gamma(t, X, epsilon, eta);
                gsl_complex kernel = f(gamma_res) * deriv_gamma_res;

                gsl_complex cur_val = gsl_complex_mul_real(kernel, w_arr[i] * (b - a)/2.0);
                val = gsl_complex_add(val, cur_val);
        }

        return val;
}
