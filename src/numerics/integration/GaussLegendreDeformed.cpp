//
// Created by past12am on 3/2/23.
//

#include "../../../include/numerics/integration/GaussLegendreDeformed.hpp"

#include <fstream>
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

gsl_complex GaussLegendreDeformed::integrateComplexDeformedAndStoreKernel(std::function<gsl_complex(gsl_complex)>& f,
        const std::function<gsl_complex(double, double, double, double)>& gamma, const std::function<gsl_complex(double, double, double, double)>& deriv_gamma,
        double a, double b, double X, double Z, double epsilon, double eta, char* file_prefix)
{
        gsl_complex val = gsl_complex_rect(0, 0);

        std::ostringstream fnamestrstream;
        fnamestrstream << "/home/past12am/OuzoCloud/Studium/Physik/6_Semester/SE_Bachelorarbeit/NNInteraction/debug-integrands/" << file_prefix << "_X=" << X << "_Z=" << Z << "_eps=" << epsilon << ".csv";

        std::ofstream data_file;
        data_file.open(fnamestrstream.str(), std::ofstream::out | std::ios::trunc);

        data_file << "x4,kernel_real,kernel_imag" << std::endl;

        for (int i = 0; i < n; i++)
        {
                double t = (b - a) / 2.0 * x_arr[i] + (b + a) / 2.0;
                gsl_complex gamma_res = gamma(t, X, epsilon, eta);
                gsl_complex deriv_gamma_res = deriv_gamma(t, X, epsilon, eta);
                gsl_complex kernel = f(gamma_res) * deriv_gamma_res;

                gsl_complex cur_val = gsl_complex_mul_real(kernel, w_arr[i] * (b - a)/2.0);
                val = gsl_complex_add(val, cur_val);

                data_file << t << "," << GSL_REAL(kernel) << "," << GSL_IMAG(kernel) << std::endl;
        }

        data_file.close();

        return val;
}
