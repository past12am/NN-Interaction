//
// Created by past12am on 3/2/23.
//

#ifndef QUARKDSE_GAUSSCHEBYSHEV_HPP
#define QUARKDSE_GAUSSCHEBYSHEV_HPP

#include <stddef.h>
#include <tuple>

#include <complex>
#include <functional>
#include <gsl/gsl_complex.h>

class GaussChebyshevSecond
{
    private:
        int n;

        double* w_arr;
        double* x_arr;

        static std::tuple<double*, double*> generageWeights(int n);

    public:
        GaussChebyshevSecond(int n);

        double integrate_f_times_sqrt(std::function<double(double)>& f);
        gsl_complex integrate_complex_f_times_sqrt(std::function<gsl_complex(double)>& f);
};


#endif //QUARKDSE_GAUSSCHEBYSHEV_HPP
