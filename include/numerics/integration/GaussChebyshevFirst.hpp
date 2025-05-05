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

class GaussChebyshevFirst
{
    private:
        int n;

    public:
        GaussChebyshevFirst(int n);

        double integrate_f_over_sqrt(std::function<double(double)>& f);
        gsl_complex integrate_complex_f_over_sqrt(std::function<gsl_complex(double)>& f);
};


#endif //QUARKDSE_GAUSSCHEBYSHEV_HPP
