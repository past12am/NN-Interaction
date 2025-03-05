//
// Created by past12am on 3/5/25.
//

#ifndef MOMENTUMLOOP_HPP
#define MOMENTUMLOOP_HPP
#include <functional>
#include <gsl/gsl_vector.h>

class MomentumLoop
{
    public:
        virtual gsl_complex integrate_4d(const std::function<gsl_complex(double, double, double, double)>& f, double cutoff) = 0;

        virtual void calc_k(double k2, double z, double y, double phi, gsl_vector_complex* k) = 0;

        virtual ~MomentumLoop() {}
};

#endif //MOMENTUMLOOP_HPP
