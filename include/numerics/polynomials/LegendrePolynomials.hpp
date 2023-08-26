//
// Created by past12am on 3/2/23.
//

#ifndef QUARKDSE_LEGENDREPOLYNOMIALS_HPP
#define QUARKDSE_LEGENDREPOLYNOMIALS_HPP

#include <cstddef>
#include <numeric>
#include <math.h>

class LegendrePolynomials
{
    private:
        static double LegendreP_0(double x);
        static double LegendreP_1(double x);

        static double DerivativeLegendreP_0(double x);
        static double DerivativeLegendreP_1(double x);

    public:
        static std::tuple<double, double> LegendreP_n(double x, int n);
};


#endif //QUARKDSE_LEGENDREPOLYNOMIALS_HPP
