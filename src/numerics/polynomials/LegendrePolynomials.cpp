//
// Created by past12am on 3/2/23.
//

#include "../../../include/numerics/polynomials/LegendrePolynomials.hpp"
#include "tuple"

double LegendrePolynomials::LegendreP_0(double x)
{
    return 1.0;
}

double LegendrePolynomials::LegendreP_1(double x)
{
    return x;
}

double LegendrePolynomials::DerivativeLegendreP_0(double x)
{
    return 0.0;
}

double LegendrePolynomials::DerivativeLegendreP_1(double x)
{
    return 1.0;
}

std::tuple<double, double> LegendrePolynomials::LegendreP_n(double x, int n)
{
    switch (n)
    {
        case 0: return std::tuple<double, double>{LegendreP_0(x), 0.0};
        case 1: return std::tuple<double, double>{LegendreP_1(x), 1.0};
    }

    double p_p = LegendreP_0(x);    // p_previous
    double p = LegendreP_1(x);      // p
    double p_n;

    // TODO check why we do not need dp_p
    //double dp_p = DerivativeLegendreP_0(x);
    double dp = DerivativeLegendreP_1(x);
    double dp_n;

    for(int i = 2; i <= n; i++)
    {
        // Calculate Legendre Polynomial
        //p_n = ((2*i + 1) * x * p - i * p_p)/(i+1);   // p_next
        p_n = ((2.0*i - 1.0) * x * p - (i - 1.0) * p_p)/i;

        // Calculate Derivative of Legendre Polynomial
        //dp_n = ((2*i + 1) * (p + x * dp) - i * dp_p)/i;
        dp_n = n * (x*p_n - p)/(pow(x, 2.0) - 1.0);

        p_p = p;
        p = p_n;

        //dp_p = dp;
        dp = dp_n;
    }

    return std::tuple<double, double>{p, dp};
}
