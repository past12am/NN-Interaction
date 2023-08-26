//
// Created by past12am on 3/2/23.
//

#include "../../../include/numerics/roots/NewtonRootFinder.hpp"

#include <math.h>
#include <tuple>

#include <iostream>

std::tuple<double, double, double> NewtonRootFinder::estimateRoot(std::tuple<double, double> (* f)(double, int n), double init_val, double epsilon, int n)
{
    double x;
    double x_n = init_val;

    double f_val;
    double df_val;

    do
    {
        x = x_n;
        std::tuple function_val_derivat = f(x, n);
        f_val = std::get<0>(function_val_derivat);
        df_val = std::get<1>(function_val_derivat);

        // std::cout << "x = " << x << " --> P_" << n << " = " << f_val << " --> P'_" << n << " = " << df_val << std::endl;

        x_n = x - f_val/df_val;
    } while (abs(x - x_n) > epsilon);

    return std::tuple<double, double, double>{x, f_val, df_val};
}
