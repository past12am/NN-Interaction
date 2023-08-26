//
// Created by past12am on 3/2/23.
//

#ifndef QUARKDSE_NEWTONROOTFINDER_HPP
#define QUARKDSE_NEWTONROOTFINDER_HPP


#include <tuple>
#include <cstdarg>

class NewtonRootFinder
{
    public:
        static std::tuple<double, double, double> estimateRoot(std::tuple<double, double> (*f)(double, int n), double init_val, double epsilon, int n);
};


#endif //QUARKDSE_NEWTONROOTFINDER_HPP
