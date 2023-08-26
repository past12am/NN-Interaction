//
// Created by past12am on 6/7/23.
//

#ifndef QUARKDSE_GSLFUNCTIONWRAPPER_HPP
#define QUARKDSE_GSLFUNCTIONWRAPPER_HPP


#include <gsl/gsl_math.h>

// https://stackoverflow.com/questions/13289311/c-function-pointers-with-c11-lambdas/18413206#18413206
template< typename F >  class GSLFunctionWrapper: public gsl_function
{
    public:
        GSLFunctionWrapper(const F& func) : _func(func) {
            function = &GSLFunctionWrapper::invoke;
            params=this;
        }
    private:
        const F& _func;
        static double invoke(double x, void *params) {
            return static_cast<GSLFunctionWrapper*>(params)->_func(x);
        }
};


#endif //QUARKDSE_GSLFUNCTIONWRAPPER_HPP
