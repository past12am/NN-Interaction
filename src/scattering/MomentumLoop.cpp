//
// Created by past12am on 8/25/23.
//

#include "../../include/scattering/MomentumLoop.hpp"

#include "gsl/gsl_complex_math.h"



MomentumLoop::~MomentumLoop()
{

}

MomentumLoop::MomentumLoop(int k2Points, int zPoints, int yPoints, int phiPoints) : gaussLegendreIntegrator_k2(k2Points),
                                                                                    gaussLegendreIntegrator_y(yPoints),
                                                                                    gaussLegendreIntegrator_phi(phiPoints),
                                                                                    gaussChebyshevIntegrator_z(zPoints)
{

}

gsl_complex MomentumLoop::k2Integral(const std::function<gsl_complex(double, double, double, double)> &f,
                                     double lowerIntegrationBound, double upperIntegrationBound)
{
    std::function<gsl_complex(double)> k2Integrand = [=, this](double k2) -> gsl_complex {
        return gsl_complex_mul_real(zIntegral(k2, f), k2);
    };

    return gaussLegendreIntegrator_k2.integrateComplex(k2Integrand, lowerIntegrationBound, upperIntegrationBound);
}

gsl_complex MomentumLoop::zIntegral(double k2, const std::function<gsl_complex(double, double, double, double)> &f)
{
    std::function<gsl_complex(double)> zIntegrand = [=, this](double z) -> gsl_complex {
        return yIntegral(k2, z, f);
    };

    return gaussChebyshevIntegrator_z.integrate_complex_f_times_sqrt(zIntegrand);
}

gsl_complex MomentumLoop::yIntegral(double k2, double z, const std::function<gsl_complex(double, double, double, double)> &f)
{
    std::function<gsl_complex(double)> yIntegrand = [=, this](double y) -> gsl_complex {
        return phiIntegral(k2, z, y, f);
    };

    return gaussLegendreIntegrator_y.integrateComplex(yIntegrand, -1, 1);
}

gsl_complex MomentumLoop::phiIntegral(double k2, double z, double y, const std::function<gsl_complex(double, double, double, double)> &f)
{
    std::function<gsl_complex(double)> phiIntegrand = [=, this](double phi) -> gsl_complex {
        return f(k2, z, y, phi);
    };

    return gaussLegendreIntegrator_phi.integrateComplex(phiIntegrand, 0.0, 2.0 * std::numbers::pi);
}
