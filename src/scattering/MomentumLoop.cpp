//
// Created by past12am on 8/25/23.
//

#include "../../include/scattering/MomentumLoop.hpp"

#include "gsl/gsl_complex_math.h"



MomentumLoop::~MomentumLoop()
{

}

MomentumLoop::MomentumLoop(int l2Points, int zPoints, int yPoints, int phiPoints) : gaussLegendreIntegrator_l2(l2Points),
                                                                                    gaussChebyshevIntegrator_z(zPoints),
                                                                                    gaussLegendreIntegrator_y(yPoints),
                                                                                    gaussLegendreIntegrator_phi(phiPoints)
{

}

gsl_complex MomentumLoop::l2Integral(const std::function<gsl_complex(double, double, double, double)> &f,
                                     double lowerIntegrationBound, double upperIntegrationBound)
{
    std::function<gsl_complex(double)> l2Integrand = [=, this](double l2) -> gsl_complex {
        return gsl_complex_mul_real(zIntegral(l2, f), l2);
    };

    return gaussLegendreIntegrator_l2.integrateComplex(l2Integrand, lowerIntegrationBound, upperIntegrationBound);
}

gsl_complex MomentumLoop::zIntegral(double l2, const std::function<gsl_complex(double, double, double, double)> &f)
{
    std::function<gsl_complex(double)> zIntegrand = [=, this](double z) -> gsl_complex {
        return yIntegral(l2, z, f);
    };

    return gaussChebyshevIntegrator_z.integrate_complex_f_times_sqrt(zIntegrand);
}

gsl_complex MomentumLoop::yIntegral(double l2, double z, const std::function<gsl_complex(double, double, double, double)> &f)
{
    std::function<gsl_complex(double)> yIntegrand = [=, this](double y) -> gsl_complex {
        return phiIntegral(l2, z, y, f);
    };

    return gaussLegendreIntegrator_y.integrateComplex(yIntegrand, -1, 1);
}

gsl_complex MomentumLoop::phiIntegral(double l2, double z, double y, const std::function<gsl_complex(double, double, double, double)> &f)
{
    std::function<gsl_complex(double)> phiIntegrand = [=, this](double phi) -> gsl_complex {
        return f(l2, z, y, phi);
    };

    return gaussLegendreIntegrator_phi.integrateComplex(phiIntegrand, 0.0, 2.0 * std::numbers::pi);
}
