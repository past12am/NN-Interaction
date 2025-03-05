//
// Created by past12am on 8/25/23.
//

#include "../../../include/scattering/momentumloops/QuarkExchangeMomentumLoop.hpp"

#include "gsl/gsl_complex_math.h"



QuarkExchangeMomentumLoop::~QuarkExchangeMomentumLoop()
{

}

QuarkExchangeMomentumLoop::QuarkExchangeMomentumLoop(int k2Points, int zPoints, int yPoints, int phiPoints) : gaussLegendreIntegrator_k2(k2Points),
                                                                                    gaussLegendreIntegrator_y(yPoints),
                                                                                    gaussLegendreIntegrator_phi(phiPoints),
                                                                                    gaussChebyshevIntegrator_z(zPoints)
{

}

gsl_complex QuarkExchangeMomentumLoop::k2Integral(const std::function<gsl_complex(double, double, double, double)> &f,
                                     double lowerIntegrationBound, double upperIntegrationBound)
{
    std::function<gsl_complex(double)> k2Integrand = [=, this](double k2) -> gsl_complex {
        return gsl_complex_mul_real(zIntegral(k2, f), k2);
    };

    return gaussLegendreIntegrator_k2.integrateComplex(k2Integrand, lowerIntegrationBound, upperIntegrationBound);
}

gsl_complex QuarkExchangeMomentumLoop::zIntegral(double k2, const std::function<gsl_complex(double, double, double, double)> &f)
{
    std::function<gsl_complex(double)> zIntegrand = [=, this](double z) -> gsl_complex {
        return yIntegral(k2, z, f);
    };

    return gaussChebyshevIntegrator_z.integrate_complex_f_times_sqrt(zIntegrand);
}

gsl_complex QuarkExchangeMomentumLoop::yIntegral(double k2, double z, const std::function<gsl_complex(double, double, double, double)> &f)
{
    std::function<gsl_complex(double)> yIntegrand = [=, this](double y) -> gsl_complex {
        return phiIntegral(k2, z, y, f);
    };

    return gaussLegendreIntegrator_y.integrateComplex(yIntegrand, -1, 1);
}

gsl_complex QuarkExchangeMomentumLoop::phiIntegral(double k2, double z, double y, const std::function<gsl_complex(double, double, double, double)> &f)
{
    std::function<gsl_complex(double)> phiIntegrand = [=, this](double phi) -> gsl_complex {
        return f(k2, z, y, phi);
    };

    return gaussLegendreIntegrator_phi.integrateComplex(phiIntegrand, 0.0, 2.0 * std::numbers::pi);
}

gsl_complex QuarkExchangeMomentumLoop::integrate_4d(const std::function<gsl_complex(double, double, double, double)>& f, double cutoff)
{
    gsl_complex res = k2Integral(f, 0, cutoff);
    res = gsl_complex_mul_real(res, 1.0/pow(2.0 * std::numbers::pi, 4) * 0.5);

    return res;
}

void QuarkExchangeMomentumLoop::calc_k(double k2, double z, double y, double phi, gsl_vector_complex* k)
{
    gsl_vector_complex_set(k, 0, gsl_complex_rect(sqrt(1.0 - pow(z, 2)) * sqrt(1.0 - pow(y, 2)) * sin(phi), 0));
    gsl_vector_complex_set(k, 1, gsl_complex_rect(sqrt(1.0 - pow(z, 2)) * sqrt(1.0 - pow(y, 2)) * cos(phi), 0));
    gsl_vector_complex_set(k, 2, gsl_complex_rect(sqrt(1.0 - pow(z, 2)) * y, 0));
    gsl_vector_complex_set(k, 2, gsl_complex_rect(z, 0));

    gsl_vector_complex_scale(k, gsl_complex_rect(sqrt(k2), 0));
}
