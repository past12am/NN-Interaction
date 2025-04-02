//
// Created by past12am on 3/5/25.
//

#include "../../../include/scattering/momentumloops/DeformedQuarkExchangeMomentumLoop.hpp"

#include <gsl/gsl_complex_math.h>

#include "../../../include/Definitions.h"

gsl_complex DeformedQuarkExchangeMomentumLoop::x_4Integral(
    const std::function<gsl_complex(gsl_complex, double, double, double)>& f, double X, double epsilon, double eta)
{
    // TODO check k_4 = M_nucleon x_4 ?
    std::function<gsl_complex(gsl_complex)> x_4Integrand = [=, this](gsl_complex x_4) -> gsl_complex {
        return gsl_complex_mul_real(absxIntegral(x_4, f), M_nucleon);
    };

    return gaussLegendreIntegrator_x_4.integrateComplexDeformed(x_4Integrand, contour_parameterization, deriv_contour_parameterization,
        -CUTOFF_x_4, CUTOFF_x_4, X, epsilon, eta);
}

gsl_complex DeformedQuarkExchangeMomentumLoop::absxIntegral(gsl_complex x_4,
    const std::function<gsl_complex(gsl_complex, double, double, double)>& f)
{
    std::function<gsl_complex(double)> absxIntegrand = [=, this](double absx) -> gsl_complex {
        return gsl_complex_mul_real(yIntegral(x_4, absx, f), M_nucleon * M_nucleon * M_nucleon * absx * absx);      // TODO check k2 = M2 * absx2 and d_absk = M * d_absx
    };

    return gaussLegendreIntegrator_absx.integrateComplex(absxIntegrand, 0, CUTOFF_absx);
}

gsl_complex DeformedQuarkExchangeMomentumLoop::yIntegral(gsl_complex x_4, double absx,
    const std::function<gsl_complex(gsl_complex, double, double, double)>& f)
{
    std::function<gsl_complex(double)> yIntegrand = [=, this](double y) -> gsl_complex {
        return phiIntegral(x_4, absx, y, f);
    };

    return gaussLegendreIntegrator_y.integrateComplex(yIntegrand, -1, 1);
}

gsl_complex DeformedQuarkExchangeMomentumLoop::phiIntegral(gsl_complex x_4, double absx, double y,
    const std::function<gsl_complex(gsl_complex, double, double, double)>& f)
{
    std::function<gsl_complex(double)> phiIntegrand = [=, this](double phi) -> gsl_complex {
        return f(x_4, absx, y, phi);
    };

    return gaussLegendreIntegrator_phi.integrateComplex(phiIntegrand, -1, 1);
}

gsl_complex DeformedQuarkExchangeMomentumLoop::contour_parameterization(double t, double X, double epsilon, double eta)
{
    double f = 0.0;
    if(t <= -2 * epsilon || t >= 2 * epsilon)
    {
        f = 0.0;
    }
    else if(t <= 0)
    {
        double A_diquark = (1.0 - eta) * sqrt(1.0 + X);

        if(t <= -epsilon)  // diquark cut (f1+)
        {
            f = 2.0 * A_diquark + A_diquark / epsilon * t;
        }
        else  // diquark cut (f1-)
        {
            f = -A_diquark / epsilon * t;
        }
    }
    else if(t > 0)
    {
        double A_quark = -eta * sqrt(1.0 + X);

        if(t <= epsilon) // quark cut   (f2-)
        {
            f = -A_quark/epsilon * t;
        }
        else // quark cut   (f2+)
        {
            f = -2.0 * A_quark + A_quark/epsilon * t;
        }
    }
    else
    {
        throw std::invalid_argument("What does that t mean? --> Its likely invalid");
    }

    return gsl_complex_rect(t, f);
}

gsl_complex DeformedQuarkExchangeMomentumLoop::deriv_contour_parameterization(double t, double X, double epsilon, double eta)
{
    double df = 0.0;
    if(t <= -2 * epsilon || t >= 2 * epsilon)
    {
        df = 0.0;
    }
    else if(t <= 0)
    {
        double A_diquark = (1.0 - eta) * sqrt(1.0 + X);

        if(t <= -epsilon)
        {
            df = A_diquark / epsilon;
        }
        else
        {
            df = -A_diquark / epsilon;
        }
    }
    else if(t > 0)
    {
        double A_quark = -eta * sqrt(1.0 + X);

        if(t <= epsilon)
        {
            df = -A_quark/epsilon;
        }
        else
        {
            df = A_quark/epsilon;
        }
    }
    else
    {
        throw std::invalid_argument("What does that t mean? --> Its likely invalid");
    }

    return gsl_complex_rect(1.0, df);
}

gsl_complex DeformedQuarkExchangeMomentumLoop::integrate_4d_deformed(
    const std::function<gsl_complex(gsl_complex, double, double, double)>& f, double X, double epsilon, double eta)
{
    gsl_complex res = x_4Integral(f, X, epsilon, eta);
    res = gsl_complex_mul_real(res, 1.0/pow(2.0 * std::numbers::pi, 4) * M_nucleon);

    return res;
}

void DeformedQuarkExchangeMomentumLoop::calc_k_deformed(gsl_complex x_4, double absx, double y, double phi, gsl_vector_complex* k)
{
    gsl_vector_complex_set(k, 0, gsl_complex_rect(M_nucleon * absx * sqrt(1.0 - pow(y, 2)) * sin(phi), 0));
    gsl_vector_complex_set(k, 1, gsl_complex_rect(M_nucleon * absx * sqrt(1.0 - pow(y, 2)) * cos(phi), 0));
    gsl_vector_complex_set(k, 2, gsl_complex_rect(M_nucleon * absx * y, 0));
    gsl_vector_complex_set(k, 3, gsl_complex_mul_real(x_4, M_nucleon));
}

gsl_complex DeformedQuarkExchangeMomentumLoop::integrate_4d(
    const std::function<gsl_complex(double, double, double, double)>& f)
{
    throw std::invalid_argument("Cannot use non-deformed function with deformed contour");
}

void DeformedQuarkExchangeMomentumLoop::calc_k(double x_4, double absx, double y, double phi, gsl_vector_complex* k)
{
    throw std::invalid_argument("Cannot use non-deformed function with deformed contour");
}

DeformedQuarkExchangeMomentumLoop::DeformedQuarkExchangeMomentumLoop(int x_4Points, int absxPoints, int yPoints,
                                                                     int phiPoints) : gaussLegendreIntegrator_x_4(x_4Points),
                                                                                      gaussLegendreIntegrator_absx(absxPoints),
                                                                                      gaussLegendreIntegrator_y(yPoints),
                                                                                      gaussLegendreIntegrator_phi(phiPoints)
{}

DeformedQuarkExchangeMomentumLoop::~DeformedQuarkExchangeMomentumLoop()
{

}
