//
// Created by past12am on 8/25/23.
//

#ifndef NNINTERACTION_MOMENTUMLOOP_HPP
#define NNINTERACTION_MOMENTUMLOOP_HPP


#include "MomentumLoop.hpp"
#include "../impulse/LoopImpulseGrid.hpp"
#include "../../numerics/integration/GaussLegendre.hpp"
#include "../../numerics/integration/GaussChebyshev.hpp"

class QuarkExchangeMomentumLoop : public MomentumLoop
{
    private:
        GaussLegendre gaussLegendreIntegrator_k2;
        GaussLegendre gaussLegendreIntegrator_y;
        GaussLegendre gaussLegendreIntegrator_phi;
        GaussChebyshev gaussChebyshevIntegrator_z;

        gsl_complex k2Integral(const std::function<gsl_complex(double, double, double, double)>& f, double lowerIntegrationBound, double upperIntegrationBound);
        gsl_complex zIntegral(double k2, const std::function<gsl_complex(double, double, double, double)>& f);
        gsl_complex yIntegral(double k2, double z, const std::function<gsl_complex(double, double, double, double)>& f);
        gsl_complex phiIntegral(double k2, double z, double y, const std::function<gsl_complex(double, double, double, double)>& f);

    public:
        virtual gsl_complex integrate_4d(const std::function<gsl_complex(double, double, double, double)>& f, double cutoff);

        QuarkExchangeMomentumLoop(int k2Points, int zPoints, int yPoints, int phiPoints);
        virtual ~QuarkExchangeMomentumLoop();
};


#endif //NNINTERACTION_MOMENTUMLOOP_HPP
