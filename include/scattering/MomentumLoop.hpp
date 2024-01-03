//
// Created by past12am on 8/25/23.
//

#ifndef NNINTERACTION_MOMENTUMLOOP_HPP
#define NNINTERACTION_MOMENTUMLOOP_HPP


#include "impulse/LoopImpulseGrid.hpp"
#include "../numerics/integration/GaussLegendre.hpp"
#include "../numerics/integration/GaussChebyshev.hpp"

class MomentumLoop
{
    private:
        GaussLegendre gaussLegendreIntegrator_k2;
        GaussLegendre gaussLegendreIntegrator_y;
        GaussLegendre gaussLegendreIntegrator_phi;
        GaussChebyshev gaussChebyshevIntegrator_z;

    public:
        gsl_complex k2Integral(const std::function<gsl_complex(double, double, double, double)>& k2, double lowerIntegrationBound, double upperIntegrationBound);
        gsl_complex zIntegral(double k2, const std::function<gsl_complex(double, double, double, double)>& f);
        gsl_complex yIntegral(double k2, double z, const std::function<gsl_complex(double, double, double, double)>& f);
        gsl_complex phiIntegral(double k2, double z, double y, const std::function<gsl_complex(double, double, double, double)>& f);

        MomentumLoop(int k2Points, int zPoints, int yPoints, int phiPoints);
        virtual ~MomentumLoop();
};


#endif //NNINTERACTION_MOMENTUMLOOP_HPP
