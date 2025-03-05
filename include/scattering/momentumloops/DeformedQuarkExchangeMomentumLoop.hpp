//
// Created by past12am on 3/5/25.
//

#ifndef DEFORMEDQUARKEXCHANGEMOMENTUMLOOP_HPP
#define DEFORMEDQUARKEXCHANGEMOMENTUMLOOP_HPP
#include "MomentumLoop.hpp"


#include "../../numerics/integration/GaussLegendre.hpp"


class DeformedQuarkExchangeMomentumLoop : public MomentumLoop
{
    private:
        GaussLegendre gaussLegendreIntegrator_k_4;
        GaussLegendre gaussLegendreIntegrator_absk;
        GaussLegendre gaussLegendreIntegrator_y;
        GaussLegendre gaussLegendreIntegrator_phi;

        gsl_complex k_4Integral(const std::function<gsl_complex(double, double, double, double)>& f, double lowerIntegrationBound, double upperIntegrationBound);
        gsl_complex abskIntegral(double k2, const std::function<gsl_complex(double, double, double, double)>& f);
        gsl_complex yIntegral(double k2, double z, const std::function<gsl_complex(double, double, double, double)>& f);
        gsl_complex phiIntegral(double k2, double z, double y, const std::function<gsl_complex(double, double, double, double)>& f);

    public:
        virtual gsl_complex integrate_4d(const std::function<gsl_complex(double, double, double, double)>& f, double cutoff);

        virtual void calc_k(double k_4, double absk, double y, double phi, gsl_vector_complex* k);

        DeformedQuarkExchangeMomentumLoop(int k2Points, int zPoints, int yPoints, int phiPoints);
        virtual ~DeformedQuarkExchangeMomentumLoop();
};



#endif //DEFORMEDQUARKEXCHANGEMOMENTUMLOOP_HPP
