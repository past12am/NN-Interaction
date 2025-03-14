//
// Created by past12am on 3/5/25.
//

#ifndef DEFORMEDQUARKEXCHANGEMOMENTUMLOOP_HPP
#define DEFORMEDQUARKEXCHANGEMOMENTUMLOOP_HPP
#include "MomentumLoop.hpp"


#include "../../numerics/integration/GaussLegendre.hpp"
#include "../../numerics/integration/GaussLegendreDeformed.hpp"


class DeformedQuarkExchangeMomentumLoop : public MomentumLoop
{
    private:
        GaussLegendreDeformed gaussLegendreIntegrator_x_4;
        GaussLegendre gaussLegendreIntegrator_absx;
        GaussLegendre gaussLegendreIntegrator_y;
        GaussLegendre gaussLegendreIntegrator_phi;

        gsl_complex x_4Integral(const std::function<gsl_complex(gsl_complex, double, double, double)>& f, double lowerIntegrationBound, double upperIntegrationBound);
        gsl_complex absxIntegral(gsl_complex x_4, const std::function<gsl_complex(gsl_complex, double, double, double)>& f, double upperIntegrationBound);
        gsl_complex yIntegral(gsl_complex x_4, double absx, const std::function<gsl_complex(gsl_complex, double, double, double)>& f);
        gsl_complex phiIntegral(gsl_complex x_4, double absx, double y, const std::function<gsl_complex(gsl_complex, double, double, double)>& f);

    public:
        gsl_complex integrate_4d_deformed(const std::function<gsl_complex(gsl_complex, double, double, double)>& f, double cutoff) override;
        void calc_k_deformed(gsl_complex x_4, double absx, double y, double phi, gsl_vector_complex* k) override;

        gsl_complex integrate_4d(const std::function<gsl_complex(double, double, double, double)>& f, double cutoff) override;
        void calc_k(double x_4, double absx, double y, double phi, gsl_vector_complex* k) override;

        DeformedQuarkExchangeMomentumLoop(int x_4Points, int absxPoints, int yPoints, int phiPoints);
        virtual ~DeformedQuarkExchangeMomentumLoop();
};



#endif //DEFORMEDQUARKEXCHANGEMOMENTUMLOOP_HPP
