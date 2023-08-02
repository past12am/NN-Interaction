//
// Created by past12am on 8/2/23.
//

#ifndef NNINTERACTION_PROJECTORSHELPER_HPP
#define NNINTERACTION_PROJECTORSHELPER_HPP

#include <complex>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>

class ProjectorsHelper
{
    private:
        gsl_matrix_complex* unitM;

    public:
        ProjectorsHelper();

    public:
        void posEnergyProjector(gsl_vector_complex* P, gsl_matrix_complex* posEnergyProj);
        void transverseProjector(gsl_vector_complex* P, gsl_matrix_complex* transvProj);
        void longitudinalProjector(gsl_vector_complex* P, gsl_matrix_complex* longitudProj);

        virtual ~ProjectorsHelper();
};


#endif //NNINTERACTION_PROJECTORSHELPER_HPP
