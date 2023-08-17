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
        static const gsl_matrix_complex* unitM;

    public:
        static void posEnergyProjector(gsl_vector_complex* P, gsl_matrix_complex* posEnergyProj);
        static void transverseProjector(gsl_vector_complex* P, gsl_matrix_complex* transvProj);
        static void longitudinalProjector(gsl_vector_complex* P, gsl_matrix_complex* longitudProj);

        static const gsl_matrix_complex* getUnitM();
};


#endif //NNINTERACTION_PROJECTORSHELPER_HPP
