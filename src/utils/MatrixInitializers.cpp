//
// Created by past12am on 8/2/23.
//

#include "../../include/utils/MatrixInitializers.hpp"

static const gsl_matrix_complex* generateUnitM()
{
    gsl_matrix_complex* unitM = gsl_matrix_complex_alloc(4, 4);
    gsl_matrix_complex_set_identity(unitM);

    return unitM;
}

const gsl_matrix_complex* MatrixInitializers::unitM = generateUnitM();
