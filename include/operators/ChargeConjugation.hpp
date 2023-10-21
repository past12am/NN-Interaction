//
// Created by past12am on 8/3/23.
//

#ifndef NNINTERACTION_CHARGECONJUGATION_HPP
#define NNINTERACTION_CHARGECONJUGATION_HPP

#include "../Definitions.h"

#include <complex>
#include <gsl/gsl_matrix.h>
#include <mutex>

class ChargeConjugation
{
    private:
        static std::mutex tmpMutexArray[NUM_THREADS];
        static gsl_matrix_complex** tmpArray;


        static void chargeConj(gsl_matrix_complex* mat, gsl_matrix_complex* tmp);

    public:
        static const gsl_matrix_complex* chargeConjMatrix;

        static void chargeConj(gsl_matrix_complex* mat);
        static void chargeConj(gsl_matrix_complex* mat, int thread_number);
};


#endif //NNINTERACTION_CHARGECONJUGATION_HPP
