//
// Created by past12am on 8/9/23.
//

#ifndef NNINTERACTION_TENSOR4_HPP
#define NNINTERACTION_TENSOR4_HPP

#include <complex>
#include <gsl/gsl_matrix.h>
#include <ostream>
#include <cassert>
#include <iostream>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_blas.h>

template<int da, int db, int dc, int dd> class Tensor22
{
    private:
        gsl_matrix_complex* tensor_a_b;
        gsl_matrix_complex* tensor_c_d;

    public:
        /*!
         * Creates a Tensor from an outer product between A and B
         * @param A
         * @param B
         */
        Tensor22(gsl_matrix_complex* A, gsl_matrix_complex* B)
        {
            tensor_a_b = gsl_matrix_complex_alloc(da, db);
            tensor_c_d = gsl_matrix_complex_alloc(dc, dd);

            gsl_matrix_complex_memcpy(tensor_a_b, A);
            gsl_matrix_complex_memcpy(tensor_c_d, B);
        }

        Tensor22(Tensor22 &other)
        {
            gsl_matrix_complex_memcpy(tensor_a_b, other.tensor_a_b);
            gsl_matrix_complex_memcpy(tensor_c_d, other.tensor_c_d);
        }

        Tensor22()
        {
            tensor_a_b = gsl_matrix_complex_alloc(da, db);
            tensor_c_d = gsl_matrix_complex_alloc(dc, dd);

            gsl_matrix_complex_set_zero(tensor_a_b);
            gsl_matrix_complex_set_zero(tensor_c_d);
        }

        virtual ~Tensor22()
        {
            gsl_matrix_complex_free(tensor_a_b);
            gsl_matrix_complex_free(tensor_c_d);
        }

        Tensor22<da, db, dc, dd>& operator=(const Tensor22<da, db, dc, dd>& other)
        {
            gsl_matrix_complex_memcpy(tensor_a_b, other.tensor_a_b);
            gsl_matrix_complex_memcpy(tensor_c_d, other.tensor_c_d);

            return *this;
        }

        Tensor22<da, db, dc, dd> operator*(const gsl_complex& scalar)
        {
            Tensor22<da, db, dc, dd> res;
            res = *this;

            gsl_matrix_complex_scale(res.tensor_a_b, scalar);
            gsl_matrix_complex_scale(res.tensor_c_d, scalar);

            return res;
        }

        Tensor22<da, db, dc, dd>& operator+=(const Tensor22<da, db, dc, dd>& other)
        {
            gsl_matrix_complex_add(tensor_a_b, other.tensor_a_b);
            gsl_matrix_complex_add(tensor_c_d, other.tensor_c_d);

            return *this;
        }

        void set_tensors(gsl_matrix_complex* t_a_b, gsl_matrix_complex* t_c_d)
        {
            gsl_matrix_complex_memcpy(tensor_a_b, t_a_b);
            gsl_matrix_complex_memcpy(tensor_c_d, t_c_d);
        }

        gsl_complex contractTauOther(Tensor22<4, 4, 4, 4>* other)
        {
            gsl_complex res_a_b = gsl_complex_rect(0, 0);
            gsl_complex res_c_d = gsl_complex_rect(0, 0);

            for(int a = 0; a < da; a++)
            {
                for(int b = 0; b < db; b++)
                {
                    res_a_b = gsl_complex_add(gsl_complex_mul(gsl_matrix_complex_get(tensor_a_b, a, b),
                                                              gsl_matrix_complex_get(other->tensor_a_b, b, a)),
                                              res_a_b);
                }
            }
            //assert(GSL_IMAG(res_a_b) == 0);

            for(int c = 0; c < dc; c++)
            {
                for(int d = 0; d < dd; d++)
                {
                    res_c_d = gsl_complex_add(gsl_complex_mul(gsl_matrix_complex_get(tensor_c_d, c, d),
                                                                 gsl_matrix_complex_get(other->tensor_c_d, d, c)),
                                              res_c_d);
                }
            }
            //assert(GSL_IMAG(res_c_d) == 0);

            return gsl_complex_mul(res_a_b, res_c_d);
        }
};

#endif //NNINTERACTION_TENSOR4_HPP
