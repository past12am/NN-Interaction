//
// Created by past12am on 8/9/23.
//

#ifndef NNINTERACTION_TENSOR4_HPP
#define NNINTERACTION_TENSOR4_HPP

#include <complex>
#include <gsl/gsl_matrix.h>
#include <ostream>
#include "gsl/gsl_complex_math.h"

template<int d1, int d2, int d3, int d4> class Tensor4
{
    private:
        gsl_complex tensor[d1][d2][d3][d4];

    public:
        /*!
         * Creates a Tensor from an outer product between A and B
         * @param A
         * @param B
         */
        Tensor4(gsl_matrix_complex* A, gsl_matrix_complex* B)
        {
            for(int i = 0; i < A->size1; i++)
            {
                for(int j = 0; j < A->size2; j++)
                {
                    for(int k = 0; k < B->size1; k++)
                    {
                        for(int l = 0; l < B->size2; l++)
                        {
                            tensor[i][j][k][l] = gsl_complex_mul(gsl_matrix_complex_get(A, i, j), gsl_matrix_complex_get(B, k, l));
                        }
                    }
                }
            }
        }

        Tensor4()
        {
            for(int i = 0; i < d1; i++)
            {
                for (int j = 0; j < d2; j++)
                {
                    for (int k = 0; k < d3; k++)
                    {
                        for (int l = 0; l < d4; l++)
                        {
                            tensor[i][j][k][l] = gsl_complex_rect(0, 0);
                        }
                    }
                }
            }
        }

        Tensor4(Tensor4 &tensor4)
        {
            for(int i = 0; i < d1; i++)
            {
                for (int j = 0; j < d2; j++)
                {
                    for (int k = 0; k < d3; k++)
                    {
                        for (int l = 0; l < d4; l++)
                        {
                            tensor[i][j][k][l] = tensor4.tensor[i][j][k][l];
                        }
                    }
                }
            }
        }

        void setElement(size_t i, size_t j, size_t k, size_t l, gsl_complex x)
        {
            tensor[i][j][k][l] = x;
        }

        friend std::ostream &operator<<(std::ostream &os, const Tensor4 &tensor4)
        {
            os << "tensor4: " << tensor4.tensor;
            return os;
        }

        gsl_complex contractTauM(Tensor4<4, 4, 4, 4> &other)
        {
            gsl_complex res = gsl_complex_rect(0, 0);

            for(int i = 0; i < d1; i++)
            {
                for (int j = 0; j < d2; j++)
                {
                    for (int k = 0; k < d3; k++)
                    {
                        for (int l = 0; l < d4; l++)
                        {
                            gsl_complex_add(res, gsl_complex_mul(tensor[i][j][k][l], other.tensor[j][i][l][k]));
                        }
                    }
                }
            }

            return res;
        }
};


#endif //NNINTERACTION_TENSOR4_HPP
