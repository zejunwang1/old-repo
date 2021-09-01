#include "algebra.h"

#include <iostream>

namespace hypertext
{

void Algebra::GramSchmidt(Eigen::MatrixXf& mat)
{
    for (int i = 0; i < mat.cols(); i++)
    {
        for (int j = 0; j < i; j++)
        {
            real r = mat.col(i).dot(mat.col(j));
            mat.col(i) -= r * mat.col(j);
        }
        real norm = mat.col(i).norm();
        if (norm < SVD_EPS)
        {
            for (int k = i; k < mat.cols(); k++)
            {
                mat.col(k).setZero();
            }
            return;
        }
        mat.col(i) *= (1.f / norm);
    }
}

void Algebra::sampleGaussianMatrix(Eigen::MatrixXf& mat)
{
    for (int i = 0; i < mat.cols(); i++)
    {
        int j = 0;
        while (j + 1 < mat.cols())
        {
            real f1, f2;
            sampleTwoGaussian(f1, f2);
            mat(i, j) = f1;
            mat(i, j + 1) = f2;
            j += 2;
        }
        while (j < mat.cols())
        {
            real f1, f2;
            sampleTwoGaussian(f1, f2);
            mat(i, j) = f1;
            j++;
        }
    }
}

void Algebra::sampleTwoGaussian(real& f1, real& f2)
{
    real v1 = (real)(rand() + 1.f) / ((real)RAND_MAX + 2.f);
    real v2 = (real)(rand() + 1.f) / ((real)RAND_MAX + 2.f);
    real len = std::sqrt(-2.f * (std::log(v1)));
    f1 = len * (std::cos(2.f * M_PI * v2));
    f2 = len * (std::sin(2.f * M_PI * v2));
}

}
