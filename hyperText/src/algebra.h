#ifndef ALGEBRA_H
#define ALGEBRA_H

#include "real.h"

#include "Eigen/Sparse"
#include "Eigen/Dense"
#include "Eigen/Core"
#include "Eigen/Eigenvalues"

namespace hypertext
{

// fastsvd
typedef Eigen::SparseMatrix<real, Eigen::RowMajor> SMatrixXf;
const real SVD_EPS = 0.00001f;

class Algebra
{
    protected:
    // fastsvd
    void sampleTwoGaussian(real&, real&);

    public:
    // fastsvd 
    void sampleGaussianMatrix(Eigen::MatrixXf&);
    void GramSchmidt(Eigen::MatrixXf&);
};

}

#endif