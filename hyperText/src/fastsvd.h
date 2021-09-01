#ifndef FASTSVD_H
#define FASTSVD_H

#include <vector>

#include "algebra.h"

#include "Eigen/Sparse"
#include "Eigen/Dense"
#include "Eigen/Core"
#include "Eigen/Eigenvalues"

namespace hypertext
{

class FastSVD
{
    protected:
    Eigen::MatrixXf _matU;
    Eigen::VectorXf _vecS;
    Eigen::MatrixXf _matV;

    public:
    const Eigen::MatrixXf& matrixU() const
    {
        return _matU;
    }

    const Eigen::VectorXf& singularValues() const
    {
        return _vecS;
    }

    const Eigen::MatrixXf& matrixV() const
    {
        return _matV;
    }

    FastSVD() {}

    template <class Mat>
    FastSVD(Mat& A)
    {
        int r = (A.rows() < A.cols()) ? A.rows() : A.cols();
        svd(A, r);
    }

    template <class Mat>
    FastSVD(Mat& A, const int r)
    {
        svd(A, r);
    }

    template <class Mat>
    void svd(Mat& A, const int r)
    {
        if (A.rows() == 0 || A.cols() == 0)
        {
            return;
        }

        int rank = (r < A.rows()) ? r : A.rows();
        rank = (rank < A.cols()) ? rank : A.cols();

        // Generate gaussian random matrix
        Eigen::MatrixXf O(A.rows(), rank);
		Algebra alg;
        alg.sampleGaussianMatrix(O);

        // Compute sample matrix of A
        Eigen::MatrixXf Y = A.transpose() * O;

        // Orthonormalize Y
        alg.GramSchmidt(Y);

        Eigen::MatrixXf B = A * Y;

        // Gaussian Random Matrix
        Eigen::MatrixXf P(B.cols(), rank);
        alg.sampleGaussianMatrix(P);

        // Compute sample matrix of B
        Eigen::MatrixXf Z = B * P;

        // Orthonormalize Z
        alg.GramSchmidt(Z);

        Eigen::MatrixXf C = Z.transpose() * B;

        // JacobiSVD
        Eigen::JacobiSVD<Eigen::MatrixXf> svdOfC(C, Eigen::ComputeThinU | Eigen::ComputeThinV);

        _matU = Z * svdOfC.matrixU();
        _vecS = svdOfC.singularValues();
        _matV = Y * svdOfC.matrixV();
    }
};

class FastPCA
{
    protected:
    Eigen::MatrixXf _principalComponents;
    Eigen::MatrixXf _scores;

    public:
    const Eigen::MatrixXf& principalComponents() const
    {
        return _principalComponents;
    }

    const Eigen::MatrixXf& scores() const
    {
        return _scores;
    }

    FastPCA() {}

    template <class Mat>
    FastPCA(Mat& A, const int r)
    {
        FastSVD fsvd;
        fsvd.svd(A, r);
        const Eigen::VectorXf& S = fsvd.singularValues();
        _principalComponents = fsvd.matrixV();
        _scores = fsvd.matrixU() * S.asDiagonal();
    }
};

class FastSymEigen
{
    protected:
    Eigen::VectorXf _eigenValues;
    Eigen::MatrixXf _eigenVectors;

    public:
    const Eigen::VectorXf& eigenValues() const
    {
        return _eigenValues;
    }

    const Eigen::MatrixXf& eigenVectors() const
    {
        return _eigenVectors;
    }

    FastSymEigen() {}

    template <class Mat>
    FastSymEigen(Mat& A, const int r)
    {
        symEigen(A, r);
    }

    template <class Mat>
    void symEigen(Mat& A, const int r)
    {
        if (A.rows() == 0 || A.cols() == 0)
        {
            return;
        }

        int rank = (r < A.rows()) ? r : A.rows();
        rank = (rank < A.cols()) ? rank : A.cols();

        // generate gaussian random matrix
        Eigen::MatrixXf O(A.rows(), rank);
		Algebra alg;
        alg.sampleGaussianMatrix(O);

        // Compute sample matrix of A
        Eigen::MatrixXf Y = A.transpose() * O;

        // Orthonormalize Y
        alg.GramSchmidt(Y);

        Eigen::MatrixXf B = Y.transpose() * A * Y;
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> eigenOfB(B);

        _eigenValues = eigenOfB.eigenvalues();
        _eigenVectors = Y * eigenOfB.eigenvectors();
    }
};

}

#endif
