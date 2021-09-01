
#include <Rcpp.h>
#include <RcppEigen.h>
using namespace Rcpp;

// This Cpp function compute eigenvalues and eigenvectors for matrix
// Learn more about Rcpp at:
//   http://www.rcpp.org/
//   http://adv-r.had.co.nz/Rcpp.html
//   http://gallery.rcpp.org/
//

// [[Rcpp::depends(RcppEigen)]]

using Eigen::Map;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::SelfAdjointEigenSolver;


// [[Rcpp::export]]

List getEigenVectors(Map<MatrixXd> M){
  SelfAdjointEigenSolver<MatrixXd> es(M);
  return List::create(_["eigenvalues"] = es.eigenvalues(),
                      _["eigenvectors"] = es.eigenvectors());
}

// You can include R code blocks in C++ files processed with sourceCpp
// (useful for testing and development). The R code will be automatically 
// run after the compilation.


