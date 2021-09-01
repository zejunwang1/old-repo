// LSA implementation 
// Need to load sparse matrix to memory

#ifndef LSA_H
#define LSA_H

#include "args.h"
#include "real.h"
#include "dictionary.h"
#include "fastsvd.h"
#include "algebra.h"

#include <istream>
#include <ostream>
#include <memory>

namespace hypertext
{

class Lsa
{
    protected:
    uint32_t capacity;
    std::shared_ptr<Args> _args;
    std::shared_ptr<Dictionary> _dict;
    std::vector<real> _wsum;    // word count statistic
    std::vector<real> _csum;    // context count statistic

    public:
    Lsa(std::shared_ptr<Args>, std::string);
    void getPMIMatrix(std::string, SMatrixXf&);
    void savePMITriples(std::string);
    void getLsaVectors(std::string, Eigen::MatrixXf&);
    void saveSparseMatrix(std::ostream&, const SMatrixXf&);
    void saveDenseMatrix(std::ostream&, const Eigen::MatrixXf&);
};

}

#endif
