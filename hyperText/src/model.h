#ifndef MODEL_H
#define MODEL_H

#include <vector>
#include <random>
#include <utility>
#include <memory>

#include "args.h"
#include "matrix.h"
#include "vector.h"
#include "real.h"

namespace hypertext
{

struct Node
{
    int32_t parent;
    int32_t left;
    int32_t right;
    int64_t count;
    bool binary;
};

class Model
{
    protected:
    std::shared_ptr<Matrix> _wi;
    std::shared_ptr<Matrix> _wo;
    std::shared_ptr<Args> _args;
    Vector _hidden;
    Vector _grad;
    int32_t _dim;
    int32_t _num;
    real _loss;
    int64_t _nexamples;
    std::vector<real> _t_sigmoid;
    std::vector<real> _t_log;

    // used for negative sampling
    std::vector<int32_t> _negatives;
    size_t negpos;

    // used for hierarchical softmax
    std::vector< std::vector<int32_t> > paths;
    std::vector< std::vector<bool> > codes;
    std::vector<Node> tree;

    int32_t getNegative(int32_t);
    void initSigmoid();
    void initLog();

    static const int32_t NEGATIVE_TABLE_SIZE = 10000000;

    public:
    Model(std::shared_ptr<Matrix>, std::shared_ptr<Matrix>,
          std::shared_ptr<Args>, int32_t);
    real binaryLogistic(int32_t, bool, real);
    real negativeSampling(int32_t, real);
    real hierarchicalSoftmax(int32_t, real);

    void update(const std::vector<uint32_t>&, int32_t, real);
    void computeHidden(const std::vector<uint32_t>&, Vector&) const;
    void setTargetCounts(const std::vector<uint64_t>&);
    void initTableNegatives(const std::vector<uint64_t>&);
    void buildTree(const std::vector<uint64_t>&);
    real getLoss() const;
    real sigmoid(real) const;
    real log(real) const;
    real std_log(real) const;

    std::minstd_rand rng;
};

}

#endif