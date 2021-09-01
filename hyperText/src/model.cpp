#include "model.h"

#include <iostream>
#include <assert.h>
#include <algorithm>
#include <stdexcept>

namespace hypertext
{

constexpr int64_t SIGMOID_TABLE_SIZE = 512;
constexpr int64_t MAX_SIGMOID = 8;
constexpr int64_t LOG_TABLE_SIZE = 512;

Model::Model(
    std::shared_ptr<Matrix> wi,
    std::shared_ptr<Matrix> wo,
    std::shared_ptr<Args> args,
    int32_t seed)
    : _hidden(args->dim),
      _grad(args->dim),
      rng(seed)
{
    _wi = wi;
    _wo = wo;
    _args = args;
    _dim = args->dim;
    _num = wo->size(0);
    negpos = 0;
    _loss = 0.0;
    _nexamples = 1;
    _t_sigmoid.reserve(SIGMOID_TABLE_SIZE + 1);
    _t_log.reserve(LOG_TABLE_SIZE + 1);
    initSigmoid();
    initLog();
}

real Model::hierarchicalSoftmax(int32_t target, real lr)
{
    real loss = 0.0;
    _grad.zero();
    const std::vector<int32_t>& pathToRoot = paths[target];
    const std::vector<bool>& binaryCode = codes[target];
    // traverse from leaf to root node
    for (int32_t i = 0; i < pathToRoot.size(); i++)
    {
        loss += binaryLogistic(pathToRoot[i], binaryCode[i], lr);
    }
    return loss;
}

real Model::negativeSampling(int32_t target, real lr)
{
    real loss = 0.0;
    _grad.zero();
    for (int32_t n = 0; n < _args->neg; n++)
    {
        // negative sampling update
        if (n == 0)
        {
            loss += binaryLogistic(target, true, lr);
        }
        else
        {
            loss += binaryLogistic(getNegative(target), false, lr);
        }
    }
    return loss;
}

real Model::binaryLogistic(int32_t target, bool label, real lr)
{
    real score = sigmoid(_wo->dotRow(_hidden, target));
    real alpha = lr * (real(label) - score);
    _grad.addRow(*(_wo), target, alpha);
    _wo->addRow(_hidden, target, alpha);
    if (label)
    {
        return -log(score);
    }
    else
    {
        return -log(1.0 - score);
    }
}

void Model::computeHidden(const std::vector<uint32_t>& input, Vector& hidden) const
{
    assert(hidden.size() == _dim);
    hidden.zero();
    for (auto it = input.cbegin(); it != input.cend(); it++)
    {
        hidden.addRow(*(_wi), *it);
    }
    hidden.mul(1.0 / input.size());
}

void Model::update(const std::vector<uint32_t>& input, int32_t target, real lr)
{
    assert(target >= 0);
    assert(target < _num);
    if (input.size() == 0)  return;
    // compute hidden vector
    computeHidden(input, _hidden);
    if (_args->loss == loss_name::ns)
    {
        _loss += negativeSampling(target, lr);
    }
    else if (_args->loss == loss_name::hs)
    {
        _loss += hierarchicalSoftmax(target, lr);
    }
    _nexamples++;

    for (auto it = input.cbegin(); it != input.cend(); it++)
    {
        _wi->addRow(_grad, *it, 1.0);
    }
}

void Model::buildTree(const std::vector<uint64_t>& counts)
{
    tree.resize(2 * _num - 1);
    // initialize huffman tree
    for (int32_t i = 0; i < 2 * _num - 1; i++)
    {
        tree[i].parent = -1;
        tree[i].left = -1;
        tree[i].right = -1;
        tree[i].count = 1e15;
        tree[i].binary = false;
    }

    for (int32_t i = 0; i < _num; i++)
    {
        tree[i].count = counts[i];
    }

    int32_t leaf = _num - 1;
    int32_t node = _num;
    for (int32_t i = _num; i < 2 * _num - 1; i++)
    {
        int32_t mininode[2];
        // search two minimal nodes
        for (int32_t j = 0; j < 2; j++)
        {
            if (leaf >= 0 && tree[leaf].count < tree[node].count)
            {
                mininode[j] = leaf;
                leaf--;
            }
            else
            {
                mininode[j] = node;
                node++;
            }
        }
        tree[i].left = mininode[0];
        tree[i].right = mininode[1];
        tree[i].count = tree[mininode[0]].count + tree[mininode[1]].count;
        tree[mininode[0]].parent = i;
        tree[mininode[1]].parent = i;
        tree[mininode[1]].binary = true;
    }

    // get left to root path and binary code
    for (int32_t i = 0; i < _num; i++)
    {
        std::vector<int32_t> path;
        std::vector<bool> code;
        int32_t j = i;
        while (tree[j].parent != -1)
        {
            path.push_back(tree[j].parent - _num);
            code.push_back(tree[j].binary);
            j = tree[j].parent;
        }
        paths.push_back(path);
        codes.push_back(code);
    }
}

real Model::getLoss() const
{
    return _loss / _nexamples;
}

void Model::setTargetCounts(const std::vector<uint64_t>& counts)
{
    assert(counts.size() == _num);
    if (_args->loss == loss_name::ns)
    {
        initTableNegatives(counts);
    }
    if (_args->loss == loss_name::hs)
    {
        buildTree(counts);
    }
}

void Model::initTableNegatives(const std::vector<uint64_t>& counts)
{
    real z = 0.0;
    for (uint32_t i = 0; i < counts.size(); i++)
    {
        z += pow(counts[i], 0.5);
    }

    for (uint32_t i = 0; i < counts.size(); i++)
    {
        real p = pow(counts[i], 0.5);
        for (uint32_t j = 0; j < p * NEGATIVE_TABLE_SIZE / z; j++)
        {
            _negatives.push_back(i);
        }
    }
    std::shuffle(_negatives.begin(), _negatives.end(), rng);
}

int32_t Model::getNegative(int32_t target)
{
    int32_t negative;
    do
    {
        negative = _negatives[negpos];
        negpos = (negpos + 1) % _negatives.size();
    } while (target == negative);
    return negative;
}

void Model::initSigmoid()
{
    for (int i = 0; i < SIGMOID_TABLE_SIZE; i++)
    {
        real x = real(i * 2 * MAX_SIGMOID) / SIGMOID_TABLE_SIZE - MAX_SIGMOID;
        _t_sigmoid.push_back(1.0 / (1.0 + std::exp(-x)));
    }
}

real Model::sigmoid(real x) const
{
    if (x < -MAX_SIGMOID)
    {
        return 0.0;
    }
    else if (x > MAX_SIGMOID)
    {
        return 1.0;
    }
    else
    {
        int64_t i = int64_t((x + MAX_SIGMOID) * SIGMOID_TABLE_SIZE / MAX_SIGMOID / 2);
        return _t_sigmoid[i];
    }
}

void Model::initLog()
{
    for (int i = 0; i < LOG_TABLE_SIZE; i++)
    {
        real x = (real(i) + 1e-5) / LOG_TABLE_SIZE;
        _t_log.push_back(std::log(x));
    }
}

real Model::log(real x) const
{
    if (x > 1.0)
    {
        return 0.0;
    }
    int64_t i = int64_t(x * LOG_TABLE_SIZE);
    return _t_log[i];
}

real Model::std_log(real x) const
{
    return std::log(x + 1e-5);
}

}