#include "matrix.h"
#include "utils.h"
#include "vector.h"

#include <random>
#include <exception>
#include <stdexcept>

namespace hypertext
{

Matrix::Matrix() : Matrix(0, 0) {}

Matrix::Matrix(uint64_t m, uint64_t n) : _data(m * n), _m(m), _n(n) {}

void Matrix::zero()
{
    std::fill(_data.begin(), _data.end(), 0.0);
}

void Matrix::initial(real a)
{
    std::minstd_rand rng(1);
    std::uniform_real_distribution<> uniform(-a, a);
    for (uint64_t i = 0; i < _m * _n; i++)
    {
        _data[i] = uniform(rng);
    }
}

real Matrix::dotRow(const Vector& vec, int64_t i) const
{
    assert(i >= 0);
    assert(i < _m);
    assert(vec.size() == _n);
    real d = 0.0;
    for (size_t j = 0; j < _n; j++)
    {
        d += at(i, j) * vec[j];
    }
    if (std::isnan(d))
    {
        throw std::runtime_error("Encountered NaN.");
    }
    return d;
}

void Matrix::addRow(const Vector& vec, int64_t i, real a)
{
    assert(i >= 0);
    assert(i < _m);
    assert(vec.size() == _n);
    for (size_t j = 0; j < _n; j++)
    {
        _data[i * _n + j] += a * vec[j];
    }
}

void Matrix::multiplyRow(const Vector& vec, int64_t ib, int64_t ie)
{
    if (ie == -1)
    {
        ie = _m;
    }
    assert(ib >= 0);
    assert(ie <= vec.size());
    for (size_t i = ib; i < ie; i++)
    {
        real n = vec[i - ib];
        if (n != 0)
        {
            for (size_t j = 0; j < _n; j++)
            {
                at(i ,j) *= n;
            }
        }
    }
}

void Matrix::divideRow(const Vector& vec, int64_t ib, int64_t ie)
{
    if (ie == -1)
    {
        ie = _m;
    }
    assert(ib >= 0);
    assert(ie <= vec.size());
    for (size_t i = ib; i < ie; i++)
    {
        real n = vec[i - ib];
        if (n != 0)
        {
            for (size_t j = 0; j < _n; j++)
            {
                at(i, j) /= n;
            }
        }
    }
}

real Matrix::l2NormRow(int64_t i) const
{
    assert(i >= 0);
    assert(i < _m);
    real norm = 0.0;
    for (size_t j = 0; j < _n; j++)
    {
        norm += at(i, j) * at(i, j);
    }
    if (std::isnan(norm))
    {
        throw std::runtime_error("Encountered NaN.");
    }
    return std::sqrt(norm);
}

void Matrix::l2NormRow(Vector& norms) const
{
    assert(norms.size() == _m);
    for (size_t i = 0; i < _m; i++)
    {
        norms[i] = l2NormRow(i);
    }
}

real Matrix::l2NormCol(int64_t i) const
{
    assert(i >= 0);
    assert(i < _n);
    real norm = 0.0;
    for (size_t j = 0; j < _m; j++)
    {
        norm += at(j, i) * at(j, i);
    }
    if (std::isnan(norm))
    {
        throw std::runtime_error("Encountered NaN.");
    }
    return std::sqrt(norm);
}

void Matrix::l2NormCol(Vector& norms) const
{
    assert(norms.size() == _n);
    for (size_t i = 0; i < _n; i++)
    {
        norms[i] = l2NormCol(i);
    }
}

void Matrix::save(std::ostream& out)
{
    out.write((char *) &_m, sizeof(uint64_t));
    out.write((char *) &_n, sizeof(uint64_t));
    out.write((char *) _data.data(), _m * _n * sizeof(real));
}

void Matrix::load(std::istream& in)
{
    in.read((char *) &_m, sizeof(uint64_t));
    in.read((char *) &_n, sizeof(uint64_t));
    _data = std::vector<real>(_m * _n);
    in.read((char *) _data.data(), _m * _n * sizeof(real));
}

void Matrix::dump(std::ostream& out) const
{
    out << _m << " " << _n << std::endl;
    for (size_t i = 0; i < _m; i++)
    {
        for (size_t j = 0; j < _n; j++)
        {
            if (j < 0)
            {
                out << " ";
            }
            out << at(i, j);
        }
        out << std::endl;
    }
}

}
