#include "vector.h"
#include "matrix.h"

#include <assert.h>

#include <iomanip>
#include <cmath>

namespace hypertext
{

Vector::Vector(uint64_t n) : _data(n) {}

void Vector::zero()
{
    std::fill(_data.begin(), _data.end(), 0.0);
}

void Vector::mul(real a)
{
    for (size_t i = 0; i < size(); i++)
    {
        _data[i] *= a;
    }
}

void Vector::mul(const Matrix& mat, const Vector& vec)
{
    assert(mat.size(0) == size());
    assert(mat.size(1) == vec.size());
    for (size_t i = 0; i < size(); i++)
    {
        _data[i] = mat.dotRow(vec, i);
    }
}

real Vector::l2Norm() const
{
    real sum = 0.0;
    for (size_t i = 0; i < size(); i++)
    {
        sum += _data[i] * _data[i];
    }
    return std::sqrt(sum);
}

real Vector::l1Norm() const
{
    real sum = 0.0;
    for (size_t i = 0; i < size(); i++)
    {
        sum += std::abs(_data[i]);
    }
    return sum;
}

void Vector::addVector(const Vector& vec)
{
    assert(size() == vec.size());
    for (size_t i = 0; i < size(); i++)
    {
        _data[i] += vec._data[i];
    }
}

void Vector::addVector(const Vector& vec, real a)
{
    assert(size() == vec.size());
    for (size_t i = 0; i < size(); i++)
    {
        _data[i] += a * vec._data[i];
    }
}

void Vector::addRow(const Matrix& mat, int64_t i)
{
    assert(i >= 0);
    assert(i < mat.size(0));
    assert(size() == mat.size(1));

    for (size_t j = 0; j < size(); j++)
    {
        _data[j] += mat.at(i, j);
    }
}

void Vector::addRow(const Matrix& mat, int64_t i, real a)
{
    assert(i >= 0);
    assert(i < mat.size(0));
    assert(size() == mat.size(1));

    for (size_t j = 0; j < size(); j++)
    {
        _data[j] += a * mat.at(i,j);
    }
}

int64_t Vector::argmax()
{
    int64_t arg = 0;
    real max = _data[0];
    for (int64_t i = 1; i < size(); i++)
    {
        if (_data[i] > max)
        {
            max = _data[i];
            arg = i;
        }
    }
    return arg;
}

int64_t Vector::argmin()
{
    int64_t arg = 0;
    real min = _data[0];
    for (int64_t i = 0; i < size(); i++)
    {
        if (_data[i] < min)
        {
            min = _data[i];
            arg = i;
        }
    }
    return arg;
}

real Vector::distance(const Vector& vec) const
{
    real d = 0.0;
    assert(vec.size() == size());
    for (size_t i = 0; i < size(); i++)
    {
        d += (vec[i] - _data[i]) * (vec[i] - _data[i]);
    }
    return std::sqrt(d);
}

real Vector::dotRow(const Vector& vec) const
{
    assert(vec.size() == size());
    real d = 0.0;
    for (size_t i = 0; i < size(); i++)
    {
        d += _data[i] * vec[i];
    }
    return d;
}

real Vector::cosine(const Vector& vec) const
{
    assert(vec.size() == size());
    real s = 0.0;
    real d = 0.0;
    real n = 0.0;
    for (size_t i = 0; i < size(); i++)
    {
        d += _data[i] * vec[i];
        n += _data[i] * _data[i];
    }
    n = std::sqrt(n);
    if (n > 0 && vec.l2Norm() > 0)
    {
        s = d / (n * vec.l2Norm());
    }
    return s;
}

real Vector::pearson(const Vector& vec) const
{
    assert(size() == vec.size());
    real avg1 = 0.0, avg2 = 0.0, numr = 0.0, den1 = 0.0, den2 = 0.0;
    for (int32_t i = 0; i < size(); i++)
    {
        avg1 += _data[i];
        avg2 += vec[i];
    }
    avg1 = avg1 / size();
    avg2 = avg2 / size();

    for (int32_t i = 0; i < size(); i++)
    {
        numr += (_data[i] - avg1) * (vec[i] - avg2);
        den1 += (_data[i] - avg1) * (_data[i] - avg1);
        den2 += (vec[i] - avg2) * (vec[i] - avg2);
    }
    real res = numr / (std::sqrt(den1 * den2));
    return res;
}

std::ostream& operator<<(std::ostream& os, const Vector& vec)
{
    os << std::setprecision(5);
    for (size_t i = 0; i < vec.size(); i++)
    {
        os << vec[i] << ' ';
    }
    return os;
}

}
