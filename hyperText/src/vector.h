#ifndef VECTOR_H
#define VECTOR_H

#include <cstdint>
#include <ostream>
#include <vector>

#include "real.h"

namespace hypertext
{

class Matrix;
class Vector
{
    protected:
    std::vector<real> _data;

    public:
    explicit Vector(uint64_t);
    Vector(const Vector&) = delete;
    Vector& operator=(const Vector&) = delete;

    inline real* data()
    {
        return _data.data();
    }

    inline const real* data() const
    {
        return _data.data();
    }

    inline const real& operator[](uint64_t i) const
    {
        return _data[i];
    }

    inline real& operator[](uint64_t i)
    {
        return _data[i];
    }

    inline uint64_t size() const
    {
        return _data.size();
    }

    void zero();
    void mul(real);
    void mul(const Matrix&, const Vector&);
    real l2Norm() const;
    real l1Norm() const;
    void addVector(const Vector&);
    void addVector(const Vector&, real);
    void addRow(const Matrix&, int64_t);
    void addRow(const Matrix&, int64_t, real);
    real distance(const Vector&) const;
    real cosine(const Vector&) const;
    real dotRow(const Vector&) const;
    real pearson(const Vector&) const;
    int64_t argmax();
    int64_t argmin();
};

std::ostream& operator<<(std::ostream&, const Vector&);

}

#endif
