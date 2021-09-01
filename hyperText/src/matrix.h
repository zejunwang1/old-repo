#ifndef MATRIX_H
#define MATRIX_H

#include "real.h"
#include <assert.h>

#include <cstdint>
#include <istream>
#include <ostream>
#include <vector>


namespace hypertext
{

class Vector;
class Matrix
{
    protected:
    std::vector<real> _data;
    const uint64_t _m;
    const uint64_t _n;

    public:
    Matrix();
    explicit Matrix(uint64_t, uint64_t);
    Matrix(const Matrix&) = default;
    Matrix& operator=(const Matrix&) = delete;

    inline real* data() 
    {
        return _data.data();
    }

    inline const real* data() const
    {
        return _data.data();
    }

    inline const real& at(uint64_t i, uint64_t j) const
    {
        return _data[i * _n + j];
    }

    inline real& at(uint64_t i, uint64_t j)
    {
        return _data[i * _n + j];
    }

    inline uint64_t size(int dim) const
    {
        assert(dim == 0 || dim == 1);
        if (dim == 0)
        {
            return _m;
        }
        return _n;
    }

    inline uint64_t rows() const
    {
        return _m;
    }

    inline uint64_t cols() const
    {
        return _n;
    }

    void zero();
    void initial(real);
    real dotRow(const Vector&, int64_t) const;
    void addRow(const Vector& vec, int64_t, real);
    void multiplyRow(const Vector&, int64_t, int64_t);
    void divideRow(const Vector&, int64_t, int64_t);
    real l2NormRow(int64_t) const;
    void l2NormRow(Vector&) const;
    real l2NormCol(int64_t) const;
    void l2NormCol(Vector&) const;
    void save(std::ostream&);
    void load(std::istream&);
    void dump(std::ostream&) const;
};

}

#endif
