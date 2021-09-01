#include "lsa.h"

#include <iostream>
#include <fstream>

namespace hypertext
{

Lsa::Lsa(std::shared_ptr<Args> args, std::string vocab_file) : _args(args)
{
    _dict = std::make_shared<Dictionary>(_args);
    std::ifstream ifs(vocab_file);
    if (!ifs.is_open())
    {
        throw std::invalid_argument(
            vocab_file + "cannot be opened for loading!");
    }

    _dict->loadNonbinaryVocab(ifs);
    ifs.close();
    capacity = _dict->ngrams();
    _wsum.resize(capacity, 0);
    _csum.resize(capacity, 0);
}

void Lsa::getPMIMatrix(std::string cooccurrence_file, SMatrixXf& PMI)
{
    std::ifstream ifs(cooccurrence_file);
    if (!ifs.is_open())
    {
        throw std::invalid_argument(
            cooccurrence_file + " cannot be opened for loading!");
    }

    std::cerr << "\rCompute PMI Matrix" << std::endl;

    uint64_t flag = 0;
    real total_sum = 0.0;
    // firstly loading for row col sum
    while (!ifs.eof())
    {
        if (flag % 100000 == 0 && _args->verbose > 0)
        {
            std::cerr << "\rLoad " << flag << " lines" << std::flush;
        }
        flag++;

        CoRec entry;
        ifs.read((char *) &entry, sizeof(CoRec));
        total_sum += entry.weight;

        _wsum[entry.word1 - 1] += entry.weight;
        _csum[entry.word2 - 1] += entry.weight;
    }
    if (_args->verbose > 0)
    {
        std::cerr << "\rLoad " << flag << " lines" << std::endl;
    }
    ifs.close();

    // secondly loading for PMI matrix
    std::ifstream in(cooccurrence_file);
    if (!in.is_open())
    {
        throw std::invalid_argument(
            cooccurrence_file + " cannot be opened for loading!");
    }
    while (!in.eof())
    {
        CoRec entry;
        in.read((char *) &entry, sizeof(CoRec));
        real val = std::log(entry.weight * total_sum / _wsum[entry.word1 - 1] / _csum[entry.word2 - 1]);
        if (val <= 0)    continue;
        PMI.insert(entry.word1 - 1, entry.word2 - 1) = val;
    }
    PMI.makeCompressed();
}

void Lsa::savePMITriples(std::string cooccurrence_file)
{
    SMatrixXf PMI(capacity, capacity);
    getPMIMatrix(cooccurrence_file, PMI);
    std::string filename = _args->output + ".pmi.triple";
    std::ofstream ofs(filename);
    if (!ofs.is_open())
    {
        throw std::invalid_argument(
            filename + " cannot be opened for saving PMI triples!");
    }
    saveSparseMatrix(ofs, PMI);
}

void Lsa::getLsaVectors(std::string cooccurrence_file, Eigen::MatrixXf& U)
{
    // PMI
    SMatrixXf PMI(capacity, capacity);
    getPMIMatrix(cooccurrence_file, PMI);

    // SVD
    if (_args->verbose > 0)
    {
        std::cerr << "Fast SVD for PMI Matrix" << std::endl;
    }
    FastSVD fsvd(PMI, _args->dim);
    
    U = fsvd.matrixU();
    Eigen::VectorXf S = fsvd.singularValues();
    Eigen::MatrixXf V = fsvd.matrixV();

    // lsa word vectors
    if (_args->transpose)   U = V;

    real r;
    if (_args->eig < 0) r = 0;
    else if (_args->eig > 1)    r = 1;
    else    r = _args->eig;
    
    for (size_t i = 0; i < S.size(); i++)
    {
        S(i) = pow(S(i), r);
    }
    for (size_t i = 0; i < U.rows(); i++)
    {
        U.row(i) = U.row(i).array() * S.transpose().array();
        U.row(i) = U.row(i) / U.row(i).norm();
    }
    
    /*
    // save LSA word vectors
    filename = _args->output + ".vec";
    std::ofstream out(filename);
    if (!out.is_open())
    {
        throw std::invalid_argument(
            filename + " cannot be opened for saving LSA vectors!");
    }

    out << U.rows() << " " << U.cols() << std::endl;
    for (size_t i = 0; i < U.rows(); i++)
    {
        std::string word = _dict->getWord(i);
        out << word;
        for (int j = 0; j < U.cols(); j++)
        {
            out << " " << U(i, j);
        }
        out << std::endl;
    }
    out.close();*/
}

void Lsa::saveDenseMatrix(std::ostream& out, const Eigen::MatrixXf& mat)
{
    if (mat.rows() == 0 || mat.cols() == 0) return;
    out << mat.rows() << " " << mat.cols() << std::endl;
    for (size_t i = 0; i < mat.rows(); i++)
    {
        out << mat(i, 0);
        for (size_t j = 1; j < mat.cols(); j++)
        {
            out << " " << mat(i, j);
        }
        out << std::endl;
    }
}

void Lsa::saveSparseMatrix(std::ostream& out, const SMatrixXf& smat)
{
    if (smat.rows() == 0 || smat.cols() == 0)   return;
    out << smat.rows() << " " << smat.cols() << std::endl;
    for (size_t k = 0; k < smat.outerSize(); k++)
    {
        for (SMatrixXf::InnerIterator it(smat, k); it; ++it)
        {
            out << it.row() << " " << it.col() << " " << it.value() << std::endl;
        }
    }
}

}
