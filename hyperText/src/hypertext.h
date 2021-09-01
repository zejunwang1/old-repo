#ifndef HYPERTEXT_H
#define HYPERTEXT_H

#include <atomic>
#include <memory>
#include <chrono>
#include <iostream>
#include <set>
#include <queue>
#include <unordered_map>

#include <time.h>

#include "args.h"
#include "dictionary.h"
#include "cooccurrence.h"
#include "matrix.h"
#include "vector.h"
#include "model.h"
#include "lsa.h"
#include "real.h"
#include "utils.h"

namespace hypertext
{

class HyperText
{
    protected:
    std::shared_ptr<Args> _args;
    std::shared_ptr<Dictionary> _dict;

    std::shared_ptr<Matrix> _input;
    std::shared_ptr<Matrix> _output;

    std::shared_ptr<Model> _model;

    std::atomic<uint64_t> _tokenCount;
    std::atomic<real> _loss;

    std::chrono::steady_clock::time_point _start;

    uint32_t version;   // hypertext version

    void startThreads();
    void signModel(std::ostream&) const;
    bool checkModel(std::istream&);

    public:
    HyperText();
    const Args getArgs() const;
    std::shared_ptr<const Dictionary> getDictionary() const;
    std::shared_ptr<const Matrix> getInputMatrix() const;
    std::shared_ptr<const Matrix> getOutputMatrix() const;
    int32_t getDimension() const;
    int32_t getWordId(const std::string&) const;
    int32_t getSubwordId(const std::string&) const;
    
    void train(const Args&);
    void LsaModel();
    void trainWord2Vec();
    void trainWord2VecThread(uint32_t);
    void cbow(Model&, real, const std::vector<std::string>&);
    void skipgram(Model&, real, const std::vector<std::string>&);
    void saveModel();
    void saveOutput();
    void loadModel();
    void saveModel(const std::string);
    void printInfo(real, real, std::ostream&);
    void loadVectors(const std::string&);
    void saveVectors();
    void getVector(Vector&, const std::string&) const;
    void getLsaVector(Vector&, const std::string&) const;
    void getPMIVector(Vector&, const std::string&, const SMatrixXf&) const;
    void getWordVector(Vector&, const std::string&) const;
    void getSubwordVector(Vector&, const std::string&) const;
    void addInputVector(Vector&, uint32_t) const;
    void getInputVector(Vector&, uint32_t) const;
    void printNgramVectors(const std::string&);
    void precomputeWordVectors(Matrix&);
    void analogies(int32_t);
    void findNN(const Matrix&, const Vector&,
                int32_t, const std::set<std::string>&,
                std::vector<std::pair<real, std::string>>&);
    real distance(const std::string&, const std::string&) const;
    real cosine(const std::string&, const std::string&) const;
    void evaluation(const Args&);
    void correlationEval();
    real pearson(std::vector<std::pair<std::string, real>>,
                 std::vector<std::pair<std::string, real>>);
    real spearman(std::vector<std::pair<std::string, real>>,
                  std::vector<std::pair<std::string, real>>);
    void topKcosine(const std::string&, int32_t, std::vector<std::string>&);
    void loadModel(const std::string&);
    void loadModel(std::istream&);
};

}

#endif