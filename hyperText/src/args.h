#ifndef ARGS_H
#define ARGS_H

#include <istream>
#include <ostream>
#include <string>
#include <vector>

namespace hypertext
{

enum class model_name : int { cbow = 1, sg, pmi, lsa };
enum class loss_name : int { hs = 1, ns };

class Args
{
    protected:
    std::string lossToString(loss_name) const;
    std::string boolToString(bool) const;
    std::string modelToString(model_name) const;

    public:
    Args();
    
    std::string input;
    std::string output;

    int ws;
    int dim;
    int verbose;
    int minCount;
    int normalize;
    model_name model;

    std::string eval;
    std::string pretrainedVectors;

    double lr;
    int lrUpdateRate;
    int wng;
    int cng;
    int epoch;
    int neg;
    loss_name loss;
    int bucket;
    int minn;
    int maxn;
    int thread;
    double t;
    bool saveOutput;

    int memory;
    int disWeight;
    uint64_t maxProduct;
    uint64_t overflow;
    uint64_t chunkSize;

    double eig;
    bool transpose;

    void printHelp();
    void printBasicHelp();
    void printDictionaryHelp();
    void printTrainingHelp();

    void parseArgs(const std::vector<std::string>&);

    void load(std::istream&);
    void save(std::ostream&) const;
    void dump(std::ostream&) const;
};

}

#endif