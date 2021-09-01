#ifndef DICTIONARY_H
#define DICTIONARY_H

#include "real.h"
#include "args.h"
#include "utils.h"

#include <istream>
#include <ostream>
#include <atomic>
#include <vector>
#include <string>
#include <random>
#include <memory>

namespace hypertext
{

struct element
{
    std::string ngram;
    uint64_t count;
    std::vector<uint32_t> subwords;
};

class Dictionary
{
    protected:
    static const uint32_t MAX_VOCAB_SIZE = 100000000;
    static const uint32_t MAX_LINE_SIZE = 1024;

    std::shared_ptr<Args> _args;
    std::vector<real> _pdiscard;
    std::vector<element> _vocab;
    std::vector<int32_t> _hash2int;
    
    uint64_t _ntokens;
    uint32_t _ngrams;
    std::atomic<uint64_t> _tokenCount;

    void reset(std::istream&) const;
    void initWordNgrams();
    void initTableDiscard();
    uint32_t find(const std::string&, uint32_t) const;

    public:
    static const std::string EOS;
    static const std::string BOW;
    static const std::string EOW;
    explicit Dictionary(std::shared_ptr<Args>);
    explicit Dictionary(std::shared_ptr<Args>, std::istream&);
    uint64_t ntokens() const;
    uint32_t ngrams() const;
    uint32_t hash(const std::string&) const;
    bool readWord(std::istream& in, std::string& word) const;
    void readFromFile(std::istream&);
    void readUnigramVocab(std::istream&);
    void readFromMultiLines(std::istream&);
    std::string getNgram(const std::vector<std::string>&, int, int) const;
    void reduceVocab(uint64_t);
    void addVocab(const std::string&);
    int32_t getId(const std::string&) const;
    std::string getWord(int32_t) const;
    uint64_t getCount(int32_t) const;
    uint64_t getCount(const std::string&) const;
    uint32_t getLine(std::istream&, std::vector<std::string>&, 
                     std::minstd_rand&) const;
    uint32_t getLine(std::istream&, std::vector<uint32_t>&,
                     std::minstd_rand&) const;
    uint32_t getLine(std::istream&, std::vector<std::string>&) const;
    uint32_t getLine(std::istream&, std::vector<uint32_t>&) const;
    bool discard(int32_t, real) const;
    std::vector<uint64_t> getCounts() const;
    void computeSubwords(const std::string&, std::vector<uint32_t>&) const;
    void computeSubwords(const std::string&, std::vector<uint32_t>&,
                         std::vector<std::string>&) const;
    std::vector<uint32_t> getSubwords(int32_t) const;
    std::vector<uint32_t> getSubwords(const std::string&) const;
    void getSubwords(const std::string&, std::vector<uint32_t>&,
                     std::vector<std::string>&) const;
    bool checkWord(const std::string&, std::minstd_rand&) const;
    void getPairs();
    void init();
    void loadBinaryVocab(std::istream&);
    void loadNonbinaryVocab(std::istream&);
    void saveBinaryVocab(std::ostream&) const;
    void saveNonbinaryVocab(std::ostream&) const;
    void dump(std::ostream&) const;
};

}

#endif