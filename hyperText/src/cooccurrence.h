#ifndef COOCCURRENCE_H
#define COOCCURRENCE_H

#include <vector>
#include <string>
#include <istream>
#include <ostream>

#include "args.h"
#include "real.h"
#include "dictionary.h"

namespace hypertext
{

struct CoRecId
{
    int32_t word1;
    int32_t word2;
    real weight;
    int id;
};

class Cooccurrence
{
    protected:
    static const uint32_t MAX_LINE_SIZE = 1024;
    std::shared_ptr<Args> _args;
    std::shared_ptr<Dictionary> _dict;
    uint64_t max_product;
    uint64_t overflow_length;
    int distance_weighting;
    
    static int compareCorec(const CoRec&, const CoRec&);
    int32_t compareCorecId(const CoRecId&, const CoRecId&) const;
    void swapEntry(CoRecId *, int, int);

    public:
    explicit Cooccurrence(std::shared_ptr<Args>, std::string);
    void initialize();
    void heapInsert(CoRecId *, CoRecId, int);
    void heapDelete(CoRecId *, int);
    void getCooccurrence();
    void writeTriple(const std::vector<CoRec>&, std::ostream&);
    void mergeFiles(int);
    int mergeWrite(CoRecId, CoRecId *, std::ostream&);
};

}

#endif