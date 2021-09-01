#ifndef SHUFFLE_H
#define SHUFFLE_H

#include <istream>
#include <ostream>
#include <vector>
#include <string>
#include <random>
#include <memory>

#include "args.h"
#include "real.h"

namespace hypertext
{

class Shuffle
{
    protected:
    std::shared_ptr<Args> _args;
    uint64_t chunk_size;
    
    public:
    explicit Shuffle(std::shared_ptr<Args>);
    void initialize();
    void shuffleByChunks(std::string);
    void shuffle(std::vector<CoRec>&, int);
    void writeTriple(const std::vector<CoRec>&, std::ostream&);
    void mergeShuffle(int);
};

}

#endif