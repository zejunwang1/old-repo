#include "cooccurrence.h"

#include <stdio.h>

#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace hypertext
{

Cooccurrence::Cooccurrence(std::shared_ptr<Args> args, std::string vocab_file) : _args(args)
{
    // load vocabulary
    _dict = std::make_shared<Dictionary>(_args);
    std::ifstream ifs(vocab_file);
    if (!ifs.is_open())
    {
        throw std::invalid_argument(
            vocab_file + " cannot be opened for loading!");
    }
    _dict->loadNonbinaryVocab(ifs);
    ifs.close();

    initialize();
}

void Cooccurrence::initialize()
{
    // initialize max product and overflow length
    real rlimit = 0.85 * (real)(_args->memory) * 1073741824 / sizeof(CoRec);
    if (_args->overflow == 0)
    {
        overflow_length = (uint64_t)(rlimit / 6);
    }
    else
    {
        overflow_length = _args->overflow;
    }

    if (_args->maxProduct == 0)
    {
        max_product = 15000000;
    }
    else
    {
        max_product = _args->maxProduct;
    }
    distance_weighting = _args->disWeight;
}

int Cooccurrence::compareCorec(const CoRec& a, const CoRec& b)
{
    if (a.word1 > b.word1)
    {
        return 0;
    }
    else if (a.word1 < b.word1)
    {
        return 1;
    }
    else
    {
        if (a.word2 > b.word2)
        {
            return 0;
        }
        else
        {
            return 1;
        }
    }
}

int32_t Cooccurrence::compareCorecId(const CoRecId& a, const CoRecId& b) const
{
    int32_t c = a.word1 - b.word1;
    if (c != 0) return c;
    else
    {
        return a.word2 - b.word2;
    }
}

void Cooccurrence::swapEntry(CoRecId* pq, int i, int j)
{
    // swap two entries of minimal heap
    CoRecId temp = pq[i];
    pq[i] = pq[j];
    pq[j] = temp;
}

void Cooccurrence::heapInsert(CoRecId *pq, CoRecId entry, int size)
{
    // insert entry into minimal heap
    int i = size - 1;
    int p = (i - 1) / 2;
    pq[i] = entry;
    while (p >= 0)
    {
        if (compareCorecId(pq[p], pq[i]) > 0)
        {
            swapEntry(pq, p, i);
            i = p;
            p = (i - 1) / 2;
        }
        else
        {
            break;
        }
    }
}

void Cooccurrence::heapDelete(CoRecId *pq, int size)
{
    // delete entry from minimal heap
    int i, p = 0;
    pq[p] = pq[size - 1];
    i = 2 * p + 1;
    while (i < (size - 1))
    {
        if (i == size -2)
        {
            if (compareCorecId(pq[p], pq[i]) > 0)
            {
                swapEntry(pq, p, i);
            }
            return;
        }
        else
        {
            if (compareCorecId(pq[i], pq[i+1]) < 0)
            {
                if (compareCorecId(pq[p], pq[i]) > 0)
                {
                    swapEntry(pq, p, i);
                    p = i;
                    i = 2 * p + 1;
                }
                else
                {
                    return;
                }
            }
            else
            {
                if (compareCorecId(pq[p], pq[i+1]) > 0)
                {
                    swapEntry(pq, p, i + 1);
                    p = i + 1;
                    i = 2 * p + 1;
                }
                else
                {
                    return;
                }
            }
        }
    }
}

void Cooccurrence::getCooccurrence()
{
    if (_args->verbose > 0)
    {
        std::cerr << "\rCounting Cooccurrences" << std::endl;
    }
    
    // build auxiliary lookup table used to bigram_table
    std::vector<uint32_t> lookup;
    lookup.resize(_dict->ngrams() + 1, 0);
    lookup[0] = 1;
    for (size_t i = 1; i <= _dict->ngrams(); i++)
    {
        if ((lookup[i] = max_product / i) < _dict->ngrams())
        {
            lookup[i] += lookup[i-1];
        }
        else
        {
            lookup[i] = lookup[i - 1] + _dict->ngrams();
        }
    }

    // top-left corner dense count matrix
    std::vector<real> bigram_table;
    bigram_table.resize(lookup[_dict->ngrams()], 0.0);

    // sparse matrix file counter
    int fidcounter = 1;

    // triple vector
    std::vector<CoRec> cr;

    // traverse input text file
    if (_args->input == "-")
    {
        throw std::invalid_argument("Cannot use stdin for cooccurrences!");
    }
    std::ifstream ifs(_args->input);
    if (!ifs.is_open())
    {
        throw std::invalid_argument(
            _args->input + " cannot be opened for cooccurrences!");
    }
    std::minstd_rand rng(0);
    uint64_t flag = 0;
    std::vector<std::string> line;
    uint64_t tokenCount = 0;
    const uint64_t ntokens = _dict->ntokens();
    std::string filename;
    while (tokenCount < ntokens)
    {
        tokenCount += _dict->getLine(ifs, line);
        for (int i = 0; i < line.size(); i++)
        {
            for (int j = 1; j <= _args->wng; j++)
            {
                std::string word = _dict->getNgram(line, i, j);
                if (word.empty())   continue;

                // subsampling
                //if (!(_dict->checkWord(word, rng)))  continue;
                int32_t w1 = _dict->getId(word) + 1;

                int boundary = _args->ws;
                for (int c = -boundary; c <= boundary; c++)
                {
                    if (c != 0 && i + c >= 0 && i + c < line.size())
                    {
                        for (int k = 1; k <= _args->cng; k++)
                        {
                            if ((c < 0 && c + k - 1 >= 0) || (c > 0 && c <= j - 1))
                            {
                                // exist overlap between word ngram and context ngram
                                continue;
                            }
                            std::string context = _dict->getNgram(line, i + c, k);
                            if (context.empty())    continue;

                            // subsampling
                            //if (!(_dict->checkWord(context, rng)))  continue;
                            int32_t w2 = _dict->getId(context) + 1;
                            
                            if (w1 * w2 < max_product)
                            {
                                // build top-left cooccurrence matrix
                                bigram_table[lookup[w2 - 1] + w1 - 2] += distance_weighting ? 1.0/((real)(std::abs(c))) : 1.0;
                            }
                            else
                            {
                                // save temporary triple file
                                if (cr.size() >= overflow_length)
                                {
                                    sort(cr.begin(), cr.end(), compareCorec);
                                    filename = "cooccurrence_000" + std::to_string(fidcounter) + ".bin";
                                    std::ofstream ofs(filename);
                                    if (!ofs.is_open())
                                    {
                                        throw std::invalid_argument(
                                            filename + " cannot be opened for cooccurrences!");
                                    }
                                    writeTriple(cr, ofs);
                                    ofs.close();
                                    fidcounter++;
                                    cr.clear();
                                }
                                // build triple sparse matrix
                                CoRec entry;
                                entry.word1 = w1;
                                entry.word2 = w2;
                                entry.weight = distance_weighting ? 1.0/((real)(std::abs(c))) : 1.0;
                                cr.push_back(entry);
                            }
                        }
                    }
                }
            }
        }
        line.clear();
        // print log info
        if (_args->verbose > 0)
        {
            flag++;
            if (flag % 100 == 0)
            {
                std::cerr << "\rProcessed " << tokenCount << " tokens" << std::flush;
            }
        }
    }
    if (_args->verbose > 0)
    {
        std::cerr << "\rProcessed " << tokenCount << " tokens" << std::endl;
    }

    sort(cr.begin(), cr.end(), compareCorec);
    filename = "cooccurrence_000" + std::to_string(fidcounter) + ".bin";
    std::ofstream ofs(filename);
    if (!ofs.is_open())
    {
        throw std::invalid_argument(
            filename + " cannot be opened for cooccurrences!");
    }
    writeTriple(cr, ofs);
    ofs.close();

    // write dense bigram_table, skipping zeros
    filename = "cooccurrence_0000.bin";
    std::ofstream out(filename);
    if (!out.is_open())
    {
        throw std::invalid_argument(
            filename + " cannot be opened for cooccurrences!");
    }
    for (int32_t x = 1; x <= _dict->ngrams(); x++)
    {
        for (int32_t y = 1; y <= (lookup[x] - lookup[x - 1]); y++)
        {
            real weight = bigram_table[lookup[x - 1] + y -2];
            if (weight != 0)
            {
                out.write((char *) &x, sizeof(int32_t));
                out.write((char *) &y, sizeof(int32_t));
                out.write((char *) &weight, sizeof(real));
            }
        }
    }
    out.close();

    // merge cooccurrence files
    mergeFiles(fidcounter + 1);
}

void Cooccurrence::writeTriple(const std::vector<CoRec>& cr, std::ostream& os)
{
    if (cr.size() == 0) return;
    uint64_t a = 0;
    CoRec old = cr[a];
    for (a = 1; a < cr.size(); a++)
    {
        if (cr[a].word1 == old.word1 && cr[a].word2 == old.word2)
        {
            old.weight += cr[a].weight;
            continue;
        }
        os.write((char *) &old, sizeof(CoRec));
        old = cr[a];
    }
    os.write((char *) &old, sizeof(CoRec));
}

void Cooccurrence::mergeFiles(int num)
{
    // merge cooccurrence files
    std::string filename;
    CoRecId *pq, prev, curr;
    std::ifstream *ifs;
    ifs = new std::ifstream[num];
    pq = new CoRecId[num];

    // build minimal heap
    for (int i = 0; i < num; i++)
    {
        filename = "cooccurrence_000" + std::to_string(i) + ".bin";
        ifs[i].open(filename);
        if (!(ifs[i].is_open()))
        {
            throw std::invalid_argument(filename + " cannot be opened for merging!");
        }
        ifs[i].read((char *) &curr, sizeof(CoRec));
        curr.id = i;
        heapInsert(pq, curr, i + 1);
    }

    int size = num;
    prev = pq[0];
    int i = pq[0].id;
    heapDelete(pq, size);
    ifs[i].read((char *) &curr, sizeof(CoRec));
    if (ifs[i].eof())   size--;
    else
    {
        curr.id = i;
        heapInsert(pq, curr, size);
    }

    uint64_t flag = 0;
    std::string ofn = _args->output + ".cooccurrence";
    std::ofstream out(ofn);
    if (!out.is_open())
    {
        throw std::invalid_argument(ofn + " cannot be opened for cooccurrence!");
    }
    // maintain minimal heap
    while (size > 0)
    {
        flag += mergeWrite(pq[0], &prev, out);
        if (flag % 100000 == 0)
        {
            if (_args->verbose > 0)
            {
                std::cerr << "\rMerge " << flag << " lines" << std::flush;
            }
        }
        i = pq[0].id;
        heapDelete(pq, size);
        ifs[i].read((char *) &curr, sizeof(CoRec));
        if (ifs[i].eof())   size--;
        else
        {
            curr.id = i;
            heapInsert(pq, curr, size);
        }
    }
    out.write((char *) &prev, sizeof(CoRec));
    flag++;
    if (_args->verbose > 0)
    {
        std::cerr << "\rMerge " << flag << " lines" << std::endl;
    }
    out.close();
    
    // delete temporary files
    for (i = 0; i < num; i++)
    {
        filename = "cooccurrence_000" + std::to_string(i) + ".bin";
        remove(filename.c_str());
    }
}

int Cooccurrence::mergeWrite(CoRecId curr, CoRecId *prev, std::ostream& out)
{
    if (curr.word1 == prev->word1 && curr.word2 == prev->word2)
    {
        prev->weight += curr.weight;
        return 0;
    }
    out.write((char *) prev, sizeof(CoRec));
    *prev = curr;
    return 1;
}

}
