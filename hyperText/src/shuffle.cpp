#include "shuffle.h"

#include <stdio.h>

#include <iostream>
#include <fstream>
#include <stdexcept>

namespace hypertext
{

Shuffle::Shuffle(std::shared_ptr<Args> args) : _args(args)
{
    initialize();
}

void Shuffle::initialize()
{
    // initialize shuffle chunk size
    if (_args->chunkSize == 0)
    {
        chunk_size = (uint64_t)(0.95 * (real)(_args->memory) * 1073741824 / (sizeof(CoRec)));
    }
    else
    {
        chunk_size = _args->chunkSize;
    }
}

void Shuffle::shuffleByChunks(std::string cooccurrence_file)
{
    std::ifstream ifs(cooccurrence_file);
    if (!ifs.is_open())
    {
        throw std::invalid_argument(
            cooccurrence_file + " cannot be opened for shuffling!");
    }

    // triple vector
    std::vector<CoRec> cr;
    
    // triple chunk file counter
    int fidcounter = 0;

    if (_args->verbose > 0)
    {
        std::cerr << "Shuffling Cooccurrences" << std::endl;
    }

    uint64_t flag = 0;
    std::string filename;
    while (!ifs.eof())
    {
        if (flag >= chunk_size)
        {
            // shuffle chunk triples
            shuffle(cr, fidcounter);
            filename = "shuffle_000" + std::to_string(fidcounter) + ".bin";
            std::ofstream ofs(filename);
            if (!ofs.is_open())
            {
                throw std::invalid_argument(
                    filename + " cannot be opened for saving shuffled cooccurrences!");
            }
            writeTriple(cr, ofs);
            ofs.close();
            fidcounter++;
            cr.clear();

            if (_args->verbose > 0)
            {
                std::cerr << "\rProcessed " << flag << " lines" << std::flush;
            }
        }
        CoRec entry;
        ifs.read((char *) &entry, sizeof(CoRec));
        cr.push_back(entry);
        flag++;
    }
    if (_args->verbose > 0)
    {
        std::cerr << "\rProcessed " << flag << " lines" << std::endl;
    }
    // shuffle last chunk
    shuffle(cr, fidcounter);
    filename = "shuffle_000" + std::to_string(fidcounter) + ".bin";
    std::ofstream ofs(filename);
    if (!ofs.is_open())
    {
        throw std::invalid_argument(
            filename + " cannot be opened for saving shuffled cooccurrences!");
    }
    writeTriple(cr, ofs);
    ofs.close();

    // merge shuffled files
    mergeShuffle(fidcounter + 1);
}

void Shuffle::shuffle(std::vector<CoRec>& cr, int fidcounter)
{
    std::minstd_rand rng(fidcounter);
    uint64_t i, j;
    uint64_t n = cr.size();
    CoRec temp;
    std::uniform_int_distribution<> uniform(0, n);
    for (i = n - 2; i > 0; i--)
    {
        j = uniform(rng);
        j = j % i;
        temp = cr[i];
        cr[i] = cr[j];
        cr[j] = temp;
    }
}

void Shuffle::writeTriple(const std::vector<CoRec>& cr, std::ostream& os)
{
    for (uint64_t i = 0; i < cr.size(); i++)
    {
        os.write((char *) &cr[i], sizeof(CoRec));
    }
}

void Shuffle::mergeShuffle(int num)
{
    uint64_t flag = 0;
    std::vector<CoRec> cr;
    std::string filename;
    std::ifstream *ifs;
    ifs = new std::ifstream[num];

    filename = _args->output + ".shuffle";
    std::ofstream ofs(filename);
    if (!ofs.is_open())
    {
        throw std::invalid_argument(
            filename + " cannot be opened for shuffled cooccurrences!");
    }
    // open shuffled files
    for (int i = 0; i < num; i++)
    {
        filename = "shuffle_000" + std::to_string(i) + ".bin";
        ifs[i].open(filename);
        if (!(ifs[i].is_open()))
        {
            throw std::invalid_argument(
                filename + " cannot be opened for merging shuffled cooccurrences!");
        }
    }
    
    while (1)
    {
        for (int i = 0; i < num; i++)
        {
            if (!(ifs[i].eof()))
            {
                for (uint64_t k = 0; k < chunk_size / num; k++)
                {
                    CoRec entry;
                    ifs[i].read((char *) &entry, sizeof(CoRec));
                    cr.push_back(entry);
                    if (ifs[i].eof())
                    {
                        break;
                    }
                }
            }
        }
        if (cr.size() == 0)
        {
            break;
        }
        
        // shuffle between temp files
        shuffle(cr, cr.size() % num);
        writeTriple(cr, ofs);
        flag += cr.size();
        if (_args->verbose > 0)
        {
            std::cerr << "\rMerge " << flag << " lines" << std::flush;
        }
        cr.clear();
    }
    if (_args->verbose > 0)
    {
        std::cerr << "\rMerge " << flag << " lines" << std::endl;
    }
    ofs.close();

    // delete temporary files
    for (int i = 0; i < num; i++)
    {
        ifs[i].close();
        filename = "shuffle_000" + std::to_string(i) + ".bin";
        remove(filename.c_str());
    }
}

}