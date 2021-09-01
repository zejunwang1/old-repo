#include "dictionary.h"

#include <assert.h>

#include <iostream>
#include <fstream>
#include <algorithm>
#include <iterator>
#include <cmath>
#include <stdexcept>

namespace hypertext
{

const std::string Dictionary::EOS = "</s>";
const std::string Dictionary::BOW = "<";
const std::string Dictionary::EOW = ">";

Dictionary::Dictionary(std::shared_ptr<Args> args) : _args(args), _ntokens(0), _ngrams(0), 
             _hash2int(MAX_VOCAB_SIZE, -1) {}
Dictionary::Dictionary(std::shared_ptr<Args> args, std::istream& in) : _args(args), 
             _ntokens(0), _ngrams(0), _hash2int(MAX_VOCAB_SIZE, -1)
{
    loadBinaryVocab(in);
}

uint64_t Dictionary::ntokens() const
{
    return _ntokens;
}

uint32_t Dictionary::ngrams() const
{
    return _ngrams;
}

bool Dictionary::readWord(std::istream& in, std::string& word) const
{
    int c;
    std::streambuf& buf = *in.rdbuf();
    word.clear();

    while ((c = buf.sbumpc()) != EOF)
    {
        if (c == ' ' || c == '\n' || c == '\r' || c == '\t' || c == '\v' || 
            c == '\f' || c == '\0')
        {
            if (word.empty())
            {
                if (c == '\n')
                {
                    word += EOS;
                    return true;
                }
                continue;
            }
            else
            {
                if (c == '\n')
                {
                    buf.sungetc();
                }
                return true;
            }
        }
        word.push_back(c);
    }
    in.get();
    return !word.empty();
}

void Dictionary::reset(std::istream& in) const
{
    if (in.eof())
    {
        in.clear();
        in.seekg(std::streampos(0));
    }
}

void Dictionary::readUnigramVocab(std::istream& in)
{
    std::string word;
    uint64_t minThreshold = 1;
    while (readWord(in, word))
    {
        _ntokens++;
        if (_ntokens % 100000 == 0 && _args->verbose > 0)
        {
            std::cerr << "\rRead " << _ntokens << " tokens" << std::flush;
        }

        // add unigram word to dictionary
        addVocab(word);
        if (_ngrams > 0.75 * MAX_VOCAB_SIZE)
        {
            minThreshold++;
            reduceVocab(minThreshold);
        }
    }
    reduceVocab(_args->minCount);
    initTableDiscard();
    initWordNgrams();

    if (_args->verbose > 0)
    {
        std::cerr << "\rRead " << _ntokens << " tokens" << std::endl;
        std::cerr << "Number of words: " << _ngrams << std::endl;
    }

    if (_ngrams == 0)
    {
        throw std::invalid_argument("Empty vocabulary. Try a smaller minCount value.");
    }
}

void Dictionary::readFromFile(std::istream& in)
{
    std::string word;
    std::vector<std::string> words;
    int ng = std::max(_args->wng, _args->cng);
    uint32_t nwords = 0;
    uint64_t minThreshold = 1;

    reset(in);
    word.clear();
    while (readWord(in, word))
    {
        nwords++;
        _ntokens++;
        words.push_back(word);

        if (_ntokens % 100000 == 0 && _args->verbose > 0)
        {
            std::cerr << "\rRead " << _ntokens << " tokens" << std::flush;
        }

        if (nwords > MAX_LINE_SIZE || word == EOS)
        {
            for (int i = 0; i < words.size(); i++)
            {
                for (int j = 1; j <= ng; j++)
                {
                    // get ngram 
                    std::string ngram;
                    ngram = getNgram(words, i, j);
                    if (ngram.empty())
                    {
                        break;
                    }
                    // add ngram to dictionary
                    addVocab(ngram);
                    if (_ngrams > 0.75 * MAX_VOCAB_SIZE)
                    {
                        minThreshold++;
                        reduceVocab(minThreshold);
                    }
                }
            }
            nwords = 0;
            words.clear();
        }
    }
    reduceVocab(_args->minCount);
    initTableDiscard();
    initWordNgrams();

    if (_args->verbose > 0)
    {
        std::cerr << "\rRead " << _ntokens << " tokens" << std::endl;
        std::cerr << "Number of words: " << _ngrams << std::endl;
    }

    if (_ngrams == 0)
    {
        throw std::invalid_argument("Empty vocabulary. Try a smaller minCount value.");
    }
}

void Dictionary::readFromMultiLines(std::istream& in)
{
    std::string line;
    uint64_t minThreshold = 1;
    int ng = std::max(_args->wng, _args->cng);
    while (std::getline(in, line))
    {
        std::vector<std::string> words;
        std::vector<std::string> lines;
        utils::splitString(line, " ", words);
        // delete empty string
        for (int i = 0; i < words.size(); i++)
        {
            if (words[i].empty())   continue;
            lines.push_back(words[i]);
        }

        for (int i = 0; i < lines.size(); i++)
        {
            _ntokens++;
            if (_ntokens % 100000 == 0 && _args->verbose > 0)
            {
                std::cerr << "\rRead " << _ntokens << " tokens" << std::flush;
            }

            for (int j = 1; j <= ng; j++)
            {
                // get ngram
                std::string ngram;
                ngram = getNgram(lines, i, j);
                if (ngram.empty())  break;
                
                // add ngram to dictionary
                addVocab(ngram);
                if (_ngrams > 0.75 * MAX_VOCAB_SIZE)
                {
                    minThreshold++;
                    reduceVocab(minThreshold);
                }
            }
        } 
    }
    reduceVocab(_args->minCount);
    initTableDiscard();
    initWordNgrams();

    if (_args->verbose > 0)
    {
        std::cerr << "\rRead " << _ntokens << " tokens" << std::endl;
        std::cerr << "Number of words: " << _ngrams << std::endl;
    }

    if (_ngrams == 0)
    {
        throw std::invalid_argument("Empty vocabulary. Try a smaller minCount value.");
    }
}

std::string Dictionary::getNgram(const std::vector<std::string>& words, int pos, int n) const
{
    std::string res;
    res = words[pos];
    for (int i = 1; i < n; i++)
    {
        if ((pos + i) >= words.size())
        {
            return "";
        }
        else
        {
            res = res + "@$" + words[pos + i];
        }
    }
    return res;
}

void Dictionary::reduceVocab(uint64_t t)
{
    // sort vocab by word count
    sort(_vocab.begin(), _vocab.end(), [](const element& e1, const element& e2)
         {return e1.count > e2.count;});

    // remove low frequency words
    _vocab.erase(remove_if(_vocab.begin(), _vocab.end(), [&](const element& e)
                 {return e.count < t;}), _vocab.end());
    _vocab.shrink_to_fit();

    // update hash table
    std::fill(_hash2int.begin(), _hash2int.end(), -1);
    _ngrams = 0;

    for (auto it = _vocab.begin(); it != _vocab.end(); it++)
    {
        uint32_t id = find(it->ngram, hash(it->ngram));
        _hash2int[id] = _ngrams;
        _ngrams++;
    }
}

void Dictionary::addVocab(const std::string& word)
{
    uint32_t id = find(word, hash(word));
    if (_hash2int[id] == -1)
    {
        element e;
        e.ngram = word;
        e.count = 1;
        _vocab.push_back(e);
        _hash2int[id] = _ngrams;
        _ngrams++;
    }
    else
    {
        _vocab[_hash2int[id]].count++;
    }
}

uint32_t Dictionary::hash(const std::string& word) const
{
    uint32_t h = 2166136261;
    for (size_t i = 0; i < word.size(); i++)
    {
        h = h ^ uint32_t(word[i]);
        h = h * 16777619;
    }
    return h;
}

uint32_t Dictionary::find(const std::string& word, uint32_t h) const
{
    uint32_t hash2intsize = _hash2int.size();
    uint32_t id = h % hash2intsize;
    while (_hash2int[id] != -1 && _vocab[_hash2int[id]].ngram != word)
    {
        id = (id + 1) % hash2intsize;
    }
    return id;
}

void Dictionary::initTableDiscard()
{
    _pdiscard.resize(_ngrams);
    for (size_t i = 0; i < _ngrams; i++)
    {
        real f = real(_vocab[i].count) / real(_ntokens);
        _pdiscard[i] = std::sqrt(_args->t / f) + _args->t / f;
    }
}

int32_t Dictionary::getId(const std::string& word) const
{
    uint32_t id = find(word, hash(word));
    return _hash2int[id];
}

std::string Dictionary::getWord(int32_t id) const
{
    assert(id >= 0);
    assert(id < _ngrams);

    return _vocab[id].ngram;
}

uint32_t Dictionary::getLine(std::istream& in, std::vector<std::string>& words,
                        std::minstd_rand& rng) const
{
    std::uniform_real_distribution<> uniform(0, 1);
    std::string word;
    uint32_t nwords = 0;

    reset(in);
    words.clear();
    while (readWord(in, word))
    {
        nwords++;
        uint32_t h = find(word, hash(word));
        int32_t id = _hash2int[h];
        if (id < 0)  continue;

        // subsampling
        if (!discard(id, uniform(rng)))
        {
            words.push_back(word);
        }
        
        if (nwords > MAX_LINE_SIZE || word == EOS)
        {
            break;
        }
    }
    return nwords;
}

uint32_t Dictionary::getLine(std::istream& in, std::vector<std::string>& words) const
{
    std::string word;
    uint32_t nwords = 0;
    
    reset(in);
    words.clear();

    while (readWord(in, word))
    {
        nwords++;
        uint32_t h = find(word, hash(word));
        int32_t id = _hash2int[h];
        if (id < 0) continue;
        words.push_back(word);
        
        if (nwords >= MAX_LINE_SIZE || word == EOS)
        {
            break;
        }
    }
    return nwords;
}

uint32_t Dictionary::getLine(std::istream& in, std::vector<uint32_t>& words) const
{
    std::string word;
    uint32_t nwords = 0;

    reset(in);
    words.clear();
    while (readWord(in, word))
    {
        nwords++;
        uint32_t h = find(word, hash(word));
        int32_t id = _hash2int[h];
        if (id < 0) continue;
        words.push_back(id);

        if (nwords > MAX_LINE_SIZE || word == EOS)
        {
            break;
        }
    }
    return nwords;
}

uint32_t Dictionary::getLine(std::istream& in, std::vector<uint32_t>& words,
                        std::minstd_rand& rng) const
{
    std::uniform_real_distribution<> uniform(0, 1);
    std::string word;
    uint32_t nwords = 0;
    
    reset(in);
    words.clear();
    while (readWord(in, word))
    {
        nwords++;
        uint32_t h = find(word, hash(word));
        int32_t id = _hash2int[h];
        if (id < 0) continue;

        if (!discard(id, uniform(rng)))
        {
            words.push_back(id);
        }

        if (nwords > MAX_LINE_SIZE || word == EOS)
        {
            break;
        }
    }
    return nwords;
}

bool Dictionary::discard(int32_t id, real r) const
{
    assert(id >= 0);
    assert(id < _ngrams);

    return r > _pdiscard[id];
}

uint64_t Dictionary::getCount(int32_t i) const
{
    assert(i >= 0);
    assert(i < _ngrams);
    return _vocab[i].count;
}

uint64_t Dictionary::getCount(const std::string& word) const
{
    int32_t i = getId(word);
    return getCount(i);
}

std::vector<uint64_t> Dictionary::getCounts() const
{
    // prepare for huffman tree
    std::vector<uint64_t> counts;
    for (auto& w : _vocab)
    {
        counts.push_back(w.count);
    }
    return counts;
}

/*
void Dictionary::initWordNgrams()
{
    // ignore bigram / trigram subwords information
    for (size_t i = 0; i < _ngrams; i++)
    {
        std::string word = BOW + _vocab[i].ngram + EOW;
        _vocab[i].subwords.clear();
        _vocab[i].subwords.push_back(i);

        std::string::size_type pos;
        pos = _vocab[i].ngram.find("@$");
        if (_vocab[i].ngram != EOS && pos == _vocab[i].ngram.npos)
        {
            computeSubwords(word, _vocab[i].subwords);
        }
    }
}*/

void Dictionary::initWordNgrams()
{
    // consider bigram / trigram subwords information
    for (size_t i = 0; i < _ngrams; i++)
    {
        std::string::size_type pos;
        pos = _vocab[i].ngram.find("@$");
        _vocab[i].subwords.clear();
        _vocab[i].subwords.push_back(i);
        if (pos == _vocab[i].ngram.npos)
        {
            std::string word = BOW + _vocab[i].ngram + EOW;
            if (_vocab[i].ngram != EOS)
            {
                computeSubwords(word, _vocab[i].subwords);
            }
        }
        else
        {
            std::vector<std::string> substrings;
            utils::splitString(_vocab[i].ngram, "@$", substrings);
            for (int k = 0; k < substrings.size(); k++)
            {
                if (substrings[k].empty())  continue;
                int32_t id = getId(substrings[k]);
                if (id < 0) continue;
                _vocab[i].subwords.push_back(id);
                std::string word = BOW + substrings[k] + EOW;
                if (substrings[k] != EOS)
                {
                    computeSubwords(word, _vocab[i].subwords);
                }
            }
        }
    }
}

void Dictionary::computeSubwords(const std::string& word, std::vector<uint32_t>& ngrams) const
{
    for (size_t i = 0; i < word.size(); i++)
    {
        std::string ngram;
        if ((word[i] & 0xC0) == 0x80)   continue;
        for (size_t j = i, n = 1; j < word.size() && n <= _args->maxn; n++)
        {
            ngram.push_back(word[j]);
            j++;
            while (j < word.size() && (word[j] & 0xC0) == 0x80)
            {
                ngram.push_back(word[j++]);
            }
            if (n >= _args->minn && !(n == 1 && (i == 0 || j == word.size())))
            {
                // hash to buckets
                uint32_t h = hash(ngram) % _args->bucket;
                ngrams.push_back(h + _ngrams);
            }
        }
    }
}

void Dictionary::computeSubwords(const std::string& word, std::vector<uint32_t>& ngrams,
                            std::vector<std::string>& substrings) const
{
    for (size_t i = 0; i < word.size(); i++)
    {
        std::string ngram;
        if ((word[i] & 0xC0) == 0x80)   continue;
        for (size_t j = i, n = 1; j < word.size() && n <= _args->maxn; n++)
        {
            ngram.push_back(word[j]);
            j++;
            while (j < word.size() && (word[j] & 0xC0) == 0x80)
            {
                ngram.push_back(word[j++]);
            }
            if (n >= _args->minn && !(n == 1 && (i == 0 || j == word.size())))
            {
                // hash to buckets
                uint32_t h = hash(ngram) % _args->bucket;
                ngrams.push_back(h + _ngrams);
                substrings.push_back(ngram);
            }
        }
    }
}

std::vector<uint32_t> Dictionary::getSubwords(int32_t id) const
{
    assert(id >= 0);
    assert(id < _ngrams);

    return _vocab[id].subwords;
}

std::vector<uint32_t> Dictionary::getSubwords(const std::string& word) const
{
    int32_t id = getId(word);
    if (id >= 0)
    {
        return getSubwords(id);
    }

    std::vector<uint32_t> ngrams;
    std::string::size_type pos;
    pos = word.find("@$");
    if (word != EOS && pos == word.npos)
    {
        computeSubwords(BOW + word + EOW, ngrams);
    }
    if (pos != word.npos)
    {
        std::vector<std::string> substrings;
        utils::splitString(word, "@$", substrings);
        for (int k = 0; k < substrings.size(); k++)
        {
            if (substrings[k].empty())  continue;
            id = getId(substrings[k]);
            if (id >= 0)    ngrams.push_back(id);
            if (substrings[k] != EOS)
            {
                computeSubwords(BOW + substrings[k] + EOW, ngrams);
            }
        }       
    }
    return ngrams;
}

void Dictionary::getSubwords(const std::string& word, std::vector<uint32_t>& ngrams,
                        std::vector<std::string>& substrings) const
{
    int32_t id = getId(word);
    ngrams.clear();
    substrings.clear();
    if (id >= 0)
    {
        ngrams.push_back(id);
        substrings.push_back(_vocab[id].ngram);
    }

    std::string::size_type pos;
    pos = word.find("@$");
    if (word != EOS && pos == word.npos)
    {
        computeSubwords(BOW + word + EOW, ngrams, substrings);
    }
    if (pos != word.npos)
    {
        std::vector<std::string> sub;
        utils::splitString(word, "@$", sub);
        for (int k = 0; k < sub.size(); k++)
        {
            if (sub[k].empty()) continue;
            id = getId(sub[k]);
            if (id >= 0)
            {
                ngrams.push_back(id);
                substrings.push_back(sub[k]);
            }
            if (sub[k] != EOS)
            {
                computeSubwords(BOW + sub[k] + EOW, ngrams, substrings);
            }
        }
    }
}

void Dictionary::getPairs()
{
    if (_args->input == "-")
    {
        throw std::invalid_argument("Cannot use stdin for generating word-context pairs!");    
    }

    std::ifstream ifs(_args->input);
    if (!ifs.is_open())
    {
        throw std::invalid_argument(
            _args->input + " cannot be opened for processing!");
    }
    readFromFile(ifs);
    ifs.close();

    std::ifstream in(_args->input);
    if (!in.is_open())
    {
        throw std::invalid_argument(
            _args->input + " cannot be opened for processing!");
    }

    std::string filename(_args->output);
    filename += ".pairs";
    std::ofstream out(filename);
    if (!out.is_open())
    {
        throw std::invalid_argument(
            filename + " cannot be opened for word context pairs!");
    }

    if (_args->verbose > 0)
    {
        std::cerr << "\rGenerate Pairs" << std::endl;
    }
    const uint64_t ntokens = _ntokens;
    std::vector<std::string> line;
    std::uniform_int_distribution<> int_uniform(1, _args->ws);
    std::minstd_rand rng(0);
    std::string word;
    uint64_t flag = 0;

    while (_tokenCount < ntokens)
    {
        // read line
        _tokenCount += getLine(in, line);

        for (int i = 0; i < line.size(); i++)
        {
            for (int j = 1; j <= _args->wng; j++)
            {
                word = getNgram(line, i, j);
                if (!checkWord(word, rng)) continue;

                int boundary = int_uniform(rng);
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
                            std::string context = getNgram(line, i + c, k);
                            if (!checkWord(context, rng))  continue;
                            // write word and context pairs
                            //out.write(word.data(), word.size() * sizeof(char));
                            //out.put(' ');
                            //out.write(context.data(), context.size() * sizeof(char));
                            //out.put('\n');
                            out << word << " " << context << std::endl;
                        }
                    }
                }
            }
        }
        line.clear();
        if (_args->verbose > 0)
        {
            flag++;
            if (flag % 100 == 0)
            {
                std::cerr << "\rProcessed " << _tokenCount << " tokens" << std::flush;
            }
        }
    }

    if (_args->verbose > 0)
    {
        std::cerr << "\rProcessed " << _tokenCount << " tokens" << std::endl;
    }
    out.close();
    in.close();
}

bool Dictionary::checkWord(const std::string& word, std::minstd_rand& rng) const
{
    if (word.empty())   return false;
    
    std::uniform_real_distribution<> real_uniform(0, 1);
    int32_t id = getId(word);
    if (id < 0 || discard(id, real_uniform(rng)))   return false;

    return true;
}

void Dictionary::init()
{
    initTableDiscard();
    initWordNgrams();
}

void Dictionary::saveBinaryVocab(std::ostream& out) const
{
    out.write((char *) &_ngrams, sizeof(uint32_t));
    out.write((char *) &_ntokens, sizeof(uint64_t));
    for (uint32_t i = 0; i < _ngrams; i++)
    {
        element e = _vocab[i];
        out.write(e.ngram.data(), e.ngram.size() * sizeof(char));
        out.put(0);
        out.write((char *) &(e.count), sizeof(uint64_t));
    }
}

void Dictionary::saveNonbinaryVocab(std::ostream& out) const
{
    out << _ngrams << " " << _ntokens << std::endl;
    for (uint32_t i = 0; i < _ngrams; i++)
    {
        element e = _vocab[i];
        out << e.ngram << " " << e.count << std::endl;
    }
}

void Dictionary::loadBinaryVocab(std::istream& in)
{
    _vocab.clear();
    in.read((char *) &_ngrams, sizeof(uint32_t));
    in.read((char *) &_ntokens, sizeof(uint64_t));
    for (uint32_t i = 0; i < _ngrams; i++)
    {
        char c;
        element e;
        while ((c = in.get()) != 0)
        {
            e.ngram.push_back(c);
        }
        in.read((char *) &(e.count), sizeof(uint64_t));
        _vocab.push_back(e);
    }
    init();
    uint32_t hash2intsize = std::ceil(_ngrams / 0.7);
    _hash2int.assign(hash2intsize, -1);
    for (uint32_t i = 0; i < _ngrams; i++)
    {
        uint32_t h = hash(_vocab[i].ngram);
        _hash2int[find(_vocab[i].ngram, h)] = i;
    }
}

void Dictionary::loadNonbinaryVocab(std::istream& in)
{
    _vocab.clear();
    in >> _ngrams >> _ntokens;
    
    for (uint32_t i = 0; i < _ngrams; i++)
    {
        element e;
        uint64_t c;
        std::string word;
        in >> word >> c;
        e.ngram = word;
        e.count = c;
        _vocab.push_back(e);
    }
    init();
    uint32_t hash2intsize = std::ceil(_ngrams / 0.7);
    _hash2int.assign(hash2intsize, -1);
    for (uint32_t i = 0; i < _ngrams; i++)
    {
        uint32_t h = hash(_vocab[i].ngram);
        _hash2int[find(_vocab[i].ngram, h)] = i;
    }
}

void Dictionary::dump(std::ostream& out) const
{
    out << _vocab.size() << std::endl;
    for (auto& it : _vocab)
    {
        out << it.ngram << " " << it.count << std::endl;
    }
}

}