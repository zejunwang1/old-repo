#include "hypertext.h"

#include <iomanip>
#include <sstream>
#include <thread>
#include <vector>
#include <string>
#include <algorithm>
#include <stdexcept>
#include <numeric>

namespace hypertext
{

constexpr uint32_t HYPERTEXT_VERSION = 13;  /* version 1c */
constexpr uint32_t HYPERTEXT_FILEFORMAT_MAGIC_INT32 = 793712345;

HyperText::HyperText() {}

const Args HyperText::getArgs() const
{
    return *(_args).get();
}

int32_t HyperText::getDimension() const
{
    return _args->dim;
}

std::shared_ptr<const Dictionary> HyperText::getDictionary() const
{
    return _dict;
}

std::shared_ptr<const Matrix> HyperText::getInputMatrix() const
{
    return _input;
}

std::shared_ptr<const Matrix> HyperText::getOutputMatrix() const
{
    return _output;
}

int32_t HyperText::getWordId(const std::string& word) const
{
    int32_t id = _dict->getId(word);
    return id;
}

int32_t HyperText::getSubwordId(const std::string& subword) const
{
    int32_t h = _dict->hash(subword) % _args->bucket;
    return _dict->ngrams() + h;
}

void HyperText::train(const Args& args)
{
    _args = std::make_shared<Args>(args);
    _dict = std::make_shared<Dictionary>(_args);

    if (_args->input == "-")
    {
        throw std::invalid_argument("Cannot use stdin for training!");
    }

    std::ifstream ifs(_args->input);
    if (!ifs.is_open())
    {
        throw std::invalid_argument(
            _args->input + " cannot be opened for training!");
    }

    // get dictionary
    _dict->readFromMultiLines(ifs);
    ifs.close();

    // save dictionary
    std::string filename = _args->output + ".vocab";
    std::ofstream ofs(filename);
    if (!ofs.is_open())
    {
        throw std::invalid_argument(
            filename + " cannot be opened for saving vocabulary!");
    }
    _dict->saveNonbinaryVocab(ofs);
    ofs.close();

    // training model
    if (_args->model == model_name::cbow || _args->model == model_name::sg)
    {
        trainWord2Vec();
        saveVectors();
    }
    else if (_args->model == model_name::lsa || _args->model == model_name::pmi)
    {
        LsaModel();
        if (_args->model == model_name::lsa)
        {
            saveVectors();
        }
    }
}

void HyperText::LsaModel()
{
    // get cooccurrence
    std::string filename = _args->output + ".vocab";
    Cooccurrence CoMat(_args, filename);
    CoMat.getCooccurrence();

    // PMI and SVD Matrix
    Lsa lsa(_args, filename);
    filename = _args->output + ".cooccurrence";

    if (_args->model == model_name::lsa)
    {
        Eigen::MatrixXf W(_dict->ngrams(), _args->dim);
        lsa.getLsaVectors(filename, W);

        // convert Eigen Matrix to customized matrix
        _input = std::make_shared<Matrix>(_dict->ngrams(), _args->dim);
        _input->zero();
        for (size_t i = 0; i < W.rows(); i++)
        {
            for (size_t j = 0; j < W.cols(); j++)
            {
                _input->at(i, j) = W(i, j);
            }
        }
    }
    else
    {
        lsa.savePMITriples(filename);
    }
}

void HyperText::trainWord2Vec()
{
    // initialize word vectors
    if (_args->pretrainedVectors.length() != 0)
    {
        loadVectors(_args->pretrainedVectors);
    }
    else
    {
        _input = std::make_shared<Matrix>(_dict->ngrams() + _args->bucket, _args->dim);
        _input->initial(1.0 / _args->dim);
    }

    // initialize intermediate variable vectors
    _output = std::make_shared<Matrix>(_dict->ngrams(), _args->dim);
    _output->zero();

    // multithreading training
    startThreads();    
    _model = std::make_shared<Model>(_input, _output, _args, 0);
    _model->setTargetCounts(_dict->getCounts());
}

void HyperText::startThreads()
{
    _start = std::chrono::steady_clock::now();
    _tokenCount = 0;
    _loss = -1;
    const uint64_t ntokens = _dict->ntokens();
    std::vector<std::thread> threads;
    for (uint32_t i = 0; i < _args->thread; i++)
    {
        threads.push_back(std::thread([=]() { trainWord2VecThread(i); }));
    }

    // print log information
    while (_tokenCount < _args->epoch * ntokens)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        if (_loss >= 0 && _args->verbose > 0)
        {
            real progress = real(_tokenCount) / (_args->epoch * ntokens);
            std::cerr << "\r";
            printInfo(progress, _loss, std::cerr);
        }
    }

    for (uint32_t i = 0; i < _args->thread; i++)
    {
        threads[i].join();
    }

    if (_args->verbose > 0)
    {
        std::cerr << "\r";
        printInfo(1.0, _loss, std::cerr);
        std::cerr << std::endl;
    }
}

void HyperText::trainWord2VecThread(uint32_t threadId)
{
    std::ifstream ifs(_args->input);
    if (!ifs.is_open())
    {
        throw std::invalid_argument(
            _args->input + " cannot be opened for training!");
    }
    utils::seek(ifs, threadId * utils::size(ifs) / _args->thread);

    Model model(_input, _output, _args, threadId);
    model.setTargetCounts(_dict->getCounts());

    const uint64_t ntokens = _dict->ntokens();
    uint64_t localTokenCount = 0;
    std::vector<std::string> line;
    //std::vector<uint32_t> line;
    while (_tokenCount < _args->epoch * ntokens)
    {
        // dynamic learning rate for gradient ascent
        real progress = real(_tokenCount) / (_args->epoch * ntokens);
        real lr = _args->lr * (1.0 - progress);
        if (_args->model == model_name::cbow)
        {
            // debug
            //localTokenCount += _dict->getLine(ifs, line, model.rng);
            localTokenCount += _dict->getLine(ifs, line);
            cbow(model, lr, line);       // train cbow model
        }
        else if (_args->model == model_name::sg)
        {
            // debug
            //localTokenCount += _dict->getLine(ifs, line, model.rng);
            localTokenCount += _dict->getLine(ifs, line);
            skipgram(model, lr, line);     // train skip-gram model
        }
        if (localTokenCount > _args->lrUpdateRate)
        {
            _tokenCount += localTokenCount;
            localTokenCount = 0;
            if (threadId == 0 && _args->verbose > 0)
            {
                _loss = model.getLoss();
            }
        }
    }
    if (threadId == 0)
    {
        _loss = model.getLoss();
    }
    ifs.close();
}

void HyperText::cbow(Model& model, real lr, const std::vector<std::string>& line)
{
    std::string word;
    std::vector<uint32_t> bow;
    std::uniform_int_distribution<> uniform(1, _args->ws);
    for (int32_t w = 0; w < line.size(); w++)
    {
        for (int32_t i = _args->wng; i > 0; i--)
        {
            bow.clear();
            // get target word
            word = _dict->getNgram(line, w, i);
            if (!(_dict->checkWord(word, model.rng)))   continue;

            // dynamic window for training
            int32_t boundary = uniform(model.rng);
            for (int32_t c = -boundary; c <= boundary; c++)
            {
                if (c != 0 && w + c >= 0 && w + c < line.size())
                {
                    for (int32_t j = 1; j <= _args->cng; j++)
                    {
                        if ((c < 0 && c + j - 1 >= 0) || (c > 0 && c <= i - 1))
                        {
                            // exist overlap between word ngram and context ngram
                            continue;
                        }
                        std::string context = _dict->getNgram(line, w + c, j);
                        if (!(_dict->checkWord(context, model.rng)))    continue;
                        const std::vector<uint32_t>& ngrams = _dict->getSubwords(context);
                        bow.insert(bow.end(), ngrams.cbegin(), ngrams.cend());
                    }
                }
            }
            // model update
            model.update(bow, _dict->getId(word), lr);
        } 
    }
}

void HyperText::skipgram(Model& model, real lr, const std::vector<std::string>& line)
{
    std::string word;
    std::uniform_int_distribution<> uniform(1, _args->ws);
    for (int32_t w = 0; w < line.size(); w++)
    {
        for (int32_t i = _args->wng; i > 0; i--)
        {
            // get target word
            word = _dict->getNgram(line, w, i);
            if (!(_dict->checkWord(word, model.rng)))   continue;
            const std::vector<uint32_t>& ngrams = _dict->getSubwords(word);

            // dynamic window for training
            int32_t boundary = uniform(model.rng);
            for (int32_t c = -boundary; c <= boundary; c++)
            {
                if (c != 0 && w + c >= 0 && w + c < line.size())
                {
                    for (int32_t j = 1; j <= _args->cng; j++)
                    {
                        if ((c < 0 && c + j - 1 >= 0) || (c > 0 && c <= i - 1))
                        {
                            // exist overlap between word ngram and context ngram
                            continue;
                        }
                        std::string context = _dict->getNgram(line, w + c, j);
                        if (!(_dict->checkWord(context, model.rng)))    continue;
                        
                        // model update
                        model.update(ngrams, _dict->getId(context), lr);
                    }
                }
            }
        }
    }
}

void HyperText::printInfo(real progress, real loss, std::ostream& log_stream)
{
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    double t = std::chrono::duration_cast<std::chrono::duration<double>> (end - _start).count();
    double lr = _args->lr * (1.0 - progress);
    double wst = 0.0;

    uint64_t eta = 2592000;

    if (progress > 0 && t >= 0)
    {
        progress *= 100;
        eta = t * (100 - progress) / progress;
        wst = double(_tokenCount) / t / _args->thread;
    }

    uint32_t etah = eta / 3600;
    uint32_t etam = (eta % 3600) / 60;

    log_stream << std::fixed;
    log_stream << "Progress: ";
    log_stream << std::setprecision(1) << std::setw(5) << progress << "%";
    log_stream << " words/sec/thread: " << std::setw(7) << uint64_t(wst);
    log_stream << " lr: " << std::setw(9) << std::setprecision(6) << lr;
    log_stream << " loss: " << std::setw(9) << std::setprecision(6) << loss;
    log_stream << " ETA: " << std::setw(3) << etah;
    log_stream << "h" << std::setw(2) << etam << "m";
    log_stream << std::flush;
}

void HyperText::loadVectors(const std::string& filename)
{
    std::ifstream in(filename);
    if (!in.is_open())
    {
        throw std::invalid_argument(
            _args->pretrainedVectors + " cannot be opened for loading!");
    }

    std::vector<std::string> words;
    std::shared_ptr<Matrix> mat;
    uint64_t n, dim;
    in >> n >> dim;   // get words number and vector dimension
    if (dim != _args->dim)
    {
        throw std::invalid_argument(
            "Dimension of pretrained vectors (" + std::to_string(dim) + 
            ") does not match dimension (" + std::to_string(_args->dim) + ")!");
    }
    mat = std::make_shared<Matrix>(n, dim);
    for (size_t i = 0; i < n; i++)
    {
        std::string word;
        in >> word;
        _dict->addVocab(word);
        words.push_back(word);
        for (size_t j = 0; j < dim; j++)
        {
            in >> mat->at(i, j);
        }
    }
    in.close();

    _dict->reduceVocab(1);
    _dict->init();
    _input = std::make_shared<Matrix>(_dict->ngrams() + _args->bucket, _args->dim);
    _input->initial(1.0 / _args->dim);

    for (size_t i = 0; i < n; i++)
    {
        int32_t id = _dict->getId(words[i]);
        if (id < 0 || id >= _dict->ngrams())  continue;
        for (size_t j = 0; j < dim; j++)
        {
            _input->at(id, j) = mat->at(i, j);
        }
    }
}

void HyperText::signModel(std::ostream& out) const
{
    const uint32_t magic = HYPERTEXT_FILEFORMAT_MAGIC_INT32;
    const uint32_t version = HYPERTEXT_VERSION;
    out.write((char *) &magic, sizeof(uint32_t));
    out.write((char *) &version, sizeof(uint32_t));
}

bool HyperText::checkModel(std::istream& in)
{
    uint32_t magic;
    in.read((char *) &magic, sizeof(uint32_t));
    if (magic != HYPERTEXT_FILEFORMAT_MAGIC_INT32)
    {
        return false;
    }

    in.read((char *) &version, sizeof(uint32_t));
    if (version > HYPERTEXT_VERSION)
    {
        return false;
    }
    return true;
}

void HyperText::saveModel()
{
    std::string filename(_args->output);
    filename += ".bin";
    saveModel(filename);
}

void HyperText::saveModel(const std::string filename)
{
    std::ofstream ofs(filename, std::ofstream::binary);
    if (!ofs.is_open())
    {
        throw std::invalid_argument(filename + " cannot be opened for saving!");
    }
    signModel(ofs);
    _args->save(ofs);    // save args
    _dict->saveBinaryVocab(ofs);    // save dictionary
    
    _input->save(ofs);    // save word vector matrix
    _output->save(ofs);   // save temp matrix

    ofs.close();
}

void HyperText::saveVectors()
{
    std::string filename(_args->output);
    filename += ".vec";
    std::ofstream ofs(filename);
    if (!ofs.is_open())
    {
        throw std::invalid_argument(
            filename + " cannot be opened for saving vectors!");
    }
    ofs << _dict->ngrams() << " " << _args->dim << std::endl;
    Vector vec(_args->dim);
    for (int32_t i = 0; i < _dict->ngrams(); i++)
    {
        std::string word = _dict->getWord(i);
        getVector(vec, word);
        ofs << word << " " << vec << std::endl;
    }
    ofs.close();
}

void HyperText::addInputVector(Vector& vec, uint32_t i) const
{
    vec.addRow(*(_input), i);
}

void HyperText::getVector(Vector& vec, const std::string& word) const
{
    if (_args->model == model_name::lsa)
    {
        getLsaVector(vec, word);
    }
    else if (_args->model == model_name::cbow || _args->model == model_name::sg)
    {
        getWordVector(vec, word);
    }
}

void HyperText::getWordVector(Vector& vec, const std::string& word) const
{
    const std::vector<uint32_t>& ngrams = _dict->getSubwords(word);
    vec.zero();
    for (int32_t i = 0; i < ngrams.size(); i++)
    {
        if (ngrams[i] < _input->rows())
        {
            addInputVector(vec, ngrams[i]);
        }
    }
    
    // normalize
    if (ngrams.size() > 0)
    {
        vec.mul(1.0 / ngrams.size());
    }

    if (_args->normalize && vec.l2Norm() > 0)
    {
        vec.mul(1.0 / vec.l2Norm());
    }
}

void HyperText::getLsaVector(Vector& vec, const std::string& word) const
{
    int32_t id = getWordId(word);
    vec.zero();
    if (id < 0) return;
    vec.addRow(*(_input), id);

    // normalize
    if (_args->normalize && vec.l2Norm() > 0)
    {
        vec.mul(1.0 / vec.l2Norm());
    }
}

void HyperText::getPMIVector(Vector& vec, const std::string& word, const SMatrixXf& W) const
{
    int32_t id = getWordId(word);
    vec.zero();
    if (id < 0) return;
    Eigen::VectorXf v = W.row(id);
    for (size_t i = 0; i < W.cols(); i++)
    {
        vec[i] = v(i);
    }

    // normalize
    if (_args->normalize && vec.l2Norm() > 0)
    {
        vec.mul(1.0 / vec.l2Norm());
    }
}

void HyperText::getSubwordVector(Vector& vec, const std::string& subword) const
{
    vec.zero();
    uint32_t h = _dict->hash(subword) % _args->bucket;
    h += _dict->ngrams();
    addInputVector(vec, h);
}

void HyperText::getInputVector(Vector& vec, uint32_t i) const
{
    vec.zero();
    addInputVector(vec, i);
}

void HyperText::printNgramVectors(const std::string& word)
{
    std::vector<uint32_t> ngrams;
    std::vector<std::string> substrings;
    Vector vec(_args->dim);
    _dict->getSubwords(word, ngrams, substrings);

    for (int32_t i = 0; i < substrings.size(); i++)
    {
        vec.zero();
        vec.addRow(*(_input), ngrams[i]);
        std::cout << substrings[i] << " " << vec << std::endl;
    }
}

void HyperText::precomputeWordVectors(Matrix& wordVectors)
{
    Vector vec(_args->dim);
    wordVectors.zero();
    for (int32_t i = 0; i < _dict->ngrams(); i++)
    {
        std::string word = _dict->getWord(i);
        getVector(vec, word);
        real norm = vec.l2Norm();
        // word vector normalization
        if (norm > 0)
        {
            wordVectors.addRow(vec, i, 1.0 / norm);
        }
    }
}

real HyperText::distance(const std::string& word1, const std::string& word2) const
{
    assert(getWordId(word1) >= 0);
    assert(getWordId(word2) >= 0);
    Vector vec1(_args->dim);
    Vector vec2(_args->dim);
    getVector(vec1, word1);
    getVector(vec2, word2);
    return vec1.distance(vec2);
}

real HyperText::cosine(const std::string& word1, const std::string& word2) const
{
    assert(getWordId(word1) >= 0);
    assert(getWordId(word2) >= 0);
    Vector vec1(_args->dim);
    Vector vec2(_args->dim);
    getVector(vec1, word1);
    getVector(vec2, word2);
    return vec1.cosine(vec2);
}

void HyperText::analogies(int32_t k)
{
    std::string word;
    Vector buffer(_args->dim), query(_args->dim);
    Matrix wordVectors(_dict->ngrams(), _args->dim);
    precomputeWordVectors(wordVectors);
    std::set<std::string> banSet;
    std::cout << "Query triplet (A - B + C)? ";
    std::vector<std::pair<real, std::string>> results;
    while (true)
    {
        banSet.clear();
        query.zero();
        // get word A
        std::cin >> word;
        banSet.insert(word);
        getVector(buffer, word);
        query.addVector(buffer, 1.0);
        // get word B
        std::cin >> word;
        banSet.insert(word);
        getVector(buffer, word);
        query.addVector(buffer, -1.0);
        // get word C
        std::cin >> word;
        banSet.insert(word);
        getVector(buffer, word);
        query.addVector(buffer, 1.0);

        findNN(wordVectors, query, k, banSet, results);
        for (auto& pairs : results)
        {
            std::cout << pairs.second << " " << pairs.first << std::endl;
        }
        std::cout << "Query triplet (A - B + C)? ";
    }
}

void HyperText::findNN(const Matrix& wordVectors, const Vector& query,
                       int32_t k, const std::set<std::string>& banSet,
                       std::vector<std::pair<real, std::string>>& results)
{
    // vector dot max heap
    std::priority_queue<std::pair<real, std::string>> pq;
    real queryNorm = query.l2Norm();
    if (queryNorm < 1e-8)
    {
        queryNorm = 1.0;
    }

    for (size_t i = 0; i < wordVectors.rows(); i++)
    {
        std::string word = _dict->getWord(i);
        real d = wordVectors.dotRow(query, i);
        pq.push(std::make_pair(d / queryNorm, word));
    }

    while (k > 0 && pq.size() > 0)
    {
        auto iter = banSet.find(pq.top().second);
        if (iter == banSet.end())
        {
            results.push_back(std::make_pair(pq.top().first, pq.top().second));
            k--;
        }
        pq.pop();
    }
}

real HyperText::pearson(std::vector<std::pair<std::string, real>> golden,
                        std::vector<std::pair<std::string, real>> similarity)
{
    real avg1 = 0.0, avg2 = 0.0, numr = 0.0, den1 = 0.0, den2 = 0.0;
    for (int32_t i = 0; i < golden.size(); i++)
    {
        avg1 += golden[i].second;
        avg2 += similarity[i].second;
    }
    avg1 = avg1 / golden.size();
    avg2 = avg2 / golden.size();

    for (int32_t i = 0; i < golden.size(); i++)
    {
        numr += (golden[i].second - avg1) * (similarity[i].second - avg2);
        den1 += (golden[i].second - avg1) * (golden[i].second - avg1);
        den2 += (similarity[i].second - avg2) * (similarity[i].second - avg2);
    }
    real corr = numr / (std::sqrt(den1 * den2));
    return corr;
}

real HyperText::spearman(std::vector<std::pair<std::string, real>> golden,
                         std::vector<std::pair<std::string, real>> similarity)
{
    // get rank vector
    sort(golden.begin(), golden.end(), [](const std::pair<std::string, real>& e1,
         const std::pair<std::string, real>& e2){return e1.second > e2.second;});
    sort(similarity.begin(), similarity.end(), [](const std::pair<std::string, real>& e1,
         const std::pair<std::string, real>& e2){return e1.second > e2.second;});
    std::unordered_map<std::string, real> d1, d2;
    for (int32_t i = 0; i < golden.size(); i++)
    {
        std::vector<int32_t> idx1, idx2;
        for (int32_t j = 0; j < golden.size(); j++)
        {
            if (golden[j].second == golden[i].second)
            {
                idx1.push_back(j + 1);
            }
            if (similarity[j].second == similarity[i].second)
            {
                idx2.push_back(j + 1);
            }
        }
        if (idx1.size() == 1)
        {
            d1.insert(std::make_pair(golden[i].first, real(i + 1.0)));
        }
        else
        {
            int32_t sum = 0;
            for (int32_t k = 0; k < idx1.size(); k++)
            {
                sum += idx1[k];
            }
            d1.insert(std::make_pair(golden[i].first, sum / idx1.size()));
        }

        if (idx2.size() == 1)
        {
            d2.insert(std::make_pair(similarity[i].first, real(i + 1.0)));
        }
        else
        {
            int32_t sum = 0;
            for (int32_t k = 0; k < idx2.size(); k++)
            {
                sum += idx2[k];
            }
            d2.insert(std::make_pair(similarity[i].first, sum / idx2.size()));
        }
    }

    // compute spearman correlation
    real avg1 = 0.0, avg2 = 0.0, numr = 0.0, den1 = 0.0, den2 = 0.0;
    for (auto& iter : d1)
    {
        std::string key = iter.first;
        avg1 += d1[key];
        avg2 += d2[key];
    }
    avg1 /= golden.size();
    avg2 /= golden.size();
    
    for (auto& iter : d1)
    {
        std::string key = iter.first;
        numr += (d1[key] - avg1) * (d2[key] - avg2);
        den1 += (d1[key] - avg1) * (d1[key] - avg1);
        den2 += (d2[key] - avg2) * (d2[key] - avg2);
    }
    real corr = numr / (std::sqrt(den1 * den2));
    return corr;
}

void HyperText::evaluation(const Args& args)
{
    _args = std::make_shared<Args>(args);
    _dict = std::make_shared<Dictionary>(_args);
    if (_args->pretrainedVectors.length() == 0)
    {
        throw std::invalid_argument("Empty pretrained vectors file!");
    }
    loadVectors(_args->pretrainedVectors);
    //correlationEval();
}

void HyperText::correlationEval()
{
    std::string fn(_args->eval);
    if (fn.empty())
    {
        throw std::invalid_argument("Empty evaluation file!");
    }

    std::ifstream ifs(fn);
    if (!ifs.is_open())
    {
        throw std::invalid_argument(
            _args->eval + "cannot be loaded for evaluation!");
    }
    
    std::string line;
    std::vector<std::pair<std::string, real>> golden;
    std::vector<std::pair<std::string, real>> similarity;
    while (std::getline(ifs, line))
    {
        std::vector<std::string> s;
        utils::splitString(line, "\t", s);
        std::string word1 = s[0];
        std::string word2 = s[1];
        if (_dict->getId(word1) < 0 || _dict->getId(word2) < 0)
            continue;
        std::string word = word1 + "@$" + word2;
        real r1 = std::atof(s[2].c_str());
        real r2 = cosine(word1, word2);
        golden.push_back(std::make_pair(word, r1));
        similarity.push_back(std::make_pair(word, r2));
        line.clear();
        s.clear();
    }
    ifs.close();

    // compute pearson correlation
    real corr = pearson(golden, similarity);
    std::cout << "Pearson correlation is: " << std::setprecision(4) << corr << std::endl;

    // compute spearman correlation
    corr = spearman(golden, similarity);
    std::cout << "Spearman correlation: " << std::setprecision(4) << corr << std::endl;
}

void HyperText::topKcosine(const std::string& word, int32_t k,
                           std::vector<std::string>& results)
{
    assert(_dict->getId(word) >= 0);
    Vector vec(_args->dim);
    getVector(vec, word);
    std::priority_queue<std::pair<real, std::string>> pq;
    for (size_t i = 0; i < _dict->ngrams(); i++)
    {
        std::string query = _dict->getWord(i);
        if (query == word)  continue;
        real score = cosine(word, query);
        pq.push(std::make_pair(score, query));
    }

    while (k > 0 && pq.size() > 0)
    {
        results.push_back(pq.top().second);
        pq.pop();
        k--;
    }
}

void HyperText::loadModel(const std::string& fn)
{
    std::ifstream ifs(fn, std::ifstream::binary);
    if (!ifs.is_open())
    {
        throw std::invalid_argument(
            fn + " cannot be opened for loading!");
    }
    if (!checkModel(ifs))
    {
        throw std::invalid_argument(fn + " has wrong file format!");
    }
    loadModel(ifs);
    ifs.close();
}

void HyperText::loadModel(std::istream& in)
{
    _args = std::make_shared<Args>();
    _input = std::make_shared<Matrix>();
    _output = std::make_shared<Matrix>();
    _args->load(in);
    _dict = std::make_shared<Dictionary>(_args, in);
    _input->load(in);
    _output->load(in);
    _model = std::make_shared<Model>(_input, _output, _args, 0);
    _model->setTargetCounts(_dict->getCounts());
}

void HyperText::saveOutput()
{
    std::ofstream ofs(_args->output + ".output");
    if (!ofs.is_open())
    {
        throw std::invalid_argument(
            _args->output + ".output" + " cannot be opened for saving vectors!");
    }

    uint32_t n = _dict->ngrams();
    ofs << n << " " << _args->dim << std::endl;
    Vector vec(_args->dim);
    for (int32_t i = 0; i < n; i++)
    {
        std::string word = _dict->getWord(i);
        vec.zero();
        vec.addRow(*(_output), i);
        ofs << word << " " << vec << std::endl;
    }
    ofs.close();
}

}
