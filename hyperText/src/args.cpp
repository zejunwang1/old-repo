#include "args.h"

#include <stdlib.h>

#include <iostream>
#include <stdexcept>

namespace hypertext
{

Args::Args()
{
    ws = 5;                 // window size
    dim = 100;              // vector dimension
    verbose = 1;            // print log info
    minCount = 5;           // min word count
    normalize = 1;          // word vectors normalization
    model = model_name::sg; // default skip-gram model

    eval = "";              // similarity evaluation dataset file
    pretrainedVectors = ""; // pretrained vectors filename

    lr = 0.05;              // learning rate
    lrUpdateRate = 100;     // dynamic rate threshold
    wng = 1;                // word ngram
    cng = 1;                // context ngram 
    epoch = 5;              // iterations
    neg = 5;                // negative samples
    loss = loss_name::ns;   // negative sampling
    bucket = 2000000;       // bucket number
    minn = 3;               // min char ngram
    maxn = 8;               // max char ngram   
    thread = 12;            // thread number   
    t = 1e-4;               // subsampling threshold   
    saveOutput = false;     // save output flag

    memory = 8;             // memory limit for cooccurrence
    disWeight = 1;          // distance weighting
    maxProduct = 0;         // top-left size
    overflow = 0;           // overflow length
    chunkSize = 0;          // shuffle chunk size

    eig = 0.0;              // singular values power
    transpose = false;      // lsa vectors selection
}

std::string Args::lossToString(loss_name ln) const
{
    switch (ln)
    {
        case loss_name::hs:
            return "hs";
        case loss_name::ns:
            return "ns";
    }
    return "Unknown loss!";    // should never happen
}

std::string Args::boolToString(bool b) const
{
    if (b)
    {
        return "true";
    }
    else
    {
        return "false";
    }
}

std::string Args::modelToString(model_name mn) const
{
    switch (mn)
    {
        case model_name::cbow:
            return "cbow";
        case model_name::sg:
            return "sg";
        case model_name::pmi:
            return "pmi";
        case model_name::lsa:
            return "lsa";
    }
    return "Unknown model name!";  // should never happen
}

void Args::printHelp()
{
    printBasicHelp();
    printDictionaryHelp();
    printTrainingHelp();
}

void Args::printBasicHelp()
{
    std::cerr
        << "\nThe following arguments are mandatory:\n" 
        << "  -input              training file path\n" 
        << "  -output             output file path\n" 
        << "\nThe following arguments are optional:\n" 
        << "  -eval               evaluation dataset file [" << eval << "]\n" 
        << "  -verbose            verbosity level [" << verbose << "]\n";
}

void Args::printDictionaryHelp()
{
    std::cerr
        << "\nThe following arguments for the dictionary are optional:\n"
        << "  -minCount           minimal number of word occurences [" << minCount << "]\n"
        << "  -bucket             number of buckets [" << bucket << "]\n"
        << "  -minn               minimal length of char ngram [" << minn << "]\n"
        << "  -maxn               maximal length of char ngram [" << maxn << "]\n"
        << "  -wng                maximal length of word ngram [" << wng << "]\n"
        << "  -cng                maximal length of context ngram [" << cng << "]\n"
        << "  -t                  sampling threshold [" << t << "]\n";
}

void Args::printTrainingHelp()
{
    std::cerr
        << "\nThe following arguments for training are optional:\n"
        << "  -ws                 size of context window [" << ws << "]\n"
        << "  -dim                size of word vectors [" << dim << "]\n"
        << "  -normalize          whether word vectors should be normalized [" << normalize << "]\n"
        << "  -model              model frame {cbow, skip-gram} [" << modelToString(model) << "]\n"
        << "  -lr                 learning rate [" << lr << "]\n"
        << "  -lrUpdateRate       change the rate of updates for the learning rate [" << lrUpdateRate << "]\n" 
        << "  -epoch              number of epochs [" << epoch << "]\n"
        << "  -neg                number of negatives sampled [" << neg << "]\n"
        << "  -loss               loss function {ns, hs} [" << lossToString(loss) << "]\n"
        << "  -thread             number of threads [" << thread << "]\n"
        << "  -pretrainedVectors  pretrained word vectors file [" << pretrainedVectors << "]\n"
        << "  -saveOutput         whether output params should be saved [" << boolToString(saveOutput) << "]\n"
        << "  -memory             memory limit for word-context cooccurrence [" << memory << "]\n"
        << "  -disWeight          whether distance weighting should be used [" << disWeight << "]\n"
        << "  -maxProduct         top-left dense cooccurrence matrix size [" << maxProduct << "]\n"
        << "  -overflow           overflow length for sparse cooccurrence matrix [" << overflow << "]\n"
        << "  -chunkSize          chunk size for shuffling word context cooccurrences [" << chunkSize << "]\n"
        << "  -eig                singular values power [" << eig << "]\n"
        << "  -transpose          lsa vectors selection [" << transpose << "]\n";
}

void Args::parseArgs(const std::vector<std::string>& args)
{
    std::string command(args[1]);
    if (command == "cbow")
    {
        model = model_name::cbow;
    }
    else if (command == "pmi")
    {
        model = model_name::pmi;
    }
    else if (command == "lsa")
    {
        model = model_name::lsa;
    }
    for (int i = 2; i < args.size(); i += 2)
    {
        if (args[i][0] != '-')
        {
            std::cerr << "Provided argument without a dash! Usage:" << std::endl;
            printHelp();
            exit(EXIT_FAILURE);
        }
        try
        {
            if (args[i] == "-h")
            {
                std::cerr << "Here is the help! Usage:" << std::endl;
                printHelp();
                exit(EXIT_FAILURE);
            }
            else if (args[i] == "-input")
            {
                input = std::string(args.at(i + 1));
            }
            else if (args[i] == "-output")
            {
                output = std::string(args.at(i + 1));
            }
            else if (args[i] == "-eval")
            {
                eval = std::string(args.at(i + 1));
            }
            else if (args[i] == "-lr")
            {
                lr = std::stof(args.at(i + 1));
            }
            else if (args[i] == "-lrUpdateRate")
            {
                lrUpdateRate = std::stoi(args.at(i + 1));
            }
            else if (args[i] == "-dim")
            {
                dim = std::stoi(args.at(i + 1));
            }
            else if (args[i] == "-ws")
            {
                dim = std::stoi(args.at(i + 1));
            }
            else if (args[i] == "-wng")
            {
                wng = std::stoi(args.at(i + 1));
            }
            else if (args[i] == "-cng")
            {
                cng = std::stoi(args.at(i + 1));
            }
            else if (args[i] == "-epoch")
            {
                epoch = std::stoi(args.at(i + 1));
            }
            else if (args[i] == "-minCount")
            {
                minCount = std::stoi(args.at(i + 1));
            }
            else if (args[i] == "-neg")
            {
                neg = std::stoi(args.at(i + 1));
            }
            else if (args[i] == "-bucket")
            {
                bucket = std::stoi(args.at(i + 1));
            }
            else if (args[i] == "-minn")
            {
                minn = std::stoi(args.at(i + 1));
            }
            else if (args[i] == "-maxn")
            {
                maxn = std::stoi(args.at(i + 1));
            }
            else if (args[i] == "-thread")
            {
                thread = std::stoi(args.at(i + 1));
            }
            else if (args[i] == "-t")
            {
                t = std::stoi(args.at(i + 1));
            }
            else if (args[i] == "-verbose")
            {
                verbose = std::stoi(args.at(i + 1));
            }
            else if (args[i] == "-pretrainedVectors")
            {
                pretrainedVectors = std::string(args.at(i + 1));
            }
            else if (args[i] == "-saveOutput")
            {
                saveOutput = true;
                i--;
            }
            else if (args[i] == "-loss")
            {
                if (args.at(i + 1) == "hs")
                {
                    loss = loss_name::hs;
                }
                else if (args.at(i + 1) == "ns")
                {
                    loss = loss_name::ns;
                }
                else
                {
                    std::cerr << "Unknown loss: " << args.at(i + 1) << std::endl;
                    printHelp();
                    exit(EXIT_FAILURE);
                }
            }
            else if (args[i] == "-memory")
            {
                memory = std::stoi(args.at(i + 1));
            }
            else if (args[i] == "-disWeight")
            {
                disWeight = std::stoi(args.at(i + 1));
            }
            else if (args[i] == "-maxProduct")
            {
                std::string param = args.at(i + 1);
                maxProduct = std::strtoll(param.c_str(), NULL, 10);
            }
            else if (args[i] == "-overflow")
            {
                std::string param = args.at(i + 1);
                overflow = std::strtoll(param.c_str(), NULL, 10);
            }
            else if (args[i] == "-chunkSize")
            {
                std::string param = args.at(i + 1);
                chunkSize = std::strtoll(param.c_str(), NULL, 10);
            }
            else if (args[i] == "-eig")
            {
                eig = std::stof(args.at(i + 1));
            }
            else if (args[i] == "-transpose")
            {
                transpose = true;
                i--;
            }
            else if (args[i] == "-normalize")
            {
                normalize = std::stoi(args.at(i + 1));
            }
            else
            {
                std::cerr << "Unknown argument: " << args[i] << std::endl;
                printHelp();
                exit(EXIT_FAILURE);
            }
        }
        catch (std::out_of_range)
        {
            std::cerr << args[i] << " is missing an argument." << std::endl;
            printHelp();
            exit(EXIT_FAILURE);
        }
    }
    if (input.empty() || output.empty())
    {
        std::cerr << "Empty input or output path." << std::endl;
        printHelp();
        exit(EXIT_FAILURE);
    }
}

void Args::load(std::istream& in)
{
    in.read((char *) &(ws), sizeof(int));
    in.read((char *) &(dim), sizeof(int));
    in.read((char *) &(minCount), sizeof(int));
    in.read((char *) &(normalize), sizeof(int));
    in.read((char *) &(model), sizeof(model_name));

    in.read((char *) &(lr), sizeof(double));
    in.read((char *) &(lrUpdateRate), sizeof(int));
    in.read((char *) &(wng), sizeof(int));
    in.read((char *) &(cng), sizeof(int));
    in.read((char *) &(epoch), sizeof(int));
    in.read((char *) &(neg), sizeof(int));
    in.read((char *) &(loss), sizeof(loss_name));
    in.read((char *) &(bucket), sizeof(int));
    in.read((char *) &(minn), sizeof(int));
    in.read((char *) &(maxn), sizeof(int));
    in.read((char *) &(thread), sizeof(int));
    in.read((char *) &(t), sizeof(double));

    in.read((char *) &(memory), sizeof(int));
    in.read((char *) &(disWeight), sizeof(int));
    in.read((char *) &(maxProduct), sizeof(uint64_t));
    in.read((char *) &(overflow), sizeof(uint64_t));
    in.read((char *) &(chunkSize), sizeof(uint64_t));

    in.read((char *) &(eig), sizeof(double));
}

void Args::save(std::ostream& out) const
{
    out.write((char *) &(ws), sizeof(int));
    out.write((char *) &(dim), sizeof(int));
    out.write((char *) &(minCount), sizeof(int));
    out.write((char *) &(normalize), sizeof(int));
    out.write((char *) &(model), sizeof(model_name));
    
    out.write((char *) &(lr), sizeof(double));
    out.write((char *) &(lrUpdateRate), sizeof(int));
    out.write((char *) &(wng), sizeof(int));
    out.write((char *) &(cng), sizeof(int));
    out.write((char *) &(epoch), sizeof(int));
    out.write((char *) &(neg), sizeof(int));
    out.write((char *) &(loss), sizeof(loss_name));
    out.write((char *) &(bucket), sizeof(int));
    out.write((char *) &(minn), sizeof(int));
    out.write((char *) &(maxn), sizeof(int));
    out.write((char *) &(thread), sizeof(int));
    out.write((char *) &(t), sizeof(double));

    out.write((char *) &(memory), sizeof(int));
    out.write((char *) &(disWeight), sizeof(int));
    out.write((char *) &(maxProduct), sizeof(uint64_t));
    out.write((char *) &(overflow), sizeof(uint64_t));
    out.write((char *) &(chunkSize), sizeof(uint64_t));

    out.write((char *) &(eig), sizeof(double));
}

void Args::dump(std::ostream& out) const
{
    out << "ws" << " " << ws << std::endl;
    out << "dim" << " " << dim << std::endl;
    out << "minCount" << " " << minCount << std::endl;
    out << "normalize" << " " << normalize << std::endl;
    out << "model" << " " << modelToString(model) << std::endl;

    out << "lr" << " " << lr << std::endl;
    out << "lrUpdateRate" << " " << lrUpdateRate << std::endl;
    out << "wng" << " " << wng << std::endl;
    out << "cng" << " " << cng << std::endl;
    out << "epoch" << " " << epoch << std::endl;
    out << "neg" << " " << neg << std::endl;
    out << "loss" << " " << lossToString(loss) << std::endl;
    out << "bucket" << " " << bucket << std::endl;
    out << "minn" << " " << minn << std::endl;
    out << "maxn" << " " << maxn << std::endl;
    out << "thread" << " " << thread << std::endl;
    out << "t" << " " << t << std::endl;
    out << "saveOutput" << " " << boolToString(saveOutput) << std::endl;

    out << "memory" << " " << memory << std::endl;
    out << "disWeight" << " " << disWeight << std::endl;
    out << "maxProduct" << " " << maxProduct << std::endl;
    out << "overflow" << " " << overflow << std::endl;
    out << "chunkSize" << " " << chunkSize << std::endl;

    out << "eig" << " " << eig << std::endl;
    out << "transpose" << " " << boolToString(transpose) << std::endl;
}

}
