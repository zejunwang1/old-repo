#include <iostream>
#include <queue>
#include <iomanip>
#include "hypertext.h"
#include "args.h"
#include "cooccurrence.h"

using namespace hypertext;

void printUsage()
{
    std::cerr
        << "usage: hypertext <command> <args>\n\n"
        << "The commands supported by hypertext are:\n\n"
        << "  skipgram                train a skipgram model\n"
        << "  cbow                    train a cbow model\n"
		<< "  pmi                     train a pmi model\n"
        << "  lsa                     train a lsa model\n"
        << std::endl;
}

void train(const std::vector<std::string>& args)
{
    Args a = Args();
    a.parseArgs(args);
    HyperText hypertext;
    //std::ofstream ofs(a.output + ".bin")
    hypertext.train(a);
    //hypertext.correlationEval();
    //hypertext.saveModel();
    //hypertext.saveWordVectors();
}

int main(int argc, char** argv)
{
	
    std::vector<std::string> args(argv, argv + argc);
    if (args.size() < 2)
    {
        printUsage();
        exit(EXIT_FAILURE);
    }
    std::string command(args[1]);
    if (command == "skipgram" || command == "cbow" || command == "lsa" || command == "pmi")
    {
        train(args);
    }
    else if (command == "evaluation")
    {
		
        Args a = Args();
        a.parseArgs(args);
        HyperText hypertext;
		hypertext.evaluation(a);
		
		std::vector<std::string> words;
		words.push_back("钟汉良");
		words.push_back("张天爱");
		words.push_back("习近平");
		words.push_back("特朗普");
		for(int k = 0; k < words.size(); k++)
		{
		std::string word = words[k];
        std::vector<std::string> results;
        hypertext.topKcosine(word, 10, results);
        for (int i = 0; i < results.size(); i++)
        {
            std::cout << results[i] << "   " << hypertext.cosine(word, results[i]) << std::endl;
        }
		}
    }
	else if (command == "cooccurrence")
	{
		Args a = Args();
		a.parseArgs(args);
		std::shared_ptr<Args> _a = std::make_shared<Args>(a);
		Dictionary d(_a);
		std::ifstream ifs(a.input);
		d.readUnigramVocab(ifs);
		ifs.close();
		std::ofstream ofs("vocab.bin");
		d.saveBinaryVocab(ofs);
		ofs.close();
		Cooccurrence c(_a, "vocab.bin");
		c.getCooccurrence();
	}
    return 0;
}
