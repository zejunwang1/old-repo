#include "utils.h"

#include <ios>
#include <iostream>

namespace hypertext
{

namespace utils
{
    int64_t size(std::ifstream& ifs)
    {
        ifs.seekg(std::streamoff(0), std::ios::end);
        return ifs.tellg();
    }

    void seek(std::ifstream& ifs, int64_t pos)
    {
        ifs.clear();
        ifs.seekg(std::streampos(pos));
    }

    void splitString(const std::string& s, const std::string& a,
                     std::vector<std::string>& res)
    {
        std::string::size_type pos1, pos2;
        pos1 = 0;
        pos2 = s.find(a);

        while (pos2 != s.npos)
        {
            res.push_back(s.substr(pos1, pos2 - pos1));
            pos1 = pos2 + a.length();
            pos2 = s.find(a, pos1);
        }

        if (pos1 != s.length())
        {
            res.push_back(s.substr(pos1));
        }
    }
}

}