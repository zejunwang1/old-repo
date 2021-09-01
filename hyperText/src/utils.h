#ifndef UTILS_H
#define UTILS_H

#include <fstream>
#include <vector>
#include <string>

#if defined(__clang__) || defined(__GNUC__)
# define HYPERTEXT_DEPRECATED(msg) __attribute__((__deprecated__(msg)))
#elif defined(_MSC_VER)
# define HYPERTEXT_DEPRECATED(msg) __declspec(deprecated(msg))
#else
# define HYPERTEXT_DEPRECATED(msg)
#endif

namespace hypertext
{

namespace utils
{
    int64_t size(std::ifstream&);
    void seek(std::ifstream&, int64_t);
    void splitString(const std::string&, const std::string&,
                     std::vector<std::string>&);
}

}

#endif