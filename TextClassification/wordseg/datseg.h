#include <darts.h>
#include <string>
#include <memory>
#include <iostream>
#include <vector>
#include <fstream>
#include <algorithm>

using namespace std;

namespace Darts
{
	class DAT
	{
	protected:
		string dict_path_;
        string user_dict_path_;
        int* WLEN;
		std::shared_ptr<DoubleArray> da;
	public:
		DAT(string dict_path, string user_dict_path)
			: dict_path_(dict_path), user_dict_path_(user_dict_path)
		{
			build_dict();
            s_initial_segdict();
		}
        ~DAT() { delete []WLEN;  }

		int build_dict()
		{
            ifstream in(dict_path_);
            cout << "Init trie dict!may take some time!!\n";
            if (!in.is_open())
            {
                throw invalid_argument("dict path cannot be opened for loading!");
            }

            int num = 0;
            vector<string> WLIST;
            std::string sTmp;
            while (in >> sTmp)
                WLIST.push_back(sTmp);

            if (!user_dict_path_.empty())
            {
                ifstream ifs(user_dict_path_);
                if (!in.is_open())
                {
                    throw invalid_argument("user dict path cannot be opened for loading!");
                }
                while (ifs >> sTmp)
                    WLIST.push_back(sTmp);
            }

            sort(WLIST.begin(), WLIST.end());
            WLIST.erase(unique(WLIST.begin(), WLIST.end()), WLIST.end());
            printf("WLIST.size()=%d %d\n", num, WLIST.size());

            int dsize = WLIST.size();
            int* w_len = new int[dsize];
            int* val = new int[dsize];

            char** skey;
            skey = new char* [dsize];
            std::ofstream opt("sortutf8dict.txt");
            for (int i = 0; i < WLIST.size(); i++)
            {
                skey[i] = new char[WLIST[i].size() + 1];
                strcpy(skey[i], WLIST[i].c_str());
                skey[i][WLIST[i].size()] = '\0';
                w_len[i] = WLIST[i].size();
                val[i] = i;
                opt << skey[i] << "\n";
            }
            num = WLIST.size();
            in.close();
            opt.close();
            printf("WLIST.size()=%d %d\n", num, WLIST.size());

            da = std::make_shared<DoubleArray>();
            int s = da->build(num, skey, 0, val);
            da->save("utf8dict.data");

            FILE* fp = fopen("utf8wlen.data", "wb+");
            fwrite(w_len, sizeof(int), num, fp);
            fclose(fp);

            for (int i = 0; i < num; i++)
            {
                delete[] skey[i];
            }
            delete[] skey;
            delete[] val;
            delete[] w_len;
            return num;
		}

        int s_initial_segdict()
        {
            if (da->open("utf8dict.data") == -1)
                return -4;
            FILE* fp = fopen("utf8wlen.data", "rb");
            if (fp == NULL)
                return -2;
            fseek(fp, 0, SEEK_END);

            int wnum = ftell(fp) / 4;
            WLEN = new int[wnum];
            fseek(fp, 0, SEEK_SET);
            int tn = fread(WLEN, 4, wnum, fp);

            printf("%d %d\n", wnum, tn);

            if (tn < 0)
                return -3;
            fclose(fp);
            return 1;
        }

        int utf8txt_seg_merge(const char* buf, int* pos, int* mark)
        {
            int r[256];
            int bLen = strlen(buf);
            int bpos = 0;
            int i = 0, state = 1, lastn = 0, lastr = 0;

            int islanguagenull = 0;

            while (bpos < bLen)
            {
                //printf("bLen,bpos==%d %d\n",bLen,bpos);
                size_t result = da->commonPrefixSearch(buf + bpos, r, 256, bLen - bpos);
                //printf("bLen,bpos==%d %d %d\n", bLen, bpos, result);

                if (result == 1)
                {
                    //for(int o=0;o<result&&o<256;o++)
                    //printf("%d ",r[o]);
                    if (r[0] == 10)
                        islanguagenull++;
                }
                if (result < 1)
                {
                    if (state == 1)
                        pos[i] = bpos;

                    if (isascii(buf[bpos]))
                        bpos++;
                    else
                    {
                        bpos++;
                        while (bpos < bLen && (buf[bpos] & 0xC0) == 0x80)
                        {
                            bpos++;
                        }
                    }
                    state = 0;
                    continue;
                }
                else
                {
                    if (state == 0)
                    {
                        pos[i + 1] = bpos - pos[i];
                        i += 2;
                    }

                    if (result == 1 && lastn == 1)
                    {
                        pos[i - 1] += WLEN[r[0]];
                        bpos += WLEN[r[0]];
                        state = 1;
                        lastn = 0;
                        mark[i - 1] = lastr;
                        lastr = 0;
                        continue;
                    }
                    pos[i++] = bpos;
                    if (WLEN[r[0]] > 3)
                    {
                        if (result > 1)
                        {
                            lastn = 1;
                            lastr = r[1];
                        }
                        else
                            lastn = 0;

                        pos[i] = WLEN[r[0]];
                        mark[i] = r[0];
                    }
                    else
                    {
                        if (result > 1)
                        {
                            pos[i] = WLEN[r[1]];
                            mark[i] = r[1];
                        }
                        else
                        {
                            pos[i] = WLEN[r[0]];
                            mark[i] = r[0];
                            lastr = 0;
                        }
                        if (result > 2)
                        {
                            lastn = 1;
                            lastr = r[2];
                        }
                        else
                        {
                            lastn = 0;
                            lastr = 0;
                        }
                    }

                    bpos += pos[i];
                    i++;
                    state = 1;
                }
            }

            if (state == 0)
            {
                pos[i + 1] = bpos - pos[i];
                i += 2;
            }

            if (bLen > 0 && ((islanguagenull * 100) / bLen) > 50)
                return -1;
            else
                return i;
        }
        
        int txt_maxmatch_utf8_seg(const char* buf, int* pos, int* mark)
        {
            int r[256];
            int bLen = strlen(buf);
            int bpos = 0;
            int i = 0, state = 1;

            while (bpos < bLen)
            {
                size_t result = da->commonPrefixSearch(buf + bpos, r, 256, bLen - bpos);
                if (result < 1)
                {
                    if (state == 1)
                        pos[i] = bpos;
                    if (isascii(buf[bpos]))
                        bpos++;
                    else
                    {
                        bpos++;
                        while (bpos < bLen && (buf[bpos] & 0xC0) == 0x80)
                        {
                            bpos++;
                        }
                    }
                    state = 0;
                    continue;
                }
                else
                {
                    if (state == 0)
                    {
                        pos[i + 1] = bpos - pos[i];
                        i += 2;
                    }
                    mark[i] = r[result - 1];
                    pos[i] = bpos;
                    pos[i + 1] = WLEN[r[result - 1]];
                    bpos += pos[i + 1];
                    state = 1;
                    i = i + 2;
                }
            }
            if (state == 0)
            {
                pos[i + 1] = bpos - pos[i];
                i += 2;
            }
            return i;
        }

        void cut(string sentence, vector< pair<int, string> >& words, bool maxmatch = true)
        {
            int len = sentence.length();
            int* pos = new int[len * 2 + 1];
            int* mark = new int[len * 2 + 1];
            memset(pos, 0, sizeof(int));
            memset(mark, 0, sizeof(int));
            int n = 0;
            if (!maxmatch)
                n = utf8txt_seg_merge(sentence.c_str(), pos, mark);
            else
                n = txt_maxmatch_utf8_seg(sentence.c_str(), pos, mark);
            for (int i = 0; i < n; i += 2)
            {
                pair<int, string> p = make_pair(pos[i], sentence.substr(pos[i], pos[i + 1]));
                //string word = sentence.substr(pos[i], pos[i + 1]);
                words.push_back(p);
            }
            delete []pos;
            delete []mark;
        }

        void cut(string sentence, vector<string>& words, bool maxmatch = true)
        {
            vector< pair<int, string> > pos_words;
            cut(sentence, pos_words, maxmatch);
            for (const auto& p : pos_words)
                words.push_back(p.second);
        }
	};
}
