import re
import os
import sys
import jieba

def preprocess(filename, output):
    out = open(output, mode = 'w')
    line_num = 0
    r = '[-，。、-【】/‘’“”：；:,.\[\]（）《》{}？！⑦()$%……>℃：.”“^-——=&#@￥]+'
    with open(filename, mode = 'r', encoding = 'utf-8', errors = 'ignore') as f:
        for line in f:
            if line_num % 100 == 0:
                sys.stdout.write("\rProcessed " + str(line_num) + " lines")
                sys.stdout.flush()
            line_num += 1
            try:
                label, content = line.strip().split('\t')
                if content:
                    content = re.sub(r, '', content)
                    words = ' '.join(jieba.cut(content))
                    out.write(label + '\t' + words + '\n')
            except:
                pass
    out.close()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError("""Usage: python data_preprocess.py [text file] [output file]""")

    jieba.enable_parallel(100)
    if os.path.exists("tags5month_cms.txt"):
        jieba.load_userdict("tags5month_cms.txt")
    if os.path.exists("30wdict_utf8.txt"):
        jieba.load_userdict("30wdict_utf8.txt")
    if os.path.exists("user_dict.txt"):
        jieba.load_userdict("user_dict.txt")

    if not os.path.exists(sys.argv[1]):
        raise ValueError("""Input text file does not exist""")

    filename = sys.argv[1]
    output = sys.argv[2]
    preprocess(filename, output)
    jieba.disable_parallel()