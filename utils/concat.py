from tqdm import tqdm
from glob import glob
import codecs

base_dir = "./corpora/classes"
files = glob(base_dir + '/*.txt', recursive=True)

result = []

for file in files:
    with codecs.open(file, "r", "utf8") as f:
        lines = f.readlines()
        text = []
        for line in lines:
            if len(line.split()) > 1:
                text.append(line)
        result.append(''.join(text))

with codecs.open("corpora/train2.txt", "w", "utf8") as f:
    f.write('\n'.join(result))