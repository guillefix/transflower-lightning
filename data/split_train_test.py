
import numpy as np

def split(elements, p):
    n = len(elements)
    n1 = int(p*n)
    split1 = list(np.random.choice(elements, size=n1, replace=False))
    split2 = [x for x in elements if x not in split1]
    return split1, split2

f = open("dance_combined/base_filenames.txt", "r")
f2 = open("../analysis/aistpp_bad_ones.txt", "r")
# f3 = open("../analysis/aistpp_base_filenames_test.txt", "r")
f4 = open("../analysis/aistpp_base_filenames_train_filtered.txt", "r")
basenames = [x[:-1] for x in f.readlines()]
aistpp_bad_ones = ["aistpp_"+x[:-1] for x in f2.readlines()]
aistpp_bad_ones = np.unique(np.array(aistpp_bad_ones)).tolist()
# aistpp_test = [x[:-1] for x in f3.readlines()]
aistpp_train = ["aistpp_"+x[:-1] for x in f4.readlines()]

for line in basenames:
    print(line)

datasets = ["groovenet", "shadermotion", "vibe", "kthmisc", "Streetdance"] # other than aistpp

basenames_dict = {d: list(filter(lambda x: x.split("_")[0]==d, basenames)) for d in datasets}
p=0.9
basenames_splits = {k: split(v,p) for k,v in basenames_dict.items()}

aistpp_basenames = list(filter(lambda x: x.split("_")[0]=="aistpp", basenames))
len(aistpp_basenames)
len(aistpp_train)
aistpp_test = [x for x in aistpp_basenames if x not in aistpp_train+aistpp_bad_ones]

len(aistpp_test) + len(aistpp_train) + len(aistpp_bad_ones)

##Train split

train_basenames = []
for k,v in basenames_splits.items():
    train_basenames += v[0]

train_basenames += aistpp_train

##Test split

test_basenames = []
for k,v in basenames_splits.items():
    test_basenames += v[1]

test_basenames += aistpp_test


assert len(train_basenames) + len(test_basenames) + len(aistpp_bad_ones) == len(basenames)

with open("dance_combined/base_filenames_train.txt", "w") as f:
    f.writelines([x+"\n" for x in train_basenames])

with open("dance_combined/base_filenames_val.txt", "w") as f:
    f.writelines([x+"\n" for x in test_basenames])
