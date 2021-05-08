# import os,sys
# THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
# sys.path.append(ROOT_DIR)
#
from analysis.pymo.parsers import BVHParser
from analysis.pymo.data import Joint, MocapData
from analysis.pymo.preprocessing import *
from analysis.pymo.viz_tools import *
from analysis.pymo.writers import *
from sklearn.pipeline import Pipeline

import matplotlib.pyplot as plt

#%%
p = BVHParser()
f1="data/dance_full/shadermotion_justdance/bvh/justdance_0.bvh"
# f2="data/dance_full/kth_streetdance_data/bvh/Streetdance_001.bvh"
# f1=sys.argv[1] #target file
# f2=sys.argv[2] #file from which to source the offsets of the skeleton (bone names and hierarchy should be the same)
f2="/home/guillefix/code/mt-lightning/data/dance_full/kth_streetdance_data/bvh/Streetdance_001.bvh"

data1 = p.parse(f1)
data2 = p.parse(f2)

data1.values
data2.values

data1.values = data1.values[data2.values.columns]

data1.skeleton = data2.skeleton

writer = BVHWriter()
with open(f1,'w') as f:
    writer.write(data1, f)

# data1.skeleton
#
# data2.skeleton
