from analysis.pymo.parsers import BVHParser
from analysis.pymo.data import Joint, MocapData
from analysis.pymo.preprocessing import *
from analysis.pymo.viz_tools import *
from analysis.pymo.writers import *
from sklearn.pipeline import Pipeline
from pathlib import Path
import sys
from feature_extraction.utils import distribute_tasks

p = BVHParser()
datas = []
filename = sys.argv[1]
cut_frames1 = int(sys.argv[2])
cut_frames2 = int(sys.argv[3])
data = p.parse(filename)

data.values = data.values.iloc[cut_frames1:-cut_frames2]

writer = BVHWriter()

with open(filename,'w') as f:
    writer.write(data, f)
