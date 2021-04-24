from analysis.pymo.parsers import BVHParser
from analysis.pymo.data import Joint, MocapData
from analysis.pymo.preprocessing import *
from analysis.pymo.viz_tools import *
from analysis.pymo.writers import *
from sklearn.pipeline import Pipeline
from pathlib import Path
import sys
path = sys.argv[1]

from feature_extraction.utils import distribute_tasks
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

path = Path(path)
candidate_audio_files = sorted(path.glob('**/*.bvh'), key=lambda path: path.parent.__str__())
tasks = distribute_tasks(candidate_audio_files,rank,size)

p = BVHParser()
datas = []
filenames = []
for i in tasks:
    f = candidate_audio_files[i]
    print(f)
    filenames.append(f)
    datas.append(p.parse(f))

with open("to_check"+str(rank),"w") as f:
    for i,data in enumerate(datas):
        bad_ones = data.values[(data.values["Hips_Xposition"] > 100000) | (data.values["Hips_Xposition"] < -100000)]
        if len(bad_ones) > 0:
            last_index = bad_ones.index[-1]
            data.values = data.values.loc[last_index:].iloc[1:]
        writer = BVHWriter()

        with open(filenames[i],'w') as out_f:
            writer.write(data, out_f)
