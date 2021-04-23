from analysis.pymo.parsers import BVHParser
from analysis.pymo.data import Joint, MocapData
from analysis.pymo.preprocessing import *
from analysis.pymo.viz_tools import *
from analysis.pymo.writers import *
from sklearn.pipeline import Pipeline
from pathlib import Path
import sys
path = sys.argv[1]
#cat to_check* | parallel -L 1 -I % python3 analysis/shift_bvh.py % -34

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
data_pipe = Pipeline([
    # ('dwnsampl', DownSampler(tgt_fps=fps,  keep_all=False)),
    ('jtsel', JointSelector(['Spine', 'Spine1', 'Neck', 'Head', 'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase'], include_root=True)),
    ('pos', MocapParameterizer('position')),
])

out_data = data_pipe.fit_transform(datas)

yposs = list(filter(lambda x: x.split("_")[1]=="Yposition", out_data[0].values.columns))

with open("to_check"+str(rank),"w") as f:
    for i,d in enumerate(out_data):
        min_y = d.values[yposs].iloc[100:].mean().min()
        if min_y < -10:
            print(min_y, filenames[i].__str__())
            f.writelines(filenames[i].__str__()+"\n")
