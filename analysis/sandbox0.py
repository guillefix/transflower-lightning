from analysis.pymo.parsers import BVHParser
from analysis.pymo.data import Joint, MocapData
from analysis.pymo.preprocessing import *
from analysis.pymo.viz_tools import *
from analysis.pymo.writers import *
from sklearn.pipeline import Pipeline

#%%
p = BVHParser()

f="data/dance_full/aistpp_bvh/bvh/gWA_sFM_cAll_d26_mWA4_ch12.bvh"

data = p.parse(f)

print_skel(data)

# f="analysis/mixamo.bvh"
#
# data = p.parse(f)
#
# print_skel(data)

#%%

fps=60
p = BVHParser()
data_pipe = Pipeline([
    ('dwnsampl', DownSampler(tgt_fps=fps,  keep_all=False)),
    # ('root', RootTransformer('pos_rot_deltas')),
    # ('mir', Mirror(axis='X', append=True)),
    ('jtsel', JointSelector(['Spine', 'Spine1', 'Neck', 'Head', 'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase'], include_root=True)),
    ('exp', MocapParameterizer('position')),
    # ('cnst', ConstantsRemover()),
    # ('np', Numpyfier())
])

out_data = data_pipe.fit_transform([data])

yposs = list(filter(lambda x: x.split("_")[1]=="Yposition", out_data[0].values.columns))

out_data[0].values[yposs].min().min()

# out_data[0].values
out_data.shape
out_data[0,:10,-1]

bvh_data=data_pipe.inverse_transform(out_data)

writer = BVHWriter()
with open('analysis/tmp/test.bvh','w') as f:
    writer.write(bvh_data[0], f)


##################
