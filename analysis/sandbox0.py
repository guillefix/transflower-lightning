from analysis.pymo.parsers import BVHParser
from analysis.pymo.data import Joint, MocapData
from analysis.pymo.preprocessing import *
from analysis.pymo.viz_tools import *
from analysis.pymo.writers import *
import analysis.pymo
import imp;imp.reload(analysis.pymo)
from sklearn.pipeline import Pipeline
from analysis.pymo.rotation_tools import euler2expmap

import matplotlib.pyplot as plt

#%%
p = BVHParser()

# f="data/dance_full/aistpp_bvh/bvh/gWA_sFM_cAll_d26_mWA4_ch12.bvh"
# f="data/dance_full/shadermotion_data2_retarget/bvh/VRChat_Dance_2.bvh"
# f="data/dance_full/shadermotion_data2_retarget/bvh/VRChat_Dance_8.bvh"
# f="data/dance_full/kth_streetdance_data/bvh/Streetdance_001.bvh"
# f="data/dance_full/shadermotion_justdance/bvh/justdance_0.bvh"
# f="data/dance_full/vibe_dance/bvh/Take1.bvh"
# f="data/dance_full/shadermotion_data2_retarget/bvh/VRChat_Dance_0.bvh"
f1="data/dance_full/kth_streetdance_data/bvh/Streetdance_001.bvh"
f2="data/dance_full/shadermotion_justdance/bvh/justdance_0.bvh"
f2="data/dance_full/shadermotion_justdance/bvh/justdance_1.bvh"
# f="data/dance_full/tmp/bvh/VRChat_Dance_0.bvh"
# f="data/dance_full/testing/VRChat_Dance_0.bvh"
# f="data/dance_full/tmp/bvh/VRChat_Dance_0.bvh"
# f="data/dance_full/testing/VRChat_Dance_0.bvh"

data = p.parse(f1)
data2 = p.parse(f2)
#%%

# print_skel(data)

# f="analysis/mixamo.bvh"
#
# data = p.parse(f)
#
# print_skel(data)
data.values["LeftFoot_Zrotation"][2]
data2.values["LeftFoot_Zrotation"][2]
data2.values["LeftFoot_Zrotation"] = data.values["LeftFoot_Zrotation"].values[:13250]
data2.values["LeftFoot_Xrotation"] = data.values["LeftFoot_Xrotation"].values[:13250]
data2.values["LeftFoot_Yrotation"] = data.values["LeftFoot_Yrotation"].values[:13250]
euler2expmap((data.values["LeftFoot_Zrotation"][1], data.values["LeftFoot_Xrotation"][1], data.values["LeftFoot_Yrotation"][1]), 'ZXY', True)
e1=euler2expmap((data2.values["LeftFoot_Zrotation"][0], data2.values["LeftFoot_Xrotation"][0], data2.values["LeftFoot_Yrotation"][0]), 'ZXY', True)
e2=euler2expmap((data2.values["LeftFoot_Zrotation"][1], data2.values["LeftFoot_Xrotation"][1], data2.values["LeftFoot_Yrotation"][1]), 'ZXY', True)
e3=euler2expmap((data2.values["LeftFoot_Zrotation"][2], data2.values["LeftFoot_Xrotation"][2], data2.values["LeftFoot_Yrotation"][2]), 'ZXY', True)

np.linalg.norm(e1) - np.linalg.norm(e2)
np.linalg.norm(e2) - np.linalg.norm(e3)
(2*np.pi - np.linalg.norm(e2)) - (np.linalg.norm(e3))

data.values["LeftFoot_Zrotation"].mean()
data2.values["LeftFoot_Zrotation"].mean()
list(data2.values.std())
list(data2.values.mean())
list(data.values.mean())

data.values

data.skeleton

#%%

# fps=60
# p = BVHParser()
data_pipe = Pipeline([
    # ('dwnsampl', DownSampler(tgt_fps=fps,  keep_all=False)),
    ('mir', Mirror(axis='X', append=True)),
    ('root', RootTransformer('pos_rot_deltas')),
    ('jtsel', JointSelector(['Spine', 'Spine1', 'Neck', 'Head', 'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase'], include_root=True)),
    # ('jtsel', JointSelector(['Spine1', 'Spine', 'Neck', 'Head', 'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase'], include_root=True)),
    # ('exp', MocapParameterizer('position')),
    ('exp', MocapParameterizer('expmap')),
    # ('cnst', ConstantsRemover()),
    # ('np', Numpyfier())
])


out_data = data_pipe.fit_transform([data])
out_data2 = data_pipe.fit_transform([data2])

out_data[0].values.columns[17]
out_data[0].values
out_data2[0].values
out_data2[0].values["LeftFoot_beta"].std()
out_data2[0].values["LeftFoot_beta"].max()
out_data2[0].values["LeftFoot_beta"].mean()
out_data[0].values["LeftFoot_beta"].std()
out_data[0].values["LeftFoot_beta"].max()
out_data[0].values["LeftFoot_beta"].mean()
(out_data[0].values["LeftFoot_alpha"]**2 + out_data[0].values["LeftFoot_beta"]**2 + out_data[0].values["LeftFoot_gamma"]**2).mean()
(out_data2[0].values["LeftFoot_alpha"]**2 + out_data2[0].values["LeftFoot_beta"]**2 + out_data2[0].values["LeftFoot_gamma"]**2).mean()
out_data[0].values["LeftFoot_gamma"][1]
out_data2[0].values["LeftFoot_gamma"][3]

(out_data[0].values["RightFoot_alpha"]**2 + out_data[0].values["RightFoot_beta"]**2 + out_data[0].values["RightFoot_gamma"]**2).mean()
(out_data2[0].values["RightFoot_alpha"][10:]**2 + out_data2[0].values["RightFoot_beta"][10:]**2 + out_data2[0].values["RightFoot_gamma"][10:]**2).mean()
(out_data2[0].values["RightFoot_alpha"][10:]**2 + out_data2[0].values["RightFoot_beta"][10:]**2 + out_data2[0].values["RightFoot_gamma"][10:]**2).diff().max()

np.diff(out_data2[0].values["LeftFoot_beta"]).max()
np.diff(out_data[0].values["LeftFoot_beta"]).max()

out_data[0].shape
inv_data = data_pipe.inverse_transform(out_data)
inv_data[0] == data

data.values
inv_data[0].values

# out_data[0][0]
# out_data[0].values.columns

# video_file = "analysis/tmp/Streetdance_001.mp4"
# video_file = "analysis/tmp/sm01.mp4"
video_file = "analysis/tmp/sm01b.mp4"
render_mp4(out_data[0], video_file, axis_scale=3, elev=45, azim=45)
# render_mp4(out_data[0], video_file, axis_scale=100, elev=45, azim=45)
# audio_file = "data/dance_full/kth_streetdance_data/music/Streetdance_001.wav"
# audio_file = "data/dance_full/vibe_dance/audio/audio_001.wav"
# audio_file = "data/dance_full/shadermotion_data2_retarget/audio/VRChat\ Dance_0.wav"
audio_file = "data/dance_full/testing/VRChat_Dance_0.mp3"
# audio_file = "data/dance_full/tmp/audio/VRChat\ Dance_0.wav"
from analysis.visualization.utils import generate_video_from_images, join_video_and_audio
join_video_and_audio(video_file, audio_file, 0)

yposs = list(filter(lambda x: x.split("_")[1]=="Yposition", out_data[0].values.columns))

out_data[0].values[yposs].iloc[100:].min().min()
out_data[0].values[yposs].min()
out_data[0].values[yposs].iloc[10:]
out_data[0].values["Hips_Yposition"].iloc[52]

# out_data[0].values
out_data.shape
out_data[0,:10,-1]

bvh_data=data_pipe.inverse_transform(out_data)

writer = BVHWriter()
with open('analysis/tmp/test.bvh','w') as f:
    writer.write(bvh_data[0], f)


####
last_index = data.values[(data.values["Hips_Xposition"] > 100000) | (data.values["Hips_Xposition"] < -100000)].index[-1]

data.values.loc[last_index:].iloc[1:]


##################
