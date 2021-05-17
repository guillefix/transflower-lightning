import numpy as np
import sklearn
import pickle
from pathlib import Path
import scipy.linalg

# feat_file = "inference/generated_1/moglow_expmap/predicted_mods/"+"aistpp_gBR_sBM_cAll_d04_mBR3_ch10.expmap_scaled_20.generated.npy"
# feats = np.load(feat_file)
#
# feats = feats[:,0,:]
# feats = np.delete(feats,[-4,-6],1)
#
# feats.shape
#
# C = np.dot(feats.T,feats)
#
# m = np.mean(feats,0)

# data_path="data/dance_combined"
# feature_name="expmap_scaled_20"
# transform_name="scaler"
# transform = pickle.load(open(Path(data_path).joinpath(feature_name+'_'+transform_name+'.pkl'), "rb"))
#
# C_data = transform.
#
# C_data.shape

#%%
root_dir = "data/predicted_mods"
# experiment_name="moglow_expmap"

# stat="2moments" # mean and covariance of poses
stat="2moments_ext" # mean and covariance of 3 consecutive poses
moments_file = root_dir+"/"+"ground_truth"+"/bvh_expmap_"+stat+".pkl"
gt_m, gt_C = pickle.load(open(moments_file,"rb"))

def FID(m,C,mg,Cg):
    mean_diff = np.sum((m-mg)**2)
    covar_diff = np.trace(C) + np.trace(Cg) -2 * np.trace(scipy.linalg.sqrtm(np.dot(C,Cg)))
    return mean_diff + covar_diff

moments_dict = {}
fids = {}
experiments = ["moglow_expmap","transflower_expmap","transflower_expmap_finetune2_old","transformer_expmap"]
for experiment_name in experiments:
    moments_file = root_dir+"/"+experiment_name+"/expmap_scaled_20.generated_"+stat+".pkl"

    m,C = pickle.load(open(moments_file,"rb"))
    if stat=="2moments":
        m = np.delete(m,[-4,-6],0)
        C = np.delete(C,[-4,-6],0)
        C = np.delete(C,[-4,-6],1)
    elif stat=="2moments_ext":
        m = np.delete(m,[-4,-6],0)
        m = np.delete(m,[-4-67,-6-67],0)
        m = np.delete(m,[-4-67*2,-6-67*2],0)
        C = np.delete(C,[-4,-6],0)
        C = np.delete(C,[-4-67,-6-67],0)
        C = np.delete(C,[-4-67*2,-6-67*2],0)
        C = np.delete(C,[-4,-6],1)
        C = np.delete(C,[-4-67,-6-67],1)
        C = np.delete(C,[-4-67*2,-6-67*2],1)
    moments_dict[experiment_name] = (m,C)
    fids[experiment_name] = FID(m,C,gt_m,gt_C)
    

fids
