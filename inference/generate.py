from training.datasets import create_dataset, create_dataloader
from models import create_model
from training.options.train_options import TrainOptions
import pytorch_lightning as pl
import numpy as np
import pickle, json
import sklearn

if __name__ == '__main__':
    print("Hi")
    experiment_name="aistpp"
    opt = json.loads(open("../training/experiments/"+experiment_name+"/opt.json","r").read())
    class Struct:
        def __init__(self, **entries):
            self.__dict__.update(entries)
    opt = Struct(**opt)
    model = create_model(opt)
    model = model.load_from_checkpoint("../lightning_logs/version_2/checkpoints/epoch=184-step=739.ckpt", opt=opt)

    # seq_id="gLH_sBM_cAll_d16_mLH1_ch04"
    # seq_id="gWA_sBM_cAll_d26_mWA1_ch10"
    # seq_id="gWA_sFM_cAll_d27_mWA2_ch17"
    # seq_id="gLO_sBM_cAll_d14_mLO4_ch05"
    seq_id="gHO_sFM_cAll_d19_mHO1_ch02"
    # seq_id="mambo"

    # sf = np.load("data/features/"+seq_id+".mel_ddcpca_scaled.npy")
    sf = np.load("../test_data/"+seq_id+".mel_ddcpca_scaled.npy")
    # sf = np.load("test_data/"+seq_id+".mp3_mel_ddcpca.npy")
    # mf = np.load("data/features/"+seq_id+".joint_angles_scaled.npy")
    mf = np.load("../test_data/"+seq_id+".joint_angles_scaled.npy")

    features = {}
    features["in_joint_angles_scaled"] = np.expand_dims(np.expand_dims(mf.transpose(1,0),0),0)
    features["in_mel_ddcpca_scaled"] = np.expand_dims(np.expand_dims(sf.transpose(1,0),0),0)
    # features["in_pkl_joint_angles_mats"] = np.expand_dims(np.expand_dims(mf.transpose(1,0),0),0)
    # features["in_mp3_mel_ddcpca"] = np.expand_dims(np.expand_dims(sf.transpose(1,0),0),0)

    model.cuda()
    predicted_modes = model.generate(features)[0].cpu().numpy()

    # predicted_modes = (predicted_modes[0].cpu().numpy()*mf_std + mf_mean)
    transform = pickle.load(open("../test_data"+'/'+'pkl_joint_angles_mats'+'_'+'scaler'+'.pkl', "rb"))
    predicted_modes = transform.inverse_transform(predicted_modes)

    print(predicted_modes)

    np.save(seq_id+".pkl_joint_angles_mats.generated.test.npz",predicted_modes)
