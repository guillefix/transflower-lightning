from scipy.spatial.transform import Rotation as R
import numpy as np

def rot_mats_to_eulers(predicted_mats):
    ident = np.eye(3, dtype=np.float32)
    angle_axes = np.zeros((predicted_mats.shape[0],72))
    for i,joints in enumerate(predicted_mats):
        joints = joints[0]
        trans = joints[216:]
        joints = joints[:216].reshape(-1,9)
        new_thing = np.zeros(72)
        # new_thing[69:] = trans
        for j,mat in enumerate(joints):
            mat = mat.reshape(3,3) + ident
            rot = R.from_matrix(mat)
            rot = rot.as_rotvec()
            new_thing[j*3:(j+1)*3] = rot
        angle_axes[i] = new_thing

    smpl_thing = {'smpl_loss':1.8,'smpl_poses':angle_axes,'smpl_trans':predicted_mats[:,0,216:], 'smpl_scaling': np.array([95])}
    # pickle.dump(smpl_thing, open("analysis/aistplusplus_api/last.generated.test.pkl", "wb"))
    return smpl_thing
