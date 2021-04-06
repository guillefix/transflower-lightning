
folder=$1
py=python
n=$(nproc)
#n=1

# code for Rotmat representation for AISTPP
#mpirun -n $n $py ./scripts/feature_extraction/aistpp_to_rotmats.py $1
#mpirun -n 1 $py ./scripts/feature_extraction/extract_transform2.py $1 --feature_name pkl_joint_angles_mats --transforms scaler
# mpirun -n $n $py ./scripts/feature_extraction/apply_transforms.py $1 --feature_name pkl_joint_angles_mats --transform_name scaler --new_feature_name joint_angles_scaled

# code for Expmap representations from bvhs
mpirun -n $n $py feature_extraction/process_motions.py $1
mpirun -n 1 $py feature_extraction/extract_transform2.py $1 --feature_name bvh_expmap --transforms scaler
mpirun -n $n $py feature_extraction/apply_transforms.py $1 --feature_name bvh_expmap --transform_name scaler --new_feature_name expmap_scaled
