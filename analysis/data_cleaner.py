import argparse
import datetime
from analysis.pymo.parsers import BVHParser
from analysis.pymo.writers import BVHWriter
from pymo.viz_tools import *
from pymo.preprocessing import *
from sklearn.pipeline import Pipeline
from matplotlib.animation import FuncAnimation
from pathlib import Path
import csv


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# %matplotlib


def mp4_for_bvh_file(filename, output_dir):

    if len(joint_names) > 0:
        data_pipe = Pipeline([
            ('dwnsampl', DownSampler(tgt_fps=args.fps, keep_all=False)),
            ('root', RootTransformer('pos_rot_deltas')),
            ("pos", MocapParameterizer("position")),
        ])
    else:
        data_pipe = Pipeline([
            ('dwnsampl', DownSampler(tgt_fps=args.fps, keep_all=False)),
            ('root', RootTransformer('pos_rot_deltas')),
            ("pos", MocapParameterizer("position")),
        ])

    parser = BVHParser()
    parsed_data = parser.parse(filename)

    piped_data = data_pipe.fit_transform([parsed_data])
    assert len(piped_data) == 1

    render_mp4(piped_data[0], output_dir / filename.stem, axis_scale=3, elev=0, azim=45)

    return piped_data, data_pipe


def load_bvh_file(filename, joint_names=[], param="position"):
    if len(joint_names) > 0:
        data_pipe = Pipeline([
            ('dwnsampl', DownSampler(tgt_fps=args.fps, keep_all=False)),
            ('root', RootTransformer('pos_rot_deltas')),
            (param, MocapParameterizer(param)),
            ('jtsel', JointSelector(joint_names, include_root=False)),
            ('np', Numpyfier())
        ])
    else:
        data_pipe = Pipeline([
            ('dwnsampl', DownSampler(tgt_fps=args.fps, keep_all=False)),
            ('root', RootTransformer('pos_rot_deltas')),
            (param, MocapParameterizer(param)),
            ('np', Numpyfier())
        ])

    parser = BVHParser()
    parsed_data = parser.parse(filename)

    piped_data = data_pipe.fit_transform([parsed_data])
    assert len(piped_data) == 1

    return piped_data, data_pipe


def save_below_floor_tuples(below_floor_tuples, outfile_path):
    with open(outfile_path, mode='w') as csv_file:
        fieldnames = ['video_path', 'offset']
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writeheader()

        for (video_name, offset) in below_floor_tuples:
            csv_writer.writerow({
                "video_path": video_name,
                "offset": offset,
            })


def save_jump_tuples(jump_tuples, outfile_path):
    with open(outfile_path, mode='w') as csv_file:
        fieldnames = ['video_path', 'jump_time', 'jump_size']
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writeheader()

        for (video_name, jump_size, jump_time) in jump_tuples:
            csv_writer.writerow({
                "video_path": video_name,
                "jump_time": jump_time,
                "jump_size": jump_size
            })

            secs = jump_time/args.fps
            print(
                "\n\nmax jump video: {}\nmax jump time: {}\n max jump: {}".format(
                    video_name, datetime.timedelta(seconds=secs), jump_size
                ))
            print('vlc command:\nvlc "{}" --start-time {}'.format(str(video_name).replace(".bvh", ""), secs-5))


def calculate_jumps(traj):
    return np.abs(traj[1:] - traj[:-1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, help='path to the bhv files dir')
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--top-k", type=int, default=10, help="save top k biggest jumps")
    parser.add_argument("--param", type=str, default="position")
    parser.add_argument("--detect-below-floor", action="store_true")
    parser.add_argument("--floor-z", type=float, default=-0.08727)
    parser.add_argument("--ignore-first-secs", type=float, default=1)
    parser.add_argument("--plot", action="store_true", help="plot jump distributions")
    parser.add_argument("--mp4", action="store_true", help="create mp4 visualisation")
    parser.add_argument("--output-dir", default="data_cleaning")
    args = parser.parse_args()

    if args.detect_below_floor:
        if args.param != "position":
            raise ValueError("param must be position for below floor and is {}.".format(args.param))

    output_dir = Path(args.output_dir)
    jumps_output_file = (output_dir / "jumps_fps_{}_param_{}".format(args.fps, args.param)).with_suffix(".csv")
    below_floor_output_file = (output_dir / "below_floor").with_suffix(".csv")

    # which joint to load

    # joint_names = ['Spine', 'Spine1', 'Spine2', 'Neck', 'Head', 'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand',
    #      'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 'RightUpLeg', 'RightLeg', 'RightFoot',
    #      'RightToeBase', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase']

    # joint_names = ['Spine', 'Spine1', 'Spine2', 'Neck', 'Head', 'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase']
    # joint_names = ["Head"]
    joint_names = []

    all_jumps = []
    below_floor_tuples = []
    jump_tuples_per_step = []

    filenames = list(Path(args.dir).glob("*.bvh"))
    for i, filename in enumerate(filenames):
        print("[{}/{}]".format(i, len(filenames)))


        piped_data, data_pipe = load_bvh_file(filename, joint_names=joint_names, param=args.param)

        if piped_data.size == 0:
            raise ValueError("No joints found. {} ".format(joint_names))

        traj = piped_data[0]

        # jumps
        traj_jumps = calculate_jumps(traj)
        all_jumps.append(traj_jumps)

        max_per_step = traj_jumps.max(axis=-1)
        for pos, st in enumerate(max_per_step):
            # ignore jumps at the beginning
            if pos/args.fps > args.ignore_first_secs:
                jump_tuples_per_step.append((filename, st, pos))

        # below the floor
        if args.detect_below_floor:
            # detect if below the floor
            traj_per_obj = traj.reshape(traj.shape[0], -1, 3)
            min_z = traj_per_obj[:, :, 1].min()

            eps = 0.01
            if min_z < (args.floor_z - eps):
                below_floor_tuples.append((filename, min_z))

        if args.mp4:
            mp4_for_bvh_file(filename=filename, output_dir=output_dir)

    # k biggest jumps
    top_k_jumps = sorted(jump_tuples_per_step, key=lambda s: s[1], reverse=True)[:args.top_k]
    save_jump_tuples(top_k_jumps, jumps_output_file)

    save_below_floor_tuples(below_floor_tuples, below_floor_output_file)

    if args.plot:
        all_jumps = np.vstack(all_jumps)
        # plot
        for j in range(all_jumps.shape[-1]):
            joint_jumps = all_jumps[:, j]
            plt.scatter(j*np.ones_like(joint_jumps), joint_jumps, s=1)

        plt.savefig(output_dir / "joint_distances.png")
        plt.savefig(output_dir / "joint_distances.svg")



