import os
import argparse
import multiprocessing as mp
import torch
import importlib
import pkgutil
import models
import training.datasets as data
import json, yaml
import training.utils as utils
from argparse import Namespace
from training.utils import get_latest_checkpoint_path

class BaseOptions:
    def __init__(self):
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                         add_help=False)  # TODO - check that help is still displayed
        # parser.add_argument('--task', type=str, default='training', help="Module from which dataset and model are loaded")
        parser.add_argument('-d', '--data_dir', type=str, default='data/scaled_features')
        parser.add_argument('--hparams_file', type=str, default=None)
        parser.add_argument('--dataset_name', type=str, default="multimodal")
        parser.add_argument('--base_filenames_file', type=str, default="base_filenames_train.txt")
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        parser.add_argument('--batch_size', default=1, type=int)
        parser.add_argument('--val_batch_size', default=1, type=int, help='batch size for validation data loader')
        parser.add_argument('--do_validation', action='store_true', help='whether to do validation steps during training')
        parser.add_argument('--do_testing', action='store_true', help='whether to do evaluation on test set at the end of training')
        parser.add_argument('--skip_training', action='store_true', help='whether to not do training (only useful when doing just testing)')
        parser.add_argument('--do_tuning', action='store_true', help='whether to not do the tuning phase (e.g. to tune learning rate)')
        # parser.add_argument('--augment', type=int, default=0)
        parser.add_argument('--model', type=str, default="transformer", help="The network model used for beatsaberification")
        # parser.add_argument('--init_type', type=str, default="normal")
        # parser.add_argument('--eval', action='store_true', help='use eval mode during validation / test time.')
        parser.add_argument('--workers', default=0, type=int, help='the number of workers to load the data')
        # see here for guidelines on setting number of workers: https://discuss.pytorch.org/t/guidelines-for-assigning-num-workers-to-dataloader/813
        # and here https://pytorch-lightning.readthedocs.io/_/downloads/en/latest/pdf/ (where they recommend to use accelerator=ddp rather than ddp_spawn)
        parser.add_argument('--experiment_name', default="experiment_name", type=str)
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        parser.add_argument('--fork_processes', action='store_true', help="Set method to create dataloader child processes to fork instead of spawn (could take up more memory)")
        parser.add_argument('--find_unused_parameters', action='store_true', help="option used with DDP which allows having parameters which are not used for producing the loss. Setting it to false is more efficient, if this option is not needeed")
        ### CHECKPOINTING STUFF
        parser.add_argument('--checkpoints_dir', default="training/experiments", type=str, help='checkpoint folder')
        parser.add_argument('--load_weights_only', action='store_true', help='if specified, we load the model weights from the last checkpoint for the specified experiment, WITHOUT loading the optimizer parameters! (allows to continue traning while changing the optimizer)')
        parser.add_argument('--no_load_hparams', action='store_true', help='if specified, we dont load the saved experiment hparams when doing continue_train')
        parser.add_argument('--ignore_in_state_dict', type=str, default="", help="substring to match in state dict, to then ignore the corresponding saved weights. Sometimes useful for models where only some part was trained e.g.")
        parser.add_argument('--only_load_in_state_dict', type=str, default="", help="substring to match in state dict, to then only load the corresponding saved weights. Sometimes useful for models where only some part was trained e.g.")
        # parser.add_argument('--override_optimizers', action='store_true', help='if specified, we will use the optimizer parameters set by the hparams, even if we are continuing from checkpoint')
        # maybe could override optimizer using this? https://github.com/PyTorchLightning/pytorch-lightning/issues/3095 but need to know the epoch at which to change it

        self.parser = parser
        self.is_train = None
        self.extra_hparams = ["is_train"]
        self.opt = None

    def gather_options(self, parse_args=None):
        # get the basic options
        if parse_args is not None:
            opt, _ = self.parser.parse_known_args(parse_args)
        else:
            opt, _ = self.parser.parse_known_args()

        defaults = vars(self.parser.parse_args([]))

        if opt.continue_train and not opt.no_load_hparams:
            logs_path = opt.checkpoints_dir+"/"+opt.experiment_name
            try:
                latest_checkpoint_path = get_latest_checkpoint_path(logs_path)
            except FileNotFoundError:
                print("checkpoint file not found. Probably trying continue_train on an experiment with no checkpoints")
                raise
            hparams_file = latest_checkpoint_path+"/hparams.yaml"
            print("Loading hparams file ",hparams_file)
        else:
            hparams_file = opt.hparams_file

        if opt.hparams_file is not None:
            if hparams_file.endswith(".json"):
                hparams_json = json.loads(jsmin(open(hparams_file).read()))
            elif hparams_file.endswith(".yaml"):
                hparams_json = yaml.load(open(hparams_file))
            hparams_json2 = {k:v for k,v in hparams_json.items() if (v != False and k in defaults)}
            self.parser.set_defaults(**hparams_json2)

        if parse_args is not None:
            opt, _ = self.parser.parse_known_args(parse_args)
        else:
            opt, _ = self.parser.parse_known_args()

        # load task module and task-specific options
        # task_name = opt.task
        # task_options = importlib.import_module("{}.options.task_options".format(task_name))  # must be defined in each task folder
        # self.parser = argparse.ArgumentParser(parents=[self.parser, task_options.TaskOptions().parser])
        # if parse_args is not None:
        #     opt, _ = self.parser.parse_known_args(parse_args)
        # else:
        #     opt, _ = self.parser.parse_known_args()

        # modify model-related parser options
        model_name = opt.model
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(self.parser, opt)
        if parse_args is not None:
            opt, _ = parser.parse_known_args(parse_args)  # parse again with the new defaults
        else:
            opt, _ = self.parser.parse_known_args()

        # modify dataset-related parser options
        dataset_name = opt.dataset_name
        print(dataset_name)
        dataset_option_setter = data.get_option_setter(dataset_name)
        parser = dataset_option_setter(parser, self.is_train)

        #add negation flags
        defaults = vars(parser.parse_args([]))
        # import pdb;pdb.set_trace()
        for key,val in defaults.items():
            if val == False:
                parser.add_argument("--no-"+key, dest=key, action="store_false")

        if hparams_file is not None:
            hparams_json2 = {}
            for k,v in hparams_json.items():
                if k in defaults or k in self.extra_hparams:
                    if v!= False:
                        hparams_json2[k] = v
                else:
                    raise Exception("Hparam "+k+" not recognized!")
            parser.set_defaults(**hparams_json2)

        self.parser = parser
        if parse_args is not None:
            return parser.parse_args(parse_args)
        else:
            return parser.parse_args()

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.experiment_name)
        utils.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        file_name_json = os.path.join(expr_dir, 'opt.json')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')
        with open(file_name_json, 'wt') as opt_file:
            opt_file.write(json.dumps(vars(opt)))

    def parse(self, parse_args=None):

        opt = self.gather_options(parse_args=parse_args)
        opt.is_train = self.is_train   # train or test

        # check options:
        # if opt.loss_weight:
        #     opt.loss_weight = [float(w) for w in opt.loss_weight.split(',')]
        #     if len(opt.loss_weight) != opt.num_class:
        #         raise ValueError("Given {} weights, when {} classes are expected".format(
        #             len(opt.loss_weight), opt.num_class))
        #     else:
        #         opt.loss_weight = torch.tensor(opt.loss_weight)

        opt = {k:v for (k,v) in vars(opt).items() if not callable(v)}
        opt = Namespace(**opt)

        self.print_options(opt)
        # set gpu ids
        # str_ids = opt.gpu_ids.split(',')
        # opt.gpu_ids = []
        # for str_id in str_ids:
        #     id = int(str_id)
        #     if id >= 0:
        #         opt.gpu_ids.append(id)
        # if len(opt.gpu_ids) > 0:
        #     torch.cuda.set_device(opt.gpu_ids[0])
        #
        # set multiprocessing
        #if opt.workers > 0 and not opt.fork_processes:
        #    mp.set_start_method('spawn', force=True)
        #mp.set_start_method('spawn', force=True)

        self.opt = opt
        return self.opt
