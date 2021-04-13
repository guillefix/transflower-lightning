import importlib
from torch.utils.data import DataLoader
from .base_dataset import BaseDataset
import numpy as np
import torch

def find_dataset_using_name(dataset_name):
    # Given the option --dataset_name [datasetname],
    # the file "data/datasetname_dataset.py"
    # will be imported.

    task_name = "training"
    task_module = importlib.import_module(task_name)
    dataset_filename = task_name + ".datasets." + dataset_name + "_dataset"
    # datasetlib = importlib.import_module(dataset_filename, package=task_module)
    datasetlib = importlib.import_module(dataset_filename, package=task_module)

    # In the file, the class called DatasetNameDataset() will
    # be instantiated. It has to be a subclass of BaseDataset,
    # and it is case-insensitive.

    def is_subclass(subclass, superclass):
        return next(iter(subclass.__bases__)).__module__.endswith(superclass.__module__)

    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower():
            if is_subclass(cls, BaseDataset) or any(is_subclass(cls_b, BaseDataset) for cls_b in cls.__bases__):
                dataset = cls

    if dataset is None:
        raise NotImplementedError("In {}.py, there should be a subclass of BaseDataset with class name that matches {} in lowercase.".format(
            dataset_filename, target_dataset_name))

    return dataset


def get_option_setter(dataset_name):
    dataset_class = find_dataset_using_name(dataset_name)
    return dataset_class.modify_commandline_options


def create_dataset(opt, split="train", *args, **kwargs):
    dataset = find_dataset_using_name(opt.dataset_name)
    instance = dataset(opt, split, *args,**kwargs)
    print('dataset [{}] was created {}'.format(instance.name(), "(val)" if split=="val" else ''))
    return instance

def paired_collate_fn(insts,tgt_dim=2):
    src_insts= list(map(lambda x: x['input'],insts))
    tgt_insts = list(map(lambda x: x['target'],insts))
    src_insts = collate_fn(src_insts,dim=2)
    tgt_insts = collate_fn(tgt_insts,dim=tgt_dim)
    return {'input':src_insts, 'target':tgt_insts}

def collate_fn(insts,dim=-1): #dim is time dim
    ''' Pad the instance to the max seq length in batch '''

    max_len = max(inst.shape[dim] for inst in insts)

    # print(max_len)
    batch_seq = [
        torch.cat([inst.long(),torch.full(inst.shape[:dim]+((max_len - inst.shape[dim]),)+inst.shape[dim+1:],PAD_STATE).long()],dim)
        for inst in insts]

    batch_pos = np.array([
        [pos_i+1 for pos_i in range(inst.shape[dim])] + [PAD_STATE]*(max_len - inst.shape[dim]) for inst in insts])

    batch_seq = torch.stack(batch_seq)
    # print(batch_seq)
    batch_pos = torch.LongTensor(batch_pos)

    return batch_seq, batch_pos

def transformer_paired_collate_fn(insts):
    return paired_collate_fn(insts,tgt_dim=2)

def wavenet_paired_collate_fn(insts):
    return paired_collate_fn(insts,tgt_dim=1)

def meta_collate_fn(pad_batches, model):
    if pad_batches:
        if model == "transformer":
            return transformer_paired_collate_fn
        else:
            return wavenet_paired_collate_fn
    else:
        return default_collate

from torch.utils.data.dataloader import default_collate
def create_dataloader(dataset, split="train"):
    is_eval = (split == "val" or split == "test")
    return DataLoader(dataset,
                      batch_size=dataset.opt.batch_size if not is_eval else dataset.opt.val_batch_size,
                      shuffle=not is_eval,
                      # collate_fn=meta_collate_fn(dataset.opt.pad_batches,dataset.opt.model),
                      collate_fn=None,
                      #pin_memory=True,
                      drop_last=True,
                      num_workers=dataset.opt.workers)
