import os
import glob

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_latest_checkpoint(logs_path):
    def get_checkpoint_files(dir):
        checkpoint_path=logs_path+"/"+dir+"/checkpoints/"
        if not os.path.isdir(checkpoint_path):
            return []
        list_of_files = glob.glob(checkpoint_path+'/*') # * means all if need specific format then *.csv
        return list_of_files

    checkpoint_subdirs = [(d,int(d.split("_")[1])) for d in os.listdir(logs_path) if (os.path.isdir(logs_path+"/"+d) and len(get_checkpoint_files(d))>0)]
    checkpoint_subdirs = sorted(checkpoint_subdirs, key=lambda t: t[1])
    list_of_files = get_checkpoint_files(checkpoint_subdirs[-1][0])
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file

def get_latest_checkpoint_path(logs_path):
    def get_checkpoint_files(dir):
        checkpoint_path=logs_path+"/"+dir+"/checkpoints/"
        if not os.path.isdir(checkpoint_path):
            return []
        list_of_files = glob.glob(checkpoint_path+'/*') # * means all if need specific format then *.csv
        return list_of_files

    checkpoint_subdirs = [(d,int(d.split("_")[1])) for d in os.listdir(logs_path) if (os.path.isdir(logs_path+"/"+d) and len(get_checkpoint_files(d))>0)]
    checkpoint_subdirs = sorted(checkpoint_subdirs, key=lambda t: t[1])
    return logs_path+"/"+checkpoint_subdirs[-1][0]
