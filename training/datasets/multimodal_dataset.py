from pathlib import Path
from itertools import tee
import numpy as np
import torch
from .base_dataset import BaseDataset

def find_example_idx(n, cum_sums, idx = 0):
    N = len(cum_sums)
    search_i = N//2 - 1
    if N > 1:
        if n < cum_sums[search_i]:
            return find_example_idx(n, cum_sums[:search_i+1], idx=idx)
        else:
            return find_example_idx(n, cum_sums[search_i+1:], idx=idx+search_i+1)
    else:
        if n < cum_sums[0]:
            return idx
        else:
            return idx + 1


class MultimodalDataset(BaseDataset):

    def __init__(self, opt, split="train"):
        super().__init__()
        self.opt = opt
        data_path = Path(opt.data_dir)
        if not data_path.is_dir():
            raise ValueError('Invalid directory:'+opt.data_dir)

        print(opt.base_filenames_file)
        if split == "train":
            temp_base_filenames = [x[:-1] for x in open(data_path.joinpath(opt.base_filenames_file), "r").readlines()]
        else:
            temp_base_filenames = [x[:-1] for x in open(data_path.joinpath("base_filenames_"+split+".txt"), "r").readlines()]
        if opt.num_train_samples > 0:
            temp_base_filenames = np.random.choice(temp_base_filenames, size=opt.num_train_samples, replace=False)
        self.base_filenames = []

        input_mods = self.opt.input_modalities.split(",")
        output_mods = self.opt.output_modalities.split(",")
        self.input_lengths = input_lengths = [int(x) for x in str(self.opt.input_lengths).split(",")]
        self.output_lengths = output_lengths = [int(x) for x in str(self.opt.output_lengths).split(",")]
        self.output_time_offsets = output_time_offsets = [int(x) for x in str(self.opt.output_time_offsets).split(",")]
        self.input_time_offsets = input_time_offsets = [int(x) for x in str(self.opt.input_time_offsets).split(",")]

        if self.opt.input_types is None:
            input_types = ["c" for inp in input_mods]
        else:
            input_types = self.opt.input_types.split(",")

        if self.opt.input_fix_length_types is None:
            input_fix_length_types = ["end" for inp in input_mods]
        else:
            input_fix_length_types = self.opt.input_fix_length_types.split(",")

        if self.opt.output_fix_length_types is None:
            output_fix_length_types = ["end" for inp in input_mods]
        else:
            output_fix_length_types = self.opt.output_fix_length_types.split(",")

        fix_length_types_dict = {mod:output_fix_length_types[i] for i,mod in enumerate(output_mods)}
        fix_length_types_dict.update({mod:input_fix_length_types[i] for i,mod in enumerate(input_mods)})

        assert len(input_types) == len(input_mods)
        assert len(input_fix_length_types) == len(input_mods)
        assert len(output_fix_length_types) == len(input_mods)
        self.input_types = input_types
        self.input_fix_length_types = input_fix_length_types
        self.output_fix_length_types = output_fix_length_types

        if self.opt.input_num_tokens is None:
            self.input_num_tokens = [0 for inp in input_mods]
        else:
            self.input_num_tokens  = [int(x) for x in self.opt.input_num_tokens.split(",")]

        if self.opt.output_num_tokens is None:
            self.output_num_tokens = [0 for inp in output_mods]
        else:
            self.output_num_tokens  = [int(x) for x in self.opt.output_num_tokens.split(",")]

        if len(output_time_offsets) < len(output_mods):
            if len(output_time_offsets) == 1:
                self.output_time_offsets = output_time_offsets = output_time_offsets*len(output_mods)
            else:
                raise Exception("number of output_time_offsets doesnt match number of output_mods")

        if len(input_time_offsets) < len(input_mods):
            if len(input_time_offsets) == 1:
                self.input_time_offsets = input_time_offsets = input_time_offsets*len(input_mods)
            else:
                raise Exception("number of input_time_offsets doesnt match number of input_mods")

        self.features = {mod:{} for mod in input_mods+output_mods}
        #self.input_features = {input_mod:{} for input_mod in input_mods}
        #self.output_features = {output_mod:{} for output_mod in output_mods}
        if opt.fix_lengths:
            self.features_filenames = {mod:{} for mod in input_mods+output_mods}
            #self.input_features_filenames = {input_mod:{} for input_mod in input_mods}
            #self.output_features_filenames = {input_mod:{} for input_mod in input_mods}

        min_length = max(max(np.array(input_lengths) + np.array(input_time_offsets)), max(np.array(output_time_offsets) + np.array(output_lengths)) ) - min(0,min(output_time_offsets))
        print(min_length)

        fix_lengths = opt.fix_lengths

        self.total_frames = 0
        self.frame_cum_sums = []

        #Get the list of files containing features (in numpy format for now), and populate the dictionaries of input and output features (separated by modality)
        for base_filename in temp_base_filenames:
            file_too_short = False
            first_length=True
            for i, mod in enumerate(input_mods):
                feature_file = data_path.joinpath(base_filename+"."+mod+".npy")
                if self.input_fix_length_types[i] == "single": continue
                #print(feature_file)
                try:
                    features = np.load(feature_file)
                    length = features.shape[0]
                    #print(features.shape)
                    #print(length)
                    if not fix_lengths:
                        if first_length:
                            length_0 = length
                            first_length=False
                        else:
                            assert length == length_0
                    if length < min_length:
                        # print("Smol sequence "+base_filename+"."+mod+"; ignoring..")
                        file_too_short = True
                        break
                except FileNotFoundError:
                    raise Exception("An unprocessed input feature found "+base_filename+"."+mod+"; need to run preprocessing script before starting to train with them")

            if file_too_short: continue

            first_length=True
            for i, mod in enumerate(output_mods):
                feature_file = data_path.joinpath(base_filename+"."+mod+".npy")
                if self.output_fix_length_types[i] == "single": continue
                try:
                    features = np.load(feature_file)
                    length = features.shape[0]
                    if not fix_lengths:
                        if first_length:
                            length_0 = length
                            first_length=False
                        else:
                            assert length == length_0
                    if length < min_length:
                        # print("Smol sequence "+base_filename+"."+mod+"; ignoring..")
                        file_too_short = True
                        break
                except FileNotFoundError:
                    raise Exception("An unprocessed output feature found "+base_filename+"."+mod+"; need to run preprocessing script before starting to train with them")

            if file_too_short: continue

            for mod in input_mods+output_mods:
                feature_file = data_path.joinpath(base_filename+"."+mod+".npy")
                features = np.load(feature_file)
                self.features[mod][base_filename] = features
                if fix_lengths:
                    self.features_filenames[mod][base_filename] = feature_file

            if fix_lengths:
                shortest_length = 99999999999
                first_match = True
                for mod in input_mods+output_mods:
                    if fix_length_types_dict[mod] == "single": continue
                    length = self.features[mod][base_filename].shape[0]
                    if length < shortest_length:
                        #print(np.abs(length-shortest_length))
                        if first_match:
                            first_match = False
                        else:
                            if np.abs(length-shortest_length) > 2:
                                print("sequence length difference")
                                print(np.abs(length-shortest_length))
                                print(base_filename)
                            #assert np.abs(length-shortest_length) <= 2
                        shortest_length = length
                for i,mod in enumerate(input_mods):
                    if self.input_fix_length_types[i] == "end":
                        np.save(self.features_filenames[mod][base_filename],self.features[mod][base_filename][:shortest_length])
                    elif self.input_fix_length_types[i] == "beg":
                        np.save(self.features_filenames[mod][base_filename],self.features[mod][base_filename][shortest_length:])
                    elif self.input_fix_length_types[i] == "single":
                        assert self.features[mod][base_filename].shape[0] == 1
                    else:
                        raise NotImplementedError("Haven't implemented input_fix_length_type "+self.input_fix_length_type[i])

                for i,mod in enumerate(output_mods):
                    if mod not in input_mods:
                        if self.output_fix_length_types[i] == "end":
                            np.save(self.features_filenames[mod][base_filename],self.features[mod][base_filename][:shortest_length])
                        elif self.output_fix_length_types[i] == "beg":
                            np.save(self.features_filenames[mod][base_filename],self.features[mod][base_filename][shortest_length:])
                        elif self.output_fix_length_types[i] == "single":
                            assert self.features[mod][base_filename].shape[0] == 1
                        else:
                            raise NotImplementedError("Haven't implemented output_fix_length_type "+self.output_fix_length_type[i])

                for mod in input_mods+output_mods:
                    self.features[mod][base_filename] = np.load(self.features_filenames[mod][base_filename])
                    length = self.features[mod][base_filename].shape[0]
                    if i == 0:
                        length_0 = length
                    else:
                        assert length == length_0

            #TODO: implement this!
            ## we pad the song features with zeros to imitate during training what happens during generation
            #x = [np.concatenate((np.zeros(( xx.shape[0],max(0,max(output_time_offsets)) )),xx),0) for xx in x]
            ## we also pad at the end to allow generation to be of the same length of sequence, by padding an amount corresponding to time_offset
            #x = [np.concatenate((xx,np.zeros(( xx.shape[0],max(0,max(input_lengths)+max(input_time_offsets)-(min(output_time_offsets)+min(output_lengths)-1)) ))),0) for xx in x]


            found_full_seq = False
            for i,mod in enumerate(input_mods):
                if self.input_fix_length_types[i] != "single":
                    sequence_length = self.features[mod][base_filename].shape[0]
                    found_full_seq = True
            if not found_full_seq:
                sequence_length = 1
            possible_init_frames = sequence_length-max(max(input_lengths)+max(input_time_offsets),max(output_time_offsets)+max(output_lengths))+1
            self.total_frames += possible_init_frames
            self.frame_cum_sums.append(self.total_frames)

            self.base_filenames.append(base_filename)

        print("sequences added: "+str(len(self.base_filenames)))
        assert len(self.base_filenames)>0, "List of files for training cannot be empty"
        for mod in input_mods+output_mods:
            assert len(self.features[mod].values()) == len(self.base_filenames)

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--sampling_rate', default=44100, type=float)
        parser.add_argument('--dins', default=None, help="input dimension for continuous inputs. Embedding dimension for discrete inputs")
        parser.add_argument('--douts', default=None)
        parser.add_argument('--input_modalities', default='mp3_mel_100')
        parser.add_argument('--output_modalities', default='mp3_mel_100')
        parser.add_argument('--input_lengths', help='input sequence length')
        parser.add_argument('--input_num_tokens', help='num_tokens. use 0 for continuous inputs')
        parser.add_argument('--output_num_tokens', help='num_tokens. use 0 for continuous inputs')
        parser.add_argument('--input_types', default=None, help='Comma-separated list of input types: d for discrete, c for continuous. E.g. d,c,c. Assumes continuous if not specified')
        parser.add_argument('--input_fix_length_types', default=None, help='Comma-separated list of approaches to fix length: end for cut end, beg for cut beginning, single for single-element sequence (e.g. sequence-level label). E.g. single,end,end. Assumes cut end if not specified')
        parser.add_argument('--output_fix_length_types', default=None, help='Comma-separated list of approaches to fix length: end for cut end, beg for cut beginning, single for single-element sequence (e.g. sequence-level label). E.g. single,end,end. Assumes cut end if not specified')
        parser.add_argument('--output_lengths', help='output sequence length')
        parser.add_argument('--output_time_offsets', default="1", help='time shift between the last read input, and the output predicted. The default value of 1 corresponds to predicting the next output')
        parser.add_argument('--input_time_offsets', default="0", help='time shift between the beginning of each modality and the first modality')
        parser.add_argument('--max_token_seq_len', type=int, default=1024)
        parser.add_argument('--fix_lengths', action='store_true', help='fix unmatching length of sequences')
        parser.add_argument('--num_train_samples', type=int, default=0, help='if 0 then use all of them')

        return parser

    def name(self):
        return "MultiModalDataset"

    def process_input(self,j,xx,index):
        input_lengths = self.input_lengths
        output_lengths = self.output_lengths
        output_time_offsets = self.output_time_offsets
        input_time_offsets = self.input_time_offsets
        if self.input_fix_length_types[j]!="single":
            return torch.tensor(xx[index+input_time_offsets[j]:index+input_time_offsets[j]+input_lengths[j]]).float()
        else:
            return torch.tensor(xx).long().unsqueeze(1)

    def process_output(self,j,yy,index):
        input_lengths = self.input_lengths
        output_lengths = self.output_lengths
        output_time_offsets = self.output_time_offsets
        input_time_offsets = self.input_time_offsets
        if self.output_fix_length_types[j]!="single":
            return torch.tensor(yy[index+output_time_offsets[j]:index+output_time_offsets[j]+output_lengths[j]]).float()
        else:
            return torch.tensor(yy).long().unsqueeze(1)

    def __getitem__(self, item):
        idx = find_example_idx(item, self.frame_cum_sums)
        base_filename = self.base_filenames[idx]

        input_lengths = self.input_lengths
        output_lengths = self.output_lengths
        output_time_offsets = self.output_time_offsets
        input_time_offsets = self.input_time_offsets

        input_mods = self.opt.input_modalities.split(",")
        output_mods = self.opt.output_modalities.split(",")

        x = [self.features[mod][base_filename] for mod in input_mods]
        y = [self.features[mod][base_filename] for mod in output_mods]
        #for i, mod in enumerate(input_mods):
        #    input_feature = self.features[mod][base_filename]
        #    x.append(input_feature)

        #for i, mod in enumerate(output_mods):
        #    output_feature = self.features[mod][base_filename]
        #    y.append(output_feature)

        # normalization of individual features for the sequence
        # not doing this any more as we are normalizing over all examples now
        #x = [(xx-np.mean(xx,0,keepdims=True))/(np.std(xx,0,keepdims=True)+1e-5) for xx in x]
        #y = [(yy-np.mean(yy,0,keepdims=True))/(np.std(yy,0,keepdims=True)+1e-5) for yy in y]

        if idx > 0: index = item - self.frame_cum_sums[idx-1]
        else: index = item

        ## CONSTRUCT TENSOR OF INPUT FEATURES ##
        input_windows = [self.process_input(j,xx,index) for j,xx in enumerate(x)]

        ## CONSTRUCT TENSOR OF OUTPUT FEATURES ##
        output_windows = [self.process_output(j,yy,index) for j,yy in enumerate(y)]

        # print(input_windows[i])
        return_dict = {}
        for i,mod in enumerate(input_mods):
            return_dict["in_"+mod] = input_windows[i]
        for i,mod in enumerate(output_mods):
            return_dict["out_"+mod] = output_windows[i]

        return return_dict

    def __len__(self):
        # return len(self.base_filenames)
        return self.total_frames
        # return 2


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)
