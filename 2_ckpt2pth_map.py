import torch
import mindspore
from mindspore import load_checkpoint
import json

def ckpt_2_pth(ckpt_file_path, output_path, map_dict, dump):
    '''Map the parameters' name in mindspore(.ckpt) format to pytorch(.pth) format.
       Idea is from mindspore, map the name that has a counterpart in pytorch, dump the name that not exit in pytorch
       User may need to tailor the code based on different situation, or edit the the final .json file directly.

    Params:
        ckpt_file_path: ckpt file path
        pth_file_path: path to save the output file
        map_dict: mapping between ckpt to pth
        dump: name that in ckpt but not in pth

    Return:
        a parameter name mapping between mindspore and pytorch
    '''
    print('Read ckpt file:', ckpt_file_path)
    mind_params_dict = load_checkpoint(ckpt_file_path)
    ckpt_in_pth_name = {}

    # loop through mindspore parameters in order 
    for k, v in mind_params_dict.items():
        temp = []
        for t in list(k.split('.'))[1:]:
            if t in dump: continue
            if t in map_dict.keys():
                temp.append(map_dict[t])
            else:
                temp.append(t)
        
        ckpt_in_pth_name[k]='.'.join(temp)

    # save to json file
    with open(pth_file_path, 'w') as fp:
        json.dump(ckpt_in_pth_name, fp)
   
    print('ckpt to pth mapping:', pth_file_path)

# file path
ckpt_file_path = '../yolos-0_340.ckpt'
pth_file_path = './ckpt2pth_map.json'

# create mapping for mindspore to pytorch
map_dict = {'gamma':'weight', 'beta':'bias','det_token':'dist_token'}
for i in range(10): map_dict[str(i)] = str(i)
dump = ['optimizer']

# create mapping
ckpt_2_pth(ckpt_file_path, pth_file_path, map_dict, dump)


