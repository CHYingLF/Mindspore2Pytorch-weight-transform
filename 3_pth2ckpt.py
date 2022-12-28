import torch
from mindspore import save_checkpoint, Tensor, load_checkpoint
import json
import numpy

# show torch tensor precision
torch.set_printoptions(precision=8)

def convert_pth_to_ckpt(pth_file_path, ckpt_fille_path, ckpt2pth_map):
    """Read the mindspore parameter name(key), find its value using the mindspore-pytorch name mapping,
       Save the key and value to mindsore checkpoint.

       Params:
           pth_file_path: pytorch weight file path
           ckpt_file_path: mindsore weight file path
           ckpt2pth_mapping: mindspore2pytorch parameters' name mapping
    """
    
    torch_params_dict = torch.load(pth_file_path)['model']
    mind_params_dict = load_checkpoint(ckpt_file_path)
    params_dict = []

    for k, v in mind_params_dict.items():
        if k == 'optimizer.pos_embed': 
            torch_key = 'backbone.pos_embed'
        elif k == 'optimizer.det_token':
            torch_key = 'backbone.det_token'
        else:
            torch_key = ckpt2pth_map[k]

        value = torch_params_dict[torch_key]
        value = value.cpu().numpy()
        value = Tensor(value)
        params_dict.append({"name":k, "data": value})
    
    # check whether the value are same for an arbitrary parameter
    print('\nmindspore:\n',params_dict[1]["data"],'\ntorch:\n', torch_params_dict['backbone.pos_embed'])
    output_path = "./yolos_tiny.ckpt"
    save_checkpoint(params_dict, output_path)
    print('saved:', output_path)
    
# file path
ckpt_file_path = '../yolos-0_340.ckpt'
pth_file_path = 'yolos_ti.pth'
ckpt2pth_map = './ckpt2pth_map.json'

# mapping
f = open(ckpt2pth_map)
ckpt2pth_map = json.load(f)

# convert
convert_pth_to_ckpt(pth_file_path, ckpt_file_path, ckpt2pth_map)

        
