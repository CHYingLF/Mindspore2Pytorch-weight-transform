import torch
import mindspore 
from mindspore import load_checkpoint


def traversal_params(pth_file_path, ckpt_file_path):
    """This function loop through the parameters of .pth and .ckpt weight file,
       and save the parameter name into .txt file in order.
    """
    # load pytorch pth file as a dictionary
    torch_params_dict = torch.load(pth_file_path)['model']
    # traversal a params dictionary
    file1 = open('pth.txt', 'w')
    for k, v in torch_params_dict.items():
        #print("param_key: ", k)
        #print("param_value: ", v.size())

        file1.write(k)
        file1.write('\n')

    file1.close()

    # load mindspore ckpt file as a dictionary
    file2 = open('ckpt.txt', 'w')
    mind_params_dict = load_checkpoint(ckpt_file_path)
    for k, v in mind_params_dict.items():
        #print("param_key: ", k)
        #print("param_value: ", v)
        file2.write(k)
        file2.write('\n')

    file2.close()
    print('saved parameter name to ckpt.txt and pth.txt file')

ckpt_file_path = '../yolos-0_340.ckpt'
pth_file_path = './yolos_ti.pth'
traversal_params(pth_file_path, ckpt_file_path)
