## Utils to setup data to train Disnet. 

import numpy as np
import os
import json
import math

import torch
import torch.nn.functional as F

## Adapted from https://github.com/vita-epfl/monoloco/blob/6299859d95838096e81f5cb527d9e0fecace02d0/monoloco/utils/kitti.py#L9
## Processes Kitti object label data and calib data into training data for disnet
def get_calibration(path_txt):
    """Read calibration parameters from txt file:
    For the left color camera we use P2 which is K * [I|t]
    P = [fu, 0, x0, fu*t1-x0*t3
         0, fv, y0, fv*t2-y0*t3
         0, 0,  1,          t3]
    check also http://ksimek.github.io/2013/08/13/intrinsic/
    Simple case test:
    xyz = np.array([2, 3, 30, 1]).reshape(4, 1)
    xyz_2 = xyz[0:-1] + tt
    uv_temp = np.dot(kk, xyz_2)
    uv_1 = uv_temp / uv_temp[-1]
    kk_1 = np.linalg.inv(kk)
    xyz_temp2 = np.dot(kk_1, uv_1)
    xyz_new_2 = xyz_temp2 * xyz_2[2]
    xyz_fin_2 = xyz_new_2 - tt
    """

    with open(path_txt, "r") as ff:
        file = ff.readlines()
    p2_str = file[2].split()[1:]
    p2_list = [float(xx) for xx in p2_str]
    p2 = np.array(p2_list).reshape(3, 4)

    p3_str = file[3].split()[1:]
    p3_list = [float(xx) for xx in p3_str]
    p3 = np.array(p3_list).reshape(3, 4)

    kk, tt = get_translation(p2)
    kk_right, tt_right = get_translation(p3)

    return [kk, tt], [kk_right, tt_right]

def get_translation(pp):
    """Separate intrinsic matrix from translation and convert in lists"""

    kk = pp[:, :-1]
    f_x = kk[0, 0]
    f_y = kk[1, 1]
    x0, y0 = kk[2, 0:2]
    aa, bb, t3 = pp[0:3, 3]
    t1 = float((aa - x0*t3) / f_x)
    t2 = float((bb - y0*t3) / f_y)
    tt = [t1, t2, float(t3)]
    return kk.tolist(), tt

def pixel_to_camera(uv_tensor, kk, z_met):
    """
    Convert a tensor in pixel coordinate to absolute camera coordinates
    It accepts lists or torch/numpy tensors of (m, 2) or (m, x, 2)
    where x is the number of keypoints
    """
    if isinstance(uv_tensor, (list, np.ndarray)):
        uv_tensor = torch.tensor(uv_tensor)
    if isinstance(kk, list):
        kk = torch.tensor(kk)
    if uv_tensor.size()[-1] != 2:
        uv_tensor = uv_tensor.permute(0, 2, 1)  # permute to have 2 as last dim to be padded
        assert uv_tensor.size()[-1] == 2, "Tensor size not recognized"
    uv_padded = F.pad(uv_tensor, pad=(0, 1), mode="constant", value=1)  # pad only last-dim below with value 1

    kk_1 = torch.inverse(kk)
    xyz_met_norm = torch.matmul(uv_padded, kk_1.t())  # More general than torch.mm
    xyz_met = xyz_met_norm * z_met

    return xyz_met

if __name__ == '__main__':
    print("Preprocessing kitti data")
    train_size = 6000
    eval_size = 1481
    label_dir = "train/label_2/"
    calib_dir = "calib/"

    dic_data = {'train': dict(X = [], Y = []),
                'eval': dict(X = [], Y = [])
    }

    for cnt in range(train_size + eval_size):
        str_cnt = str(cnt)
        str_cnt_final = str_cnt
        for _ in range(6 - len(str_cnt)):
            str_cnt_final = "0" + str_cnt_final

        p_left, _ = get_calibration(os.path.join(calib_dir, str_cnt_final + ".txt"))
        kk = p_left[0]
        with open(os.path.join(label_dir, str_cnt_final + ".txt"), "r") as f_gt:
            for line_gt in f_gt:
                line = line_gt.split()
                if line[0].lower() == 'pedestrian': # only parse pedestrians
                    xyz = [float(x) for x in line[11:14]]
                    dd = float(math.sqrt(xyz[0] ** 2 + xyz[1] ** 2 + xyz[2] ** 2))
                    bboxes =[float(x) for x in line[4:8]]
                    xy1 = bboxes[:2]
                    xy2 = bboxes[2:]
                    xy1_norm = pixel_to_camera(xy1, kk, 1).tolist()
                    xy2_norm = pixel_to_camera(xy2, kk, 1).tolist()

                    if cnt < train_size:
                        dic_data['train']['X'].append(xy1_norm[:2] + xy2_norm[:2])
                        dic_data['train']['Y'].append([dd])
                    else:
                        dic_data['eval']['X'].append(xy1_norm[:2] + xy2_norm[:2])
                        dic_data['eval']['Y'].append([dd])

    with open('data.json', 'w') as fp:
        json.dump(dic_data, fp)