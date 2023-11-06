import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage import img_as_float, img_as_ubyte
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr

# for latest scikit-image
# from skimage.metrics import structural_similarity as ssim
# from skimage.metrics import peak_signal_noise_ratio as psnr

import matplotlib.pyplot as plt
import os
import glob
import imageio
import torch
import sys
import fire

#TODO scikit-image 0.16.2
# python 3.6.10

from tqdm import tqdm

# sys.path.append('PerceptualSimilarity')
from PerceptualSimilarity.models import dist_model as dm

# tgt_dir = '/f_data/G/i2v/sdm/SDM_S512_novel_view_AR_V1_evalRealEstate/EP130_eval_tgt_10th/gt_10th/'
# tgt_dir = '/f_data/G/i2v/sdm/SDM_S512_novel_view_AR_V1_evalRealEstate/EP130_eval_tgt_5th/gt_5th/'
# tgt_dir = '/f_data/G/i2v/sdm/SDM_S512_novel_view_AR_V1_evalMC/EP130_eval_tgt_3rd/gt_5th'
# tgt_dir = '/f_data/G/i2v/sdm/SDM_S512_novel_view_AR_V1_evalMC/EP130_eval_tgt_5th/gt_10th'

# our_dir = '/f_data/G/i2v/sdm/SDM_S512_novel_view_AR_V1_evalRealEstate/Fix_EPLast_caption_tgt_5th/pred_10th/'

def main(
    tgt_dir='/f_data/G/i2v/sdm/SDM_S512_novel_view_AR_V1_evalRealEstate/EP130_eval_tgt_10th/gt_10th/',
    our_dir='/f_data/G/SceneScape/RE10k_eval_10th_2023_C0/',
):

    multiple_number = 1
    filenames = sorted(glob.glob(os.path.join(our_dir, '*_blend.png')))
    filenames_gt = sorted(glob.glob(os.path.join(tgt_dir, '*.png')))
    ssim_dsts = dict()
    ssim_baselines = dict()
    psnr_dsts = dict()
    psnr_baselines = dict()
    lpips_baselines = dict()
    lpips_dsts = dict()
    model = dm.DistModel()
    model.initialize(model='net-lin',net='alex',use_gpu=True,version='0.1')
    tgt_shape = (256, 256) #TODO if 256?
    print(f'tgt and pred imgs are resize to {tgt_shape}')

    perf = dict()
    total_number = 0
    skip_id = 0

    print(f'detect {len(filenames)} images in {our_dir}')
    print(f'detect {len(filenames_gt)} images in {tgt_dir}')


    for filename in tqdm(filenames):
        filename = os.path.basename(filename)
        # tgt_img = imageio.imread(os.path.join(tgt_dir, filename))[..., :3]
        tgt_img = imageio.imread(os.path.join(tgt_dir, filename.replace("_blend", "")))[..., :3]
        tgt_img = img_as_float(tgt_img)

        tgt_img = cv2.resize(tgt_img, tgt_shape, interpolation=cv2.INTER_AREA)

        continue_flag = False
        for source_dir in [our_dir]:
            replace_str = ""# if "our" in source_dir else "_pred"
            real_name = filename.replace(replace_str, "")
            if os.path.exists(os.path.join(source_dir, real_name)) is False:
                continue_flag = True
        if continue_flag is True:
            print("Continue")
            skip_id = skip_id + 1
            continue
        for source_dir in [our_dir]:
            # src_name = os.path.basename(os.path.normpath(source_dir)).split('a4_quant_')[1].split('_')[0]
            src_name = 'pred'
            if perf.get(src_name) is None:
                perf[src_name] = dict(NAME=[], SSIM=[], PSNR=[], LPIPS=[])
            replace_str = ""# if "our" in source_dir else "_pred"
            real_name = filename.replace(replace_str, "")
            img = imageio.imread(os.path.join(source_dir, real_name))[..., :3]
            img = img_as_float(img)

            img = cv2.resize(img, tgt_shape, interpolation=cv2.INTER_AREA)

            # print(f'tgt img {tgt_img.shape}')
            # print(f'img {img.shape}')

            cur_ssim = ssim(tgt_img, img, multichannel=True, data_range=1)
            cur_psnr = psnr(tgt_img, img, data_range=1)
            cur_lpips = float(model.forward((torch.FloatTensor(tgt_img).permute(2,0,1)[None, ...] * 2. - 1.), 
                                            (torch.FloatTensor(img).permute(2,0,1)[None, ...] * 2. -1.)))
            perf[src_name]['NAME'].append(real_name)
            perf[src_name]['SSIM'].append(cur_ssim)
            perf[src_name]['PSNR'].append(cur_psnr)
            perf[src_name]['LPIPS'].append(cur_lpips)
        total_number += 1    
    ks = [*perf.keys()]
    for k in ks:
        met_ks = [*perf[k].keys()]
        for met_k in met_ks:
            if met_k != 'NAME':
                perf[k]['MEAN_' + met_k] = np.mean(np.array(perf[k][met_k]))

    out_string = "\t"
    for k in sorted(perf.keys()):
        for met_k in sorted(perf[k].keys()):
            if 'MEAN' in met_k:
                out_string += met_k.replace("MEAN", "") + '\t'
        print(out_string)
        break
    for k in sorted(perf.keys()):
        met_kid = 0
        out_string = "%s: \t" % k
        for met_k in sorted(perf[k].keys()):
            if 'MEAN' in met_k:
                out_string += "%.5f, \t" % perf[k][met_k]
        print(out_string)
    print("Total Samples = ", total_number)

if __name__ == "__main__":
    fire.Fire(main)