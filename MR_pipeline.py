import copy
import gc
import random
from argparse import ArgumentParser
from pathlib import Path
import os
import json_lines

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.cuda.amp import GradScaler
from torchvision.transforms import ToPILImage
from tqdm import tqdm

from models.dp_pipeline import WarpInpaintModel
from util.finetune_utils import finetune_depth_model, finetune_decoder
from util.general_utils import apply_depth_colormap, save_video


def evaluate(model):
    fps = model.config["save_fps"]
    save_root = Path(model.run_dir)
    # save_dict = {
    #     "images": torch.cat(model.images, dim=0),
    #     "images_orig_decoder": torch.cat(model.images_orig_decoder, dim=0),
    #     "masks": torch.cat(model.masks, dim=0),
    #     "disparities": torch.cat(model.disparities, dim=0),
    #     "depths": torch.cat(model.depths, dim=0),
    #     "cameras": model.cameras_extrinsics,
    # }
    # torch.save(save_dict, save_root / "results.pt")
    # if not model.config["use_splatting"]:
        # model.save_mesh("full_mesh")

    # video = (255 * torch.cat(model.images, dim=0)).to(torch.uint8).detach().cpu()
    # video_reverse = (255 * torch.cat(model.images[::-1], dim=0)).to(torch.uint8).detach().cpu()

    # save_video(video, save_root / "output.mp4", fps=fps)
    # save_video(video_reverse, save_root / "output_reverse.mp4", fps=fps)


def evaluate_epoch(model, epoch):
    # disparity = model.disparities[epoch]
    # disparity_colored = apply_depth_colormap(disparity[0].permute(1, 2, 0))
    # disparity_colored = disparity_colored.clone().permute(2, 0, 1).unsqueeze(0).float()
    # save_root = Path(model.run_dir) / "images"
    # save_root.mkdir(exist_ok=True, parents=True)
    # (save_root / "frames").mkdir(exist_ok=True, parents=True)
    # (save_root / "images_orig_decoder").mkdir(exist_ok=True, parents=True)
    # (save_root / "masks").mkdir(exist_ok=True, parents=True)
    # (save_root / "warped_images").mkdir(exist_ok=True, parents=True)
    # (save_root / "disparities").mkdir(exist_ok=True, parents=True)

    # ToPILImage()(model.images[epoch][0]).save(save_root / "frames" / f"{epoch}.png")
    # ToPILImage()(model.images_orig_decoder[epoch][0]).save(save_root / "images_orig_decoder" / f"{epoch}.png")
    # ToPILImage()(model.masks[epoch][0]).save(save_root / "masks" / f"{epoch}.png")
    # ToPILImage()(model.warped_images[epoch][0]).save(save_root / "warped_images" / f"{epoch}.png")
    # ToPILImage()(disparity_colored[0]).save(save_root / "disparities" / f"{epoch}.png")

    # save only epoch 1
    save_path = os.path.join(model.run_dir, model.video_name + '_' + model.file_name)
    ToPILImage()(model.images[1][0]).save(save_path)

    # if epoch == 0:
    #     with open(Path(model.run_dir) / "config.yaml", "w") as f:
    #         OmegaConf.save(model.config, f)


def run(config, image_path, prompt, intrinsics, extrinsics, videoname, file_name):
    seed = config["seed"]
    if seed == -1:
        seed = np.random.randint(2 ** 32)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    print(f"running with seed: {seed}.")
    model = WarpInpaintModel(config, image_path=image_path, prompt=prompt, extrinsics=extrinsics, intrinsics=intrinsics).to(config["device"])
    # evaluate_epoch(model, 0)
    scaler = GradScaler(enabled=config["enable_mix_precision"])
    for epoch in tqdm(range(1, config["frames"] + 1)):
        if config["use_splatting"]:
            warp_output = model.warp_splatting(epoch)
        else:
            warp_output = model.warp_mesh(epoch)

        print('data type of warped image: ', warp_output["warped_image"].dtype, warp_output["inpaint_mask"].dtype)
        # size
        print('data size', warp_output["warped_image"].shape, warp_output["inpaint_mask"].shape)
        inpaint_output = model.inpaint(warp_output["warped_image"], warp_output["inpaint_mask"])

        if config["finetune_decoder"]:
            finetune_decoder(config, model, warp_output, inpaint_output)

        model.update_images_masks(inpaint_output["latent"], warp_output["inpaint_mask"])

        if config["finetune_depth_model"]:
            # reload depth model
            del model.depth_model
            gc.collect()
            torch.cuda.empty_cache()
            model.depth_model = torch.hub.load("intel-isl/MiDaS", "DPT_Large").to(model.device)

            finetune_depth_model(config, model, warp_output, epoch, scaler)

        model.update_depth(model.images[epoch])

        if not config["use_splatting"]:
            # update mesh with the correct mask
            if config["mask_opening_kernel_size"] > 0:
                mesh_mask = 1 - torch.maximum(model.masks[epoch], model.masks_diffs[epoch - 1])
            else:
                mesh_mask = 1 - model.masks[epoch]
            extrinsic = model.get_extrinsics(model.current_camera)
            # model.update_mesh(model.images[epoch], model.depths[epoch], mesh_mask > 0.5, extrinsic, epoch)

        # reload decoder
        model.vae.decoder = copy.deepcopy(model.decoder_copy)

        model.images_orig_decoder.append(model.decode_latents(inpaint_output["latent"]).detach())
        evaluate_epoch(model, epoch)

        torch.cuda.empty_cache()
        gc.collect()

    evaluate(model)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "--base-config",
        default="./config/dp.yaml",
        help="Config path",
    )
    parser.add_argument(
        "--example_config",
        default="./config/example_configs/1.yaml",
        help="Config path",
    )
    args = parser.parse_args()
    base_config = OmegaConf.load(args.base_config)
    example_config = OmegaConf.load(args.example_config)
    config = OmegaConf.merge(base_config, example_config)

    # read files
    data_file = os.path.join('/f_data/G', 'dataset/MannequinChallenge/meta_200_random.jsonl')
    with open(data_file, 'rb') as f:
        data = [e for e in json_lines.reader(f)]
    
    root = os.path.join('/f_data/G', 'dataset/MannequinChallenge/testimages_135')

    caption_file = os.path.join('/f_data/G', 'dataset/MannequinChallenge/captions_200.jsonl')
    with open(caption_file, 'rb') as f:
        caption_data = [e for e in json_lines.reader(f)]

    for idx in tqdm(range(len(data))):

        item = data[idx]
        videoname = item['video_name']

        first_name = str(item['1st'][0])+'.png'
        second_name = str(item['3rd'][0]) +'.png'
        three_name = str(item['5th'][0]) + '.png'

        filename_ori = os.path.join(root, videoname, first_name)

        intrinsics = np.array(item['1st'][1:5]).astype(np.float32)
        src_pose = np.array(item['1st'][5:17]).astype(np.float32).reshape(3, 4)
        tgt_pose_3rd = np.array(item['3rd'][1:13]).astype(np.float32).reshape(3, 4)
        tgt_pose_5th = np.array(item['5th'][1:13]).astype(np.float32).reshape(3, 4)

        width, height = 512, 512
        k = np.array(
            [
                [intrinsics[0]*width, 0, intrinsics[2]*width],
                [0, intrinsics[1]*height, intrinsics[3]*height],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )
        intrinsicss = np.array([k,k])

        extrinsics = np.array([src_pose, tgt_pose_3rd])

        prompt = caption_data[idx][videoname].strip()

        run(config, image_path=filename_ori, prompt=prompt, intrinsics=intrinsicss, extrinsics=extrinsics, videoname=videoname, file_name=first_name)
