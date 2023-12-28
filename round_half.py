import copy
import gc
import random
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.cuda.amp import GradScaler
from torchvision.transforms import ToPILImage
from tqdm import tqdm

from models.image_round import WarpInpaintModel
from util.finetune_utils import finetune_depth_model_global as finetune_depth_model
from util.finetune_utils import finetune_decoder
from util.general_utils import apply_depth_colormap, save_video

round_images = []
warp_images = []

def evaluate(model):
    fps = model.config["save_fps"]
    save_root = Path(model.run_dir)
    save_dict = {
        "images": torch.cat(model.images, dim=0),
        "images_orig_decoder": torch.cat(model.images_orig_decoder, dim=0),
        "masks": torch.cat(model.masks, dim=0),
        "disparities": torch.cat(model.disparities, dim=0),
        "depths": torch.cat(model.depths, dim=0),
        "cameras": model.cameras_extrinsics,
    }
    # torch.save(save_dict, save_root / "results.pt")
    # if not model.config["use_splatting"]:
    #     model.save_mesh("full_mesh")

    video = (255 * torch.cat(round_images, dim=0)).to(torch.uint8).detach().cpu()
    warp_video = (255 * torch.cat(warp_images, dim=0)).to(torch.uint8).detach().cpu()
    # video_reverse = (255 * torch.cat(model.images[::-1], dim=0)).to(torch.uint8).detach().cpu()

    save_video(video, save_root / "output.mp4", fps=fps)
    save_video(warp_video, save_root / "output_warp.mp4", fps=fps)
    # save_video(video_reverse, save_root / "output_reverse.mp4", fps=fps)


def evaluate_epoch(model, epoch):
    disparity = model.disparities[epoch]
    disparity_colored = apply_depth_colormap(disparity[0].permute(1, 2, 0))
    disparity_colored = disparity_colored.clone().permute(2, 0, 1).unsqueeze(0).float()
    save_root = Path(model.run_dir) / "images"
    save_root.mkdir(exist_ok=True, parents=True)
    (save_root / "frames").mkdir(exist_ok=True, parents=True)
    (save_root / "images_orig_decoder").mkdir(exist_ok=True, parents=True)
    (save_root / "masks").mkdir(exist_ok=True, parents=True)
    (save_root / "warped_images").mkdir(exist_ok=True, parents=True)
    (save_root / "disparities").mkdir(exist_ok=True, parents=True)

    ToPILImage()(model.images[epoch][0]).save(save_root / "frames" / f"{epoch}.png")
    ToPILImage()(model.images_orig_decoder[epoch][0]).save(save_root / "images_orig_decoder" / f"{epoch}.png")
    ToPILImage()(model.masks[epoch][0]).save(save_root / "masks" / f"{epoch}.png")
    ToPILImage()(model.warped_images[epoch][0]).save(save_root / "warped_images" / f"{epoch}.png")
    ToPILImage()(disparity_colored[0]).save(save_root / "disparities" / f"{epoch}.png")

    if epoch == 0:
        with open(Path(model.run_dir) / "config.yaml", "w") as f:
            OmegaConf.save(model.config, f)


def run(config, prompt=None, image_path=None, round_reverse=False):
    seed = config["seed"]
    if seed == -1:
        seed = np.random.randint(2 ** 32)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    print(f"running with seed: {seed}.")
    model = WarpInpaintModel(config, prompt, image_path).to(config["device"])
    evaluate_epoch(model, 0)
    scaler = GradScaler(enabled=config["enable_mix_precision"])
    # left
    for epoch in tqdm(range(1, config["frames"] + 1)):
        if config["use_splatting"]:
            warp_output = model.warp_splatting(epoch)
        else:
            warp_output = model.warp_mesh(epoch, round_reverse=round_reverse)

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
            model.update_mesh(model.images[epoch], model.depths[epoch], mesh_mask > 0.5, extrinsic, epoch)

        # reload decoder
        model.vae.decoder = copy.deepcopy(model.decoder_copy)

        model.images_orig_decoder.append(model.decode_latents(inpaint_output["latent"]).detach())
        evaluate_epoch(model, epoch)

        torch.cuda.empty_cache()
        gc.collect()
    
    left_images = model.images
    for image in left_images:
        round_images.append(image)
    
    left_warp_images = model.warped_images
    for image in left_warp_images:
        warp_images.append(image)

    evaluate(model)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "--base-config",
        default="./config/fast.yaml",
        help="Config path",
    )
    parser.add_argument(
        "--example_config",
        default="./config/example_configs/nerf.yaml",
        help="Config path",
    )
    parser.add_argument(
        "--prompt",
        default="In a garden, a wooden table with a vase on it.",
        help="In a garden, a wooden table with a vase on it."
    )
    parser.add_argument(
        "--image",
        default="/mnt/workspace/SceneScape/nerf.png",
        help="image path"
    )
    parser.add_argument(
        "--round_reverse",
        default=False,
        action="store_true",
        help="round reverse"
    )
    
    args = parser.parse_args()
    base_config = OmegaConf.load(args.base_config)
    example_config = OmegaConf.load(args.example_config)
    config = OmegaConf.merge(base_config, example_config)

    run(config, args.prompt, args.image, args.round_reverse)
