import torch
import os
from pydantic import BaseModel, Field
from typing import Optional, Literal
from PIL import Image, ImageOps
import logging
import phantom_wan
from phantom_wan.configs import WAN_CONFIGS, SIZE_CONFIGS
from phantom_wan.utils.utils import cache_video, cache_image
import torch.distributed as dist
from datetime import datetime
from huggingface_hub import snapshot_download

logging.basicConfig(level=logging.INFO)


class Item(BaseModel):
    task: str = Field(default="s2v-1.3B", description="The task to run")
    size: str = Field(
        default="1280*720",
        description="The area (width*height) of the generated video",
    )
    frame_num: Optional[int] = Field(
        default=1,
        description="How many frames to sample from a image or video. The number should be 4n+1",
    )
    sample_fps: Optional[int] = Field(
        default=None, description="The fps of the generated video"
    )
    ckpt_dir: Optional[str] = Field(
        default=None, description="The path to the checkpoint directory"
    )
    phantom_ckpt: Optional[str] = Field(
        default=None, description="The path to the Phantom-Wan checkpoint"
    )
    offload_model: Optional[bool] = Field(
        default=None,
        description="Whether to offload the model to CPU after each model forward",
    )
    ulysses_size: int = Field(
        default=1, description="The size of the ulysses parallelism in DiT"
    )
    ring_size: int = Field(
        default=1,
        description="The size of the ring attention parallelism in DiT",
    )
    t5_fsdp: bool = Field(
        default=False, description="Whether to use FSDP for T5"
    )
    t5_cpu: bool = Field(
        default=False, description="Whether to place T5 model on CPU"
    )
    dit_fsdp: bool = Field(
        default=False, description="Whether to use FSDP for DiT"
    )
    save_file: Optional[str] = Field(
        default=None,
        description="The file to save the generated image or video to",
    )
    prompt: Optional[str] = Field(
        default=None,
        description="The prompt to generate the image or video from",
    )
    use_prompt_extend: bool = Field(
        default=False, description="Whether to use prompt extend"
    )
    prompt_extend_method: Literal["dashscope", "local_qwen"] = Field(
        default="local_qwen", description="The prompt extend method to use"
    )
    prompt_extend_model: Optional[str] = Field(
        default=None, description="The prompt extend model to use"
    )
    prompt_extend_target_lang: Literal["ch", "en"] = Field(
        default="ch", description="The target language of prompt extend"
    )
    base_seed: int = Field(
        default=-1,
        description="The seed to use for generating the image or video",
    )
    image: Optional[str] = Field(
        default=None, description="The image to generate the video from"
    )
    ref_image: Optional[str] = Field(
        default=None, description="The reference images used by Phantom-Wan"
    )
    sample_solver: Literal["unipc", "dpm++"] = Field(
        default="unipc", description="The solver used to sample"
    )
    sample_steps: Optional[int] = Field(
        default=50, description="The sampling steps"
    )
    sample_shift: Optional[float] = Field(
        default=5.0,
        description="Sampling shift factor for flow matching schedulers",
    )
    sample_guide_scale: float = Field(
        default=5.0, description="Classifier free guidance scale"
    )
    sample_guide_scale_img: float = Field(
        default=5.0,
        description="Classifier free guidance scale for reference images",
    )
    sample_guide_scale_text: float = Field(
        default=7.5, description="Classifier free guidance scale for text"
    )


hf_token = os.environ.get("HF_TOKEN")
logging.info(f"HF_TOKEN: {hf_token}")


snapshot_download(
    repo_id="Wan-AI/Wan2.1-T2V-1.3B",
    local_dir="/persistent-storage/Wan2.1-T2V-1.3B",
    token=hf_token,
)
snapshot_download(
    repo_id="bytedance-research/Phantom",
    local_dir="/persistent-storage/Phantom-Wan-Models",
    token=hf_token,
)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _init_logging(rank):
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )


def load_ref_images(path, size):
    # Load size.
    h, w = size[1], size[0]
    # Load images.
    ref_paths = path.split(",")
    ref_images = []
    for image_path in ref_paths:
        with Image.open(image_path) as img:
            img = img.convert("RGB")

            # Calculate the required size to keep aspect ratio and fill the rest with padding.
            img_ratio = img.width / img.height
            target_ratio = w / h

            if img_ratio > target_ratio:  # Image is wider than target
                new_width = w
                new_height = int(new_width / img_ratio)
            else:  # Image is taller than target
                new_height = h
                new_width = int(new_height * img_ratio)

            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Create a new image with the target size and place the resized image in the center
            delta_w = w - img.size[0]
            delta_h = h - img.size[1]
            padding = (
                delta_w // 2,
                delta_h // 2,
                delta_w - (delta_w // 2),
                delta_h - (delta_h // 2),
            )
            new_img = ImageOps.expand(img, padding, fill=(255, 255, 255))
            ref_images.append(new_img)
    logging.info(f"ref_images loaded {ref_images}")
    return ref_images


def run(input):
    # Setup distributed environment
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    _init_logging(rank)
    # Parse input into Item model
    item = Item(**input)
    current_path = os.getcwd()
    directories = [d for d in os.listdir(current_path) if os.path.isdir(os.path.join(current_path, d))]
    with open("/persistent-storage/emptyfile.txt", "w") as f:
        f.write("test")
        pass
    logging.info(f"Current path: {current_path}")
    logging.info("Directories:")
    for d in directories:
        logging.info(f" dir: {d}")
    logging.info("persistent Directories:")
    for d in os.listdir('/persistent-storage/'):
        logging.info(f" dir: {d}")


    device = local_rank

    # Log environment variables
    logging.info(f"RANK: {rank}")
    logging.info(f"WORLD_SIZE: {world_size}")
    logging.info(f"LOCAL_RANK: {local_rank}")

    # Handle offload_model setting
    if item.offload_model is None:
        item.offload_model = False if world_size > 1 else True
        logging.info(
            f"offload_model is not specified, set to {item.offload_model}."
        )

    # Initialize distributed process group if needed
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size,
        )
    else:
        assert not (item.t5_fsdp or item.dit_fsdp), (
            "t5_fsdp and dit_fsdp are not supported in non-distributed environments."
        )
        assert not (item.ulysses_size > 1 or item.ring_size > 1), (
            "context parallel are not supported in non-distributed environments."
        )

    # Handle context parallel setup
    if item.ulysses_size > 1 or item.ring_size > 1:
        assert item.ulysses_size * item.ring_size == world_size, (
            "The number of ulysses_size and ring_size should be equal to the world size."
        )
        from xfuser.core.distributed import (
            initialize_model_parallel,
            init_distributed_environment,
        )

        init_distributed_environment(
            rank=dist.get_rank(), world_size=dist.get_world_size()
        )
        initialize_model_parallel(
            sequence_parallel_degree=dist.get_world_size(),
            ring_degree=item.ring_size,
            ulysses_degree=item.ulysses_size,
        )

    # Get configuration
    cfg = WAN_CONFIGS[item.task]
    if item.ulysses_size > 1:
        assert cfg.num_heads % item.ulysses_size == 0, (
            "`num_heads` must be divisible by `ulysses_size`."
        )

    if item.sample_fps is not None:
        cfg.sample_fps = item.sample_fps

    logging.info(f"Generation job args: {item}")
    logging.info(f"Generation model config: {cfg}")
    logging.info(f"Number of training timesteps: {cfg.num_train_timesteps}")
    logging.info(f"Parameter dtype: {cfg.param_dtype}")

    # Handle base seed in distributed setting
    if dist.is_initialized():
        base_seed = [item.base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        item.base_seed = base_seed[0]

    # Generate based on task type
    if "s2v" in item.task:
        ref_images = load_ref_images(item.ref_image, SIZE_CONFIGS[item.size])
        logging.info("Creating Phantom-Wan pipeline.")
        wan_s2v = phantom_wan.Phantom_Wan_S2V(
            config=cfg,
            checkpoint_dir=item.ckpt_dir,
            phantom_ckpt=item.phantom_ckpt,
            device_id=device,
            rank=rank,
            t5_fsdp=item.t5_fsdp,
            dit_fsdp=item.dit_fsdp,
            use_usp=(item.ulysses_size > 1 or item.ring_size > 1),
            t5_cpu=item.t5_cpu,
        )

        logging.info(
            f"Generating {'image' if 't2i' in item.task else 'video'} ..."
        )
        video = wan_s2v.generate(
            item.prompt,
            ref_images,
            size=SIZE_CONFIGS[item.size],
            frame_num=item.frame_num,
            shift=item.sample_shift,
            sample_solver=item.sample_solver,
            sampling_steps=item.sample_steps,
            guide_scale_img=item.sample_guide_scale_img,
            guide_scale_text=item.sample_guide_scale_text,
            seed=item.base_seed,
            offload_model=item.offload_model,
        )
    logging.info(f"rank is {rank} and item.save_file is {item.save_file}")
    # Save the result
    if rank == 0:
        if item.save_file is None:
            formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            formatted_prompt = item.prompt.replace(" ", "_").replace("/", "_")[
                :50
            ]
            suffix = ".png" if "t2i" in item.task else ".mp4"
            item.save_file = f"{item.task}_{item.size}_{item.ulysses_size}_{item.ring_size}_{formatted_prompt}_{formatted_time}{suffix}"

        if "t2i" in item.task:
            logging.info(f"Saving generated image to {item.save_file}")
            cache_image(
                tensor=video.squeeze(1)[None],
                save_file=item.save_file,
                nrow=1,
                normalize=True,
                value_range=(-1, 1),
            )
        else:
            logging.info(f"Saving generated video to {item.save_file}")
            cache_video(
                tensor=video[None],
                save_file=item.save_file,
                fps=cfg.sample_fps,
                nrow=1,
                normalize=True,
                value_range=(-1, 1),
            )

    logging.info("Finished and returning content")

    # Return the result
    if rank == 0:
        return {"result": {"file": item.save_file}}
    return {"result": f"finishe executing! Output saved to {item.save_file}"}
