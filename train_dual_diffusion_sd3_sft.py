

"""
A minimal training script using PyTorch FSDP.
"""
import argparse
from collections import OrderedDict
import contextlib
from copy import deepcopy
from datetime import datetime
import functools
import json
import logging
import os
import random
import socket
from time import time
import warnings
import gc
import sys
# byte-wandb huggingface
if sys.version_info < (3, 8):
    import importlib_metadata
else:
    import importlib.metadata as importlib_metadata
old_metadata = importlib_metadata.metadata

def new_metadata(name):
    if name == 'wandb':
        name =  'byted-wandb'
    return old_metadata(name)

importlib_metadata.metadata = new_metadata

import fairscale.nn.model_parallel.initialize as fs_init
import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_constant_schedule_with_warmup

from grad_norm import calculate_l2_grad_norm, get_model_parallel_dim_dict, scale_grad
from parallel import distributed_init, get_intra_node_process_group

from models.dit import MMDIT

import wandb
from mmcv import Config
from custom_dataset.wds_data import get_urls, wds_dataloader
from custom_dataset.iterable_data import FlexibleInternalData
from custom_dataset.text_data import get_dataset
from sd3_modules.sd_loss_utils import ImageFlowMatchingLoss, TextMaskedDiffusionLoss
from diffusers import StableDiffusion3Pipeline as SDPipe
from sd3_modules.sd3_model import SD3JointModelFlexible

import warnings
warnings.filterwarnings("ignore")  # ignore warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

MASK_TOKEN_IDS = 32099 # <'extra_id0'>


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999, copy=False):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    assert set(ema_params.keys()) == set(model_params.keys())
    if not copy:
        for name, param in model_params.items():
            # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
            ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)
    else:
        # for initialization
        for name, param in model_params.items():
            ema_params[name].data.copy_(param.data)


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format="[\033[34m%(asctime)s\033[0m] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f"{logging_dir}/log.txt"),
            ],
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def read_config(file):
    # solve config loading conflict when multi-processes
    import time
    while True:
        config = Config.fromfile(file)
        if len(config) == 0:
            time.sleep(0.1)
            continue
        break
    return config

def setup_lm_fsdp_sync(model: nn.Module) -> FSDP:
    # LM FSDP always use FULL_SHARD among the node.
    model = FSDP(
        model,
        auto_wrap_policy=functools.partial(
            lambda_auto_wrap_policy,
            lambda_fn=lambda m: m in list(model.encoder.block), # this assumes huggingface T5Encodermodel
        ),
        process_group=get_intra_node_process_group(),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=MixedPrecision(
            param_dtype=next(model.parameters()).dtype,
        ),
        device_id=torch.cuda.current_device(),
        sync_module_states=True,
        limit_all_gathers=True,
        use_orig_params=True,
    )
    torch.cuda.synchronize()
    return model



def setup_fsdp_sync(model: nn.Module, args: argparse.Namespace) -> FSDP:
    model = FSDP(
        model,
        auto_wrap_policy=functools.partial(
            lambda_auto_wrap_policy,
            lambda_fn=lambda m: m in model.get_fsdp_wrap_module_list(),
        ),
        # process_group=fs_init.get_data_parallel_group(),
        sharding_strategy={
            "h_sdp": ShardingStrategy._HYBRID_SHARD_ZERO2,
            "h_fsdp": ShardingStrategy.HYBRID_SHARD,
            "fsdp": ShardingStrategy.FULL_SHARD,
            "sdp": ShardingStrategy.SHARD_GRAD_OP,
        }[args.data_parallel],
        mixed_precision=MixedPrecision(
            param_dtype={
                "fp32": torch.float,
                "tf32": torch.float,
                "bf16": torch.bfloat16,
                "fp16": torch.float16,
            }[args.precision],
            reduce_dtype={
                "fp32": torch.float,
                "tf32": torch.float,
                "bf16": torch.bfloat16,
                "fp16": torch.float16,
            }[args.grad_precision or args.precision],
        ),
        device_id=torch.cuda.current_device(),
        sync_module_states=True,
        limit_all_gathers=True,
        use_orig_params=True,
    )
    torch.cuda.synchronize()

    return model


def setup_mixed_precision(config):
    if config.precision == "tf32":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    elif config.precision in ["bf16", "fp16", "fp32"]:
        pass
    else:
        raise NotImplementedError(f"Unknown precision: {args.precision}")


#############################################################################
#                                Training Loop                              #
#############################################################################


def main(args):
    """
    Trains a MMDiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    config = read_config(args.config)

    distributed_init(args)

    dp_world_size = fs_init.get_data_parallel_world_size()
    dp_rank = fs_init.get_data_parallel_rank()
    mp_world_size = fs_init.get_model_parallel_world_size()
    mp_rank = fs_init.get_model_parallel_rank()

    assert config.global_batch_size % dp_world_size == 0, f"Batch size ({config.global_batch_size}) must be divisible by data parrallel world size ({dp_world_size})."
    local_batch_size = config.global_batch_size // dp_world_size
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    setup_mixed_precision(args)
    # print(f"Starting rank={rank}, seed={seed}, "
    #       f"world_size={dist.get_world_size()}.")

    # Setup an experiment folder:
    os.makedirs(args.results_dir, exist_ok=True)
    checkpoint_dir = os.path.join(args.results_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    if rank == 0:
        logger = create_logger(args.results_dir)
        logger.info(f"Experiment directory: {args.results_dir}")
        # initialize wandb
        wandb.init(
            project = config.project_name,
            name = config.run_name, 
        )
    else:
        logger = create_logger(None)

    logger.info("Training arguments: " + json.dumps(args.__dict__, indent=2))
    
    # load sd3 pipe
    sd_model_pipe = SDPipe.from_pretrained(config.sd3_pipeline_load_from, torch_dtype=torch.bfloat16).to(device)
    sd_model_pipe.tokenizer_3.pad_token = sd_model_pipe.tokenizer_3.eos_token   # change padding token to eos token
    orig_sd_transformer = sd_model_pipe.transformer

    text_tokenizer = sd_model_pipe.tokenizer_3

    logger.info('sd pipeline loaded, text encoder was also prepared')
    logger.info(f"Creating text diffusion model from SD3")

    model = SD3JointModelFlexible(
        len(sd_model_pipe.tokenizer_3),
            **orig_sd_transformer.config
    ).train()

    logger.info(f"Model trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # remove unnecessary stuff from sd3 model pipe to save memory
    # delete all the encoder/weight not used in sd model pipe
    sd_model_pipe.transformer = None
    sd_model_pipe.text_encoder_1 = None
    sd_model_pipe.text_encoder_2 = None
    sd_model_pipe.text_encoder_3.requires_grad_(False)
    sd_model_pipe.text_encoder_3.eval()
    gc.collect()
    torch.cuda.empty_cache()

    if config.resume_from_legacy:
        resume_path = config.resume_from_legacy
        logger.info(f'Resuming legacy model from {resume_path}')

        checkpoint = torch.load(resume_path, map_location="cpu")
        state_dict = checkpoint['state_dict']

        missing, unexpect = model.load_state_dict(state_dict, strict=False)
        logger.warning(f'Missing keys: {missing}')
        logger.warning(f'Unexpected keys: {unexpect}')
        del checkpoint
        torch.cuda.empty_cache()
        gc.collect()
        logger.info(f'Pretrained mdoel loaded from {resume_path}')

    
    model_parallel_dim_dict = get_model_parallel_dim_dict(model)

    if args.auto_resume and args.resume is None:
        try:
            if os.path.exists(os.path.join(checkpoint_dir, 'current')):
                args.resume = os.path.join(checkpoint_dir, 'current')
            else:
                existing_checkpoints = os.listdir(checkpoint_dir)
                if len(existing_checkpoints) > 0:
                    existing_checkpoints.sort()
                    args.resume = os.path.join(checkpoint_dir, existing_checkpoints[-1])
        except Exception:
            pass
        if args.resume is not None:
            logger.info(f"Auto resuming from: {args.resume}")

    # always reload t5 token from our aligned checkpoint
    if args.resume_t5:
        try:
            mask_token_emb_fn = (
                            "mask_token_emb."
                            f"{fs_init.get_model_parallel_rank():02d}-of-"
                            f"{fs_init.get_model_parallel_world_size():02d}"
                            ".pth"
                        )
            logger.info(f"Resuming mask token embedding from: {args.resume_t5}")
            with torch.no_grad():
                sd_model_pipe.text_encoder_3.shared.weight.data[MASK_TOKEN_IDS] = torch.load(
                    os.path.join(args.resume_t5, mask_token_emb_fn),
                    map_location="cpu",
                )
        except Exception as e:
            logger.info("Error loading mask token embedding: {e}")
            raise e

    # Note that parameter initialization is done within the DiT constructor
    model_ema = deepcopy(model)
    if args.resume:
        logger.info(f"Resuming model weights from: {args.resume}")
        model.load_state_dict(
            torch.load(
                os.path.join(
                    args.resume,
                    f"consolidated.{mp_rank:02d}-of-{mp_world_size:02d}.pth",
                ),
                map_location="cpu",
            ),
            strict=True,
        )
        
        try:
            if os.path.exists(os.path.join(
                        args.resume,
                        f"consolidated_ema.{mp_rank:02d}-of-{mp_world_size:02d}.pth",
                    )):
                
                logger.info(f"Resuming ema weights from: {args.resume}")
                model_ema.load_state_dict(
                    torch.load(
                        os.path.join(
                            args.resume,
                            f"consolidated_ema.{mp_rank:02d}-of-{mp_world_size:02d}.pth",
                        ),
                        map_location="cpu",
                    ),
                    strict=True,
                )
        except Exception as e:
            logger.info("Error loading ema weights: {e}")

    dist.barrier()

    model = setup_fsdp_sync(model, args)
    model_ema = setup_fsdp_sync(model_ema, args)

    opt = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.wd)
    scheduler = get_constant_schedule_with_warmup(opt,
                                                num_warmup_steps=config.num_warmup_steps)
    if args.resume:
        opt_state_world_size = len(
            [x for x in os.listdir(args.resume) if x.startswith("optimizer.") and x.endswith(".pth")]
        )
        if opt_state_world_size != dist.get_world_size():
            logger.info(
            f"Resuming from a checkpoint with unmatched world size "
            f"({dist.get_world_size()} vs. {opt_state_world_size}) "
            f"is currently not supported."
            )
        else:
            logger.info(f"Resuming optimizer states from: {args.resume}")
            try:
                opt.load_state_dict(
                    torch.load(
                        os.path.join(
                            args.resume,
                            f"optimizer.{dist.get_rank():05d}-of-" f"{dist.get_world_size():05d}.pth",
                        ),
                        map_location="cpu",
                    )
                )
                for param_group in opt.param_groups:
                    param_group["lr"] = config.lr
                    param_group["weight_decay"] = config.wd
            except Exception as e:
                logger.info("Error Loading Optimizer State:", e)
                logger.info("Continuing without loading optimizer state...")

        # resume scheduler
        scheduler_state = torch.load(
            os.path.join(
                args.resume,
                f"scheduler.pth",
            ),
            map_location="cpu",
        )
        scheduler.load_state_dict(scheduler_state)

        with open(os.path.join(args.resume, "resume_step.txt")) as f:
            resume_step = int(f.read().strip())
    else:
        resume_step = 0

    # Setup data:
    logger.info("Creating dataset...")
    t2i_dataset = FlexibleInternalData(tokenizer=sd_model_pipe.tokenizer_3, **config.t2i_data_config)
    t2i_sampler = DistributedSampler(t2i_dataset)

    t2i_dataloader = torch.utils.data.DataLoader(
                                        t2i_dataset,
                                        batch_size=int(local_batch_size),
                                        num_workers=config.num_workers,
                                        pin_memory=True,
                                        shuffle=None,
                                        sampler=t2i_sampler,
                                        drop_last=True,
                                        persistent_workers=True)
    image_text_data_iter = iter(t2i_dataloader)

    # build sft finetune dataloader
    if args.new_dataset:
        from custom_dataset.llava_data_new import LlavaDataset
    else:
        from custom_dataset.llava_data import LlavaDataset
    
    from torch.utils.data import DataLoader

    ld = LlavaDataset(tokenizer=text_tokenizer, max_length=128,  # length is hard-coded for now
                      res=config.resolution,
                      add_vqav2=True, add_okvqa=True, add_gqa=True, add_textvqa=True, add_vizwiz=True, reverse=args.reverse)
    ft_sampler = DistributedSampler(ld)

    ft_dataloader = DataLoader(ld, 
                                batch_size=int(local_batch_size),
                                num_workers=config.num_workers,
                                pin_memory=True,
                                shuffle=None,
                                sampler=ft_sampler,
                                drop_last=True,
                                persistent_workers=True)
    ft_data_iter = iter(ft_dataloader)


    model.train()

    # good, now we setup the loss function
    text_diffusion_loss_module = TextMaskedDiffusionLoss(
        config,
        model_pipe = sd_model_pipe,
    )

    image_diffusion_loss_module = ImageFlowMatchingLoss(
        model_pipe = sd_model_pipe,
        text_max_length=128    # hard-coded to fit sft length
    )


    # Variables for monitoring/logging purposes:
    log_steps = 0
    running_loss = 0
    start_time = time()

    # steps_chunk = config.total_steps_chunk
    # text_steps = int(steps_chunk * config.training.text_training_weight)
    # image_steps = steps_chunk - text_steps

    # logger.info(f'For every {steps_chunk}, text steps: {text_steps}, image steps: {image_steps}')

    step = resume_step
    # step = 0
    # while step < resume_step:
    #     if step % 100 ==0:
    #         logger.info(f'Skipped step {step} in dataloader')
    #         gc.collect()
    #         torch.cuda.empty_cache()

    #     try:
    #         _ = next(ft_data_iter)
    #         _ = next(image_text_data_iter)
    #         _ = next(text_data_iter)
    #     except StopIteration:
    #         ft_data_iter = iter(ft_dataloader)
    #         _ = next(image_text_data_iter)

    #         image_text_data_iter = iter(image_text_dataloader)
    #         text_data_iter = iter(text_train_dataloader)
    #         _ = next(text_data_iter)
    #         _ = next(ft_data_iter)
    #     step += 1

    running_image_loss = 0.0
    running_ft_loss = 0.0

    logger.info(f"Training for {config.max_steps:,} steps...")
    while step < config.max_steps:

        try:
            ft_batch = next(ft_data_iter)

            image_text_batch = next(image_text_data_iter)
            # text_batch = next(text_data_iter)
        except StopIteration:
            ft_data_iter = iter(ft_dataloader)
            ft_batch = next(ft_data_iter)

            image_text_data_iter = iter(t2i_dataloader)
            # text_data_iter = iter(text_train_dataloader)
            image_text_batch = next(image_text_data_iter)
            # text_batch = next(text_data_iter)

        # text_training_flag = (step % steps_chunk) < text_steps
        imgs = image_text_batch[0].to(device, non_blocking=True)

        ft_ids = ft_batch[1]['input_ids'].to(device)
        ft_label_mask = ft_batch[1]['label_mask'].to(device)
        ft_imgs = ft_batch[0].to(device, non_blocking=True)
 
        p_real_cap = np.random.rand()
        if p_real_cap < config.train_real_cap_ratio: # always use re-caption to do text generation
            i2t_caption_ids = image_text_batch[1]['input_ids'].to(device).squeeze()
            t2i_caption_ids = image_text_batch[2]['input_ids'].to(device).squeeze()
        else:
            t2i_caption_ids = i2t_caption_ids = image_text_batch[1]['input_ids'].to(device).squeeze()

        # text_ids = text_batch['input_ids'].to(device)

        def get_vae_feats(imgs):
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=True):         #(config.mixed_precision == 'fp16' or config.mixed_precision == 'bf16')):
                    image_vae_feat = sd_model_pipe.vae.encode(imgs).latent_dist.sample()
                    image_vae_feat = (image_vae_feat - sd_model_pipe.vae.config.shift_factor) * sd_model_pipe.vae.config.scaling_factor
                
                image_vae_feat = image_vae_feat.detach()
            return image_vae_feat

        image_vae_feat = get_vae_feats(imgs)
        ft_image_vae_feat = get_vae_feats(ft_imgs)
        
        
        # no model parallel as for now
        # if mp_world_size > 1:
        #     mp_src = fs_init.get_model_parallel_src_rank()
        #     mp_group = fs_init.get_model_parallel_group()
        #     for img in x:
        #         dist.broadcast(img, mp_src, mp_group)
        #     dist.broadcast(text_ids, mp_src, mp_group)
        #     assert text_ids.size(0) % mp_world_size == 0
        #     text_ids = text_ids[
        #         text_ids.size(0) // mp_world_size * mp_rank,
        #         text_ids.size(0) // mp_world_size * (mp_rank + 1),
        #     ]

        # get clean text emb
        with torch.no_grad():
            t2i_caption_emb = sd_model_pipe.text_encoder_3(t2i_caption_ids)[0]

        loss_item = 0.0
        
        opt.zero_grad()
        # the micro batch size controls gradient accumulation
        for mb_idx in range((local_batch_size - 1) // config.micro_batch_size + 1):
            mb_st = mb_idx * config.micro_batch_size
            mb_ed = min((mb_idx + 1) * config.micro_batch_size, local_batch_size)
            last_mb = mb_ed == local_batch_size

            ft_ids_mb = ft_ids[mb_st:mb_ed]
            ft_x_mb = ft_image_vae_feat[mb_st:mb_ed]
            ft_label_mask_mb = ft_label_mask[mb_st:mb_ed]

            x_mb = image_vae_feat[mb_st:mb_ed]
            # i2t_ids_mb = i2t_caption_ids[mb_st:mb_ed]     # not used
            t2i_caption_mb = t2i_caption_emb[mb_st:mb_ed]
            # text_ids_mb = text_ids[mb_st*config.text_ratio:mb_ed*config.text_ratio]  # not used

            with {
                "bf16": torch.cuda.amp.autocast(dtype=torch.bfloat16),
                "fp16": torch.cuda.amp.autocast(dtype=torch.float16),
                "fp32": contextlib.nullcontext(),
                "tf32": contextlib.nullcontext(),
            }[args.precision]:
                # instruction tuning loss
                fine_tune_loss = text_diffusion_loss_module.compute_loss(
                                                    model, ft_ids_mb, ft_x_mb, None, label_mask=ft_label_mask_mb,
                                                    use_dummy_loss=False, 
                                                )
                
                # image loss
                image_diffusion_loss = image_diffusion_loss_module.compute_loss(
                                                    model, t2i_caption_mb, x_mb, 
                                                )


                loss = image_diffusion_loss + fine_tune_loss * config.training.caption_training_weight
                    
            running_caption_loss += caption_diffusion_loss.item()
            running_ft_loss += fine_tune_loss.item()
                
            loss_item += loss.item()
            with model.no_sync() if args.data_parallel in ['h_sdp', 'h_fsdp', "sdp", "fsdp"] and not last_mb else contextlib.nullcontext():
                loss.backward()

        grad_norm = calculate_l2_grad_norm(model, model_parallel_dim_dict)
        if grad_norm > config.grad_clip:
            scale_grad(model, config.grad_clip / grad_norm)

        opt.step()
        scheduler.step()
        step += 1
        update_ema(model_ema, model)

        # Log loss values:
        running_loss += loss_item
        log_steps += 1
        if step % config.log_every == 0:
            gradient_accumulation_steps = (local_batch_size - 1) // config.micro_batch_size + 1
            # Measure training speed:
            torch.cuda.synchronize()
            end_time = time()
            secs_per_step = (end_time - start_time) / log_steps
            imgs_per_sec = config.global_batch_size * log_steps / (end_time - start_time)

            # Reduce loss history over all processes:
            avg_loss = torch.tensor(running_loss / log_steps, device=device)
            dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
            avg_loss = avg_loss.item() / dist.get_world_size() / gradient_accumulation_steps

            avg_ft_loss = torch.tensor(running_ft_loss / log_steps, device=device)
            dist.all_reduce(avg_ft_loss, op=dist.ReduceOp.SUM)
            avg_ft_loss = avg_ft_loss.item() / dist.get_world_size() / gradient_accumulation_steps

            avg_image_loss = torch.tensor(running_image_loss / log_steps, device=device)
            dist.all_reduce(avg_image_loss, op=dist.ReduceOp.SUM)
            avg_image_loss = avg_image_loss.item() / dist.get_world_size() / gradient_accumulation_steps
            logger.info(
                f"(Step={step + 1:07d}) "
                f"Image Loss: {avg_image_loss:.4f}, "
                f"SFT Loss: {avg_ft_loss:.4f}, "
                f"Train Secs/Step: {secs_per_step:.2f}, "
                f"Train Imgs/Sec: {imgs_per_sec:.2f}, "
                f"Train grad norm: {grad_norm:.2f},"
            )
            # call wandb log on main rank
            if rank == 0:
                # collect text loss, image loss, grad norm, and lr
                wandb.log({"train_loss": avg_loss,
                            "train_image_loss": avg_image_loss,
                            'train_sft_loss': avg_ft_loss,
                            "lr": opt.param_groups[0]["lr"],
                            "grad_norm": grad_norm,
                            }, step=step)


            # Reset monitoring variables:
            running_loss = 0
            running_image_loss = 0
            running_ft_loss = 0

            log_steps = 0

            start_time = time()

        # Save DiT checkpoint:
        if step % 500 == 0 or step % config.ckpt_every == 0 or step == config.max_steps:
            if step % config.ckpt_every == 0 or step == config.max_steps:
                checkpoint_path = f"{checkpoint_dir}/{step + 1:07d}"
            else:
                checkpoint_path = f"{checkpoint_dir}/current"
                
            os.makedirs(checkpoint_path, exist_ok=True)

            with FSDP.state_dict_type(
                model,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
            ):
                consolidated_model_state_dict = model.state_dict()
                if fs_init.get_data_parallel_rank() == 0:
                    consolidated_fn = (
                        "consolidated."
                        f"{fs_init.get_model_parallel_rank():02d}-of-"
                        f"{fs_init.get_model_parallel_world_size():02d}"
                        ".pth"
                    )
                    torch.save(
                        consolidated_model_state_dict,
                        os.path.join(checkpoint_path, consolidated_fn),
                    )
            dist.barrier()
            del consolidated_model_state_dict
            logger.info(f"Saved consolidated to {checkpoint_path}.")


            with FSDP.state_dict_type(
                model_ema,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
            ):
                consolidated_ema_state_dict = model_ema.state_dict()
                if fs_init.get_data_parallel_rank() == 0:
                    consolidated_ema_fn = (
                        "consolidated_ema."
                        f"{fs_init.get_model_parallel_rank():02d}-of-"
                        f"{fs_init.get_model_parallel_world_size():02d}"
                        ".pth"
                    )
                    torch.save(
                        consolidated_ema_state_dict,
                        os.path.join(checkpoint_path, consolidated_ema_fn),
                    )
            dist.barrier()
            del consolidated_ema_state_dict
            logger.info(f"Saved consolidated_ema to {checkpoint_path}.")

            with FSDP.state_dict_type(
                model_ema,
                StateDictType.LOCAL_STATE_DICT,
            ):
                opt_state_fn = f"optimizer.{dist.get_rank():05d}-of-" f"{dist.get_world_size():05d}.pth"
                torch.save(opt.state_dict(), os.path.join(checkpoint_path, opt_state_fn))
            dist.barrier()
            logger.info(f"Saved optimizer to {checkpoint_path}.")

            # just save scheduler on the main rank
            if rank == 0:
                scheduler_state_fn = f"scheduler.pth"
                torch.save(scheduler.state_dict(), os.path.join(checkpoint_path, scheduler_state_fn))
            dist.barrier()
            logger.info(f"Saved scheduler to {checkpoint_path}.")

            if dist.get_rank() == 0:
                torch.save(args, os.path.join(checkpoint_path, "model_args.pth"))
                with open(os.path.join(checkpoint_path, "resume_step.txt"), "w") as f:
                    print(step + 1, file=f)
            dist.barrier()
            logger.info(f"Saved training arguments to {checkpoint_path}.")

    model.eval()

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument(
        "--no_auto_resume",
        action="store_false",
        dest="auto_resume",
        help="Do NOT auto resume from the last checkpoint in --results_dir.",
    )
    parser.add_argument("--resume", type=str, help="Resume training from a checkpoint folder.")
    parser.add_argument("--resume_t5", type=str, help="Folder containing t5 mask token embedding.")

    parser.add_argument("--model_parallel_size", type=int, default=1)
    parser.add_argument("--data_parallel", type=str, choices=["h_sdp", "h_fsdp", "sdp", "fsdp"], default="fsdp")
    parser.add_argument("--precision", choices=["fp32", "tf32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--grad_precision", choices=["fp32", "fp16", "bf16"], default='fp32')
    
    parser.add_argument("--global_seed", type=int, default=941011)
    parser.add_argument("--new_dataset", action="store_true")
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()

    args.max_seq_len = 388   # not used

    main(args)

