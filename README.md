# Dual Diffusion for Unified Image Generation & Understanding
[paper](https://www.arxiv.org/abs/2501.00289) | [webpage](https://zijieli-jlee.github.io/dualdiff.github.io/)

### 1. Environment

We provided an environment file for pip install. </br>

```bash
pip install -r requirements.txt
```

### 2. Inference of pretrained model

The pre-trained checkpoints can be downloaded through belowing links

| Name       |Specification | Link   |
|---------------|---------|------------------------------------------------------|
|  512 Base  |Dual-diffusion pretrained, can do generation and caption| [model](https://huggingface.co/JleeOfficial/dual_diff_sd3_512_base) |
| 512 SFT |SFT on LLaVA data, can do generation and vqa| [model](https://huggingface.co/JleeOfficial/dual_diff_sd3_512_sft/tree/main) |

After downloading the checkpoints, check the Jupyter notebook  `notebooks/demo.ipynb` for example usage.

Minimal working example:

```python
from sd3_modules.dual_diff_pipeline import DualDiffSD3Pipeline

dual_diff_pipe = DualDiffSD3Pipeline.from_pretrained("./pretrained_models/dual_diff_sd3_512_base", torch_dtype=torch.bfloat16).to('cuda')
imgs = dual_diff_pipe(
        prompt="A gourmet hamburger set on a rustic wooden table. The burger is made with a perfectly grilled, juicy beef patty topped with melted gourmet cheese, crispy bacon, fresh lettuce, ripe tomatoes, and caramelized onions.",
        height=512,
        width=512,
        num_images_per_prompt=1)
```

### 3. Data 

We support two kinds of image-text data, wrapped webdataset (.tar) data and unwrapped data. For unwrapped data, a json file storing data information is needed. The meta data should be a list of dictionaries like below:

```python
[
    {
        "image_path": "images/img1.jpg",
        "ratio": 1.33,
        "height": 600,
        "width": 800,
        "caption": "A sunny day in the park.",
        "re_caption": "A bright, lively park scene."
    },
    {
        "image_path": "images/img2.jpg",
        "ratio": 0.75,
        "height": 400,
        "width": 300,
        "caption": "A night sky full of stars.",
        "re_caption": "The starry night illuminates the scene."
    },

]
```

Following dataset are used in our project:

| Name       |Usage | Link   |
|---------------|---------|------------------------------------------------------|
|  Datacomp-recap  |Base pretraining| [data](https://huggingface.co/datasets/UCSC-VLAA/Recap-DataComp-1B) |
| ShareGPT4V pretrain |T5 embedding alignment, text diffusion training| [data](https://huggingface.co/datasets/Lin-Chen/ShareGPT4V) |
| LAION aesthetic |Image diffusion training| [data](https://huggingface.co/datasets/laion/laion2B-en-aesthetic) |
| MidJourney 1.1M |Image diffusion training| [data](https://huggingface.co/datasets/CaptionEmporium/midjourney-niji-1m-llavanext) |
| LLaVA 1.5 |Text SFT| [data](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_v1_5_mix665k.json) |


### 4. Training

To train the model, a SD3-medium checkpoint is needed, which can be downloaded from [here](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers). In addition, we also have an aligned embedding that corresponds to the "mask token" in T5's volcabulary [here](https://huggingface.co/JleeOfficial/aligned_t5_mask_emb/tree/main).

The example configuration for training are provided under ```configs``` directory. 
We use 32 H100 for the pretraining, 16 A100 for SFT.

* Dual-diffusion training on image-text data (fill the torchrun argument with your machine's setting):
```bash
export OMP_NUM_THREADS=8
precision=bf16

torchrun --nnodes $WORKER_NUM \
    --node_rank $ID \
    --nproc_per_node $WORKER_GPU \
    --master_addr $WORKER_0_HOST \
    --master_port $port \
    train_dual_diffusion_sd3.py  \
        --config configs/dual_diff_pretrain.py \
        --results_dir results/ \
        --model_parallel_size 1 \
        --data_parallel h_sdp \
        --precision ${precision} --grad_precision fp32
```

* Supervised fine-tune (text diffusion with prompt + image diffusion) on some visual-instruction dataset (we used LLaVA 1.5's):
```bash
export OMP_NUM_THREADS=8
precision=bf16

torchrun --nnodes $WORKER_NUM \
    --node_rank $ID \
    --nproc_per_node $WORKER_GPU \
    --master_addr $WORKER_0_HOST \
    --master_port $port \
    train_dual_diffusion_sd3.py  \
        --config configs/dual_diff_sft.py \
        --results_dir results/ \
        --model_parallel_size 1 \
        --data_parallel h_sdp \
        --precision ${precision} --grad_precision fp32 \
        --resume_t5 ${t5_mask_emb_pth}
```



### Acknowledgement

The implementation of this project is inspired from the great codebase of [PixArt](https://github.com/PixArt-alpha/PixArt-alpha), [MDLM](https://github.com/kuleshov-group/mdlm/tree/master), [Lumina-Next](https://github.com/Alpha-VLLM/Lumina-T2X). 

Our DiT backbone is finetuned from [SD3-medium](https://stability.ai/news/stable-diffusion-3-medium). 

### Reference

If you find this project useful, please kindly consider citing our work:
```
@misc{li2024dualdiffusionunifiedimage,
      title={Dual Diffusion for Unified Image Generation and Understanding}, 
      author={Zijie Li and Henry Li and Yichun Shi and Amir Barati Farimani and Yuval Kluger and Linjie Yang and Peng Wang},
      year={2024},
      eprint={2501.00289},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2501.00289}, 
}