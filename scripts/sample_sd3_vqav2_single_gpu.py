import argparse
import datetime
import os
import sys
import sys
# Get the absolute path of the parent directory
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

# Add the parent directory to sys.path
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
import numpy as np
import torch

from mmcv import Config
from torchvision.utils import save_image
import json

import random
from datasets import load_dataset

from torchvision import transforms as T
from torchvision.transforms.functional import InterpolationMode
from diffusers import StableDiffusion3Pipeline as SDPipe
from sd3_modules.sd3_model import SD3JointModelFlexible

from vqa import VQAEval
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"
MASK_TOKEN_IDS = 32099
def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class ImageProcessor:
    def __init__(self, resolution):
        self.resolution = resolution
        self.interpolate_model = InterpolationMode.BICUBIC
        self.transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB')),
            T.Resize(resolution, interpolation=self.interpolate_model),  # Image.BICUBIC
            T.CenterCrop(resolution),
            T.ToTensor(),
            T.Normalize([.5], [.5]),
        ])

    def preprocess(self, img):
        # transform pil image to pytorch tensor
        return self.transform(img)[None]


class SimpleVAEWrapper:
    def __init__(self, vae, sample_posterior):
        self.vae = vae
        self.sample_posterior = sample_posterior

    @torch.no_grad()
    def encode(self, img):
        posterior = self.vae.encode(img).latent_dist
        if self.sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        return z

    @torch.no_grad()
    def decode(self, z):
        samples = self.vae.decode(z / self.vae.config.scaling_factor).sample
        return samples


from typing import Any, Callable, Dict, List, Optional, Union

def _sample_categorical(categorical_probs):
  gumbel_norm = (
    1e-10
    - (torch.rand_like(categorical_probs) + 1e-10).log())
  return (categorical_probs / gumbel_norm).argmax(dim=-1)


class LogLinearNoise(torch.nn.Module):
    """Log Linear noise schedule.

    Built such that 1 - 1/e^(n(t)) interpolates between 0 and
    ~1 when t varies from 0 to 1. Total noise is
    -log(1 - (1 - eps) * t), so the sigma will be
    (1 - eps) * t.
    """
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps
        self.sigma_max = self.total_noise(torch.tensor(1.0))
        self.sigma_min = self.eps + self.total_noise(torch.tensor(0.0))

    def rate_noise(self, t):
        return (1 - self.eps) / (1 - (1 - self.eps) * t)

    def total_noise(self, t):
        return -torch.log1p(-(1 - self.eps) * t)

    def importance_sampling_transformation(self, t):
        f_T = torch.log1p(- torch.exp(- self.sigma_max))
        f_0 = torch.log1p(- torch.exp(- self.sigma_min))
        sigma_t = - torch.log1p(- torch.exp(t * f_T + (1 - t) * f_0))
        t = - torch.expm1(- sigma_t) / (1 - self.eps)
        return t

    def forward(self, t):
        # Assume time goes from 0 to 1
        return self.total_noise(t), self.rate_noise(t)

@torch.no_grad()
def prepare_text_inputs(model_pipe,
                         t5_input_ids,
                         attention_mask):
    t5_embeds = model_pipe.text_encoder_3(t5_input_ids, attention_mask=attention_mask)[0]
    return t5_embeds



class TextModelWrapper:
    def __init__(self, model, model_pipe):
        self.model = model
        self.neg_infinity = -1000000.0
        self.model_pipe = model_pipe

    def __call__(self, xt, # maybe zero to one
                 y, mask_index):
        cond = y
        text_hidden_states = prepare_text_inputs(self.model_pipe, xt, attention_mask=None)
        # print(y.device)
        # print(text_hidden_states.device)
        # print(self.model.device)
        # with torch.cuda.amp.autocast(enabled=True):  
            # note that xt is indices   
        # disable autocast, just use fp16
        logits = self.model(hidden_states=cond,
                            timestep=torch.zeros(xt.shape[0], device=xt.device), # note that, this time embedding is for image, we don't use time embedding for text
                            encoder_hidden_states=text_hidden_states.detach(),
                            pooled_projections=None)[1]
            
        logits = logits.float()
        # print(logits[..., 0])

        # log prob at the mask index = - infinity
        logits[:, :, mask_index] += self.neg_infinity
        
        # Normalize the logits such that x.exp() is
        # a probability distribution over vocab_size.
        logits = logits - torch.logsumexp(logits, dim=-1,
                                        keepdim=True)

        # Apply updates directly in the logits matrix.
        # For the logits of the unmasked tokens, set all values
        # to -infinity except for the indices corresponding to
        # the unmasked tokens.
        unmasked_indices = (xt != mask_index)
        logits[unmasked_indices] = self.neg_infinity
        logits[unmasked_indices, xt[unmasked_indices]] = 0
        return logits


class ConditionalMaskedDiffusionSampler:

    def __init__(self, mask_index, length, model_wrapper):
    # the model needs to be wrapped with SUBS parameterization at first
        self.mask_index = mask_index
        self.length = length
        self.model = model_wrapper
        self.noise = LogLinearNoise()
        self.y = None
        self.noise_removal = True
        self.sampler = 'ddpm_cache'

    def register_condition(self, y, x, x_mask):
        self.y = y
        self.x = x
        self.x_mask = x_mask
    
    def clear_condition(self):
        self.y = None

    def _sample_prior(self, *batch_dims):
        masked = self.mask_index * torch.ones(
        * batch_dims, dtype=torch.int64)
        return torch.where(self.x_mask, self.x, masked)

    def forward(self, x):
        return self.model(x, self.y, self.mask_index)

    def _ddpm_update(self, x, t, dt):
        sigma_t, _ = self.noise(t)
        sigma_s, _ = self.noise(t - dt)
        if sigma_t.ndim > 1:
            sigma_t = sigma_t.squeeze(-1)
        if sigma_s.ndim > 1:
            sigma_s = sigma_s.squeeze(-1)
        assert sigma_t.ndim == 1, sigma_t.shape
        assert sigma_s.ndim == 1, sigma_s.shape
        move_chance_t = 1 - torch.exp(-sigma_t)
        move_chance_s = 1 - torch.exp(-sigma_s)
        move_chance_t = move_chance_t[:, None, None]
        move_chance_s = move_chance_s[:, None, None]
        log_p_x0 = self.forward(x)
        assert move_chance_t.ndim == log_p_x0.ndim
        # Technically, this isn't q_xs since there's a division
        # term that is missing. This division term doesn't affect
        # the samples.
        q_xs = log_p_x0.exp() * (move_chance_t
                                - move_chance_s)
        q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]
        _x = _sample_categorical(q_xs)

        copy_flag = (x != self.mask_index).to(x.dtype)
        return copy_flag * x + (1 - copy_flag) * _x
    
    def _ddpm_caching_update(self, x, t, dt, p_x0):
        sigma_t, _ = self.noise(t)
        if t.ndim > 1:
            t = t.squeeze(-1)
        assert t.ndim == 1
        move_chance_t = t[:, None, None]
        move_chance_s = (t - dt)[:, None, None]
        assert move_chance_t.ndim == 3, move_chance_t.shape
        if p_x0 is None:
            p_x0 = self.forward(x).exp()
        
        assert move_chance_t.ndim == p_x0.ndim
        q_xs = p_x0 * (move_chance_t - move_chance_s)
        q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]
        _x = _sample_categorical(q_xs)
        
        copy_flag = (x != self.mask_index).to(x.dtype)
        return p_x0, copy_flag * x + (1 - copy_flag) * _x

    @torch.no_grad()
    def sample(self,  num_steps,
               eps=1e-5, batch_size_per_gpu=1, device='cuda'):
        """Generate samples from the model."""
        # assert self.y is not None
        batch_size_per_gpu = batch_size_per_gpu

        x = self._sample_prior(
            batch_size_per_gpu,
            self.length).to(device)

        timesteps = torch.linspace(
            1, eps, num_steps + 1, device=device)
        dt = (1 - eps) / num_steps
        p_x0_cache = None


        for i in range(num_steps):
            t = timesteps[i] * torch.ones(x.shape[0], 1, device=device)
            if self.sampler == 'ddpm':
                x = self._ddpm_update(x, t, dt)
            elif self.sampler == 'ddpm_cache':
                p_x0_cache, x_next = self._ddpm_caching_update(
                                    x, t, dt, p_x0=p_x0_cache)
                if not torch.allclose(x_next, x):
                    # Disable caching
                    p_x0_cache = None
                x = x_next

        if self.noise_removal:
            t = timesteps[-1] * torch.ones(x.shape[0], 1,
                                            device=device)
            
            x = self.forward(x).argmax(dim=-1)
        return x


def prepare_image_inputs(image_tsr):
    image_tsr = image_tsr.to(sd_model_pipe.device)
    image_vae_feat = sd_model_pipe.vae.encode(image_tsr).latent_dist.sample()
    image_vae_feat = (image_vae_feat - sd_model_pipe.vae.config.shift_factor) * sd_model_pipe.vae.config.scaling_factor

    image_vae_feat = image_vae_feat.detach()

    return image_vae_feat

    
def format_question(text, max_length=128):
    if args.dataset == 'vizwiz':
        preamble = f'Q: {text} Answer the question using a single word or phrase. Reply unanswerable if information is not present in the image. A: '
    else:
        preamble = f'Q: {text} Answer the question using a single word or phrase. A: '
    tokens = sd_model_pipe.tokenizer_3(preamble, max_length=max_length, truncation=True, return_tensors='pt')

    question = tokens['input_ids'][:, :-1]
    seq_length = question.shape[1]
    if seq_length > 54:
        print(f"WARNING: Prompt is more than 54 tokens {seq_length}, leaving < 10 tokens for sampling. Prompt: {preamble}")

    question_mask = torch.ones_like(question, dtype=torch.bool)
    pad_length = max_length - seq_length
    if pad_length:
        n, d = question_mask.shape[0], pad_length
        question_mask = torch.cat([question_mask, torch.zeros(size=(n, d), dtype=torch.bool)], dim=1)
        question = torch.cat([question, torch.ones(size=(n, d), dtype=torch.long) * sd_model_pipe.tokenizer_3.pad_token_id], dim=1)

    return preamble, question, question_mask, seq_length

@torch.no_grad()
def sample_and_save(seed=42):
    print(f'Captions to generate: {args.exit_after}')

    bs = 1
    sample_steps = args.sample_steps
    print(f'Start sampling, using sample_steps={sample_steps}')
    p = np.random.RandomState(seed).permutation(len(dataset))

    if args.start > 0:
        with open(f"{args.split}_vqav2.json", 'r') as f:
            results = json.load(f)
        with open(f"{args.split}_vqav2_submit.json", 'r') as f:
            results_slim = json.load(f)
        assert np.abs(len(results) - args.start) < 100
        args.start = len(results)
    else:
        results = []
        results_slim = [None] * len(dataset)
        if args.split == 'testdev':
            try:
                results_fpath = f"testdev_example.json"
                with open(results_fpath, 'r') as f:
                    results_slim = json.load(f)
                print("Loaded empty results file from...")
                assert len(results_slim) == 447793
            except Exception as e:
                print(f"Error: {e}. Regenerating empty results file.")
                testdev = set()
                for i in tqdm(range(len(p))):
                    qid = dataset[int(p[i])]['question_id']
                    results_slim.append({'question_id': qid, 'answer': ''})
                    testdev.add(qid)

                for i in tqdm(range(len(dataset['testdev']))):
                    qid = dataset[i]['question_id']
                    if qid not in testdev:
                        results_slim.append({'question_id': qid, 'answer': ''})
    accuracies = []
    running_accs = []
    for i in tqdm(range(args.start, args.exit_after)):
        idx = int(p[i])
        batch = dataset[idx]
        image_tsr = img_processor.preprocess(batch['image']).to(model.dtype).to(device)
        result = {}
        result['question_id'] = batch['question_id']

        preamble, question, question_mask, q_len = format_question(batch['question'])

        question = question
        question_mask = question_mask
        result['question'] = preamble

        # print(question.device, question_mask.device, image_tsr.device)
        
        with torch.inference_mode():    
            # for each image we sample 1 caption
            img_feat = prepare_image_inputs(image_tsr)
            caption_sampler.register_condition(img_feat, question, question_mask)
            pred = caption_sampler.sample(args.sample_steps, batch_size_per_gpu=img_feat.shape[0], device=device)

        if (i + 1) % 5 == 0:
            print(f'Sampled captions: {(i+1) * bs}')

        a = pred[:, q_len:]
        b = pred[torch.logical_not(question_mask)].reshape(len(a), -1)
        assert torch.allclose(a, b), print(a.shape, b.shape)
        assert a.shape == b.shape
        a = sd_model_pipe.tokenizer_3.batch_decode(a)
        b = sd_model_pipe.tokenizer_3.batch_decode(b)
        if a != b:
            print(type(a), type(b), len(a), len(b), a[0], b[0])
            c = pred[:, q_len:]
            d = pred[torch.logical_not(question_mask)]
            print("A B", torch.allclose(c, d), a[a.find(sd_model_pipe.tokenizer_3.eos_token)], b[b.find(sd_model_pipe.tokenizer_3.eos_token)], c, d)
        
        answer = sd_model_pipe.tokenizer_3.batch_decode(pred[:, q_len:])[0]
        # print(preamble)
        # print(answer)
        idx = answer.find(sd_model_pipe.tokenizer_3.eos_token)
        result['pred_answer'] = answer[:idx]

        result['processed_answer'] = vqa_eval.process_answer(result['pred_answer'])
        if args.dataset == 'vizwiz' and 'unanswer' in result['pred_answer']:
            result['pred_answer'] = 'unanswerable'
        
        if args.split == 'testdev':
            result['acc'] = 0.
            ex_answers = None
        else:
            result['answers'] = batch['answers'] if 'answers' in batch else batch['answer']
            if args.dataset == 'vizwiz' and False:
                our_pred = result['pred_answer'] == 'unanswerable'
                answers = [x['answer'] == 'unanswerable' for x in result['answers']]
                result['acc'] = min(sum([int(our_pred == x) for x in answers]), 1)
            elif args.dataset == 'vqav2':
                vqa_eval.evaluate(quesIds=[0], trues=[result['answers']], preds=[result['processed_answer']])
                result['acc'] = vqa_eval.accuracy['overall']
                ex_answers = [x['answer'] for x in result['answers'][:3]]

            elif args.dataset == 'pope':
                  # Only keep the first sentence
                if result['processed_answer'].find('.') != -1:
                    result['processed_answer'] = result['processed_answer'].split('.')[0]
                text = result['processed_answer'].replace(',', '')
                words = text.split(' ')
                if 'No' in words or 'not' in words or 'no' in words:
                    ans = 'no'
                else:
                    ans = 'yes'
                if ans == result['answers']:
                    result['acc'] = 1.
                else:
                    result['acc'] = 0.


        results.append(result)
        if results_slim[i] is not None:
            assert results_slim[i]['question_id'] == result['question_id'], f"{results_slim[i]['question_id']} != {result['question_id']}"
        results_slim[i] = {'question_id': result['question_id'], 'answer': result['processed_answer']}
        accuracies.append(result['acc'])
        running_accs.append(result['acc'])
        
        # print(result['question'], result['processed_answer'], ex_answers, result['acc'], np.mean(accuracies))
        

        if i % 100 == 0 or i >= args.exit_after-1:
            print("Saving to:", save_root)
            print('Accs since last:', np.mean(running_accs))
            print('Accs so far:', np.mean(accuracies))
            running_accs = []
            with open(os.path.join(save_root, f"{args.split}_{args.dataset}.json"), 'w') as f:
                json.dump(results, f)
            with open(os.path.join(save_root, f"{args.split}_{args.dataset}_submit.json"), 'w') as f:
                json.dump(results_slim, f)

        if i >= args.exit_after-1:
            print(np.mean(accuracies))
            break
    print(np.mean(accuracies))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pipeline_load_from", default='/mnt/data/pretrain_ckpt/sd3_orig',
        type=str
    )
    parser.add_argument('--resume_from', type=str)
    parser.add_argument('--work_dir', type=str, default='./vqa_eval')
    parser.add_argument('--dataset', type=str, default='vqav2')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--sample_steps', default=32, type=int)
    parser.add_argument('--exit_after', default=-1, type=int)
    parser.add_argument('--start', default=0, type=int)
    parser.add_argument('--split', type=str, default='validation')
    parser.add_argument('--res', default=512, type=int)
    return parser.parse_args()


def set_env(seed=0):
    torch.manual_seed(seed)
    torch.set_grad_enabled(False)
    for _ in range(30):
        torch.randn(1, 4, 256, 256)


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


if __name__ == '__main__':
    args = get_args()
    # Setup PyTorch:
    seed = args.seed
    set_env(seed)

    os.umask(0o000)
    os.makedirs(args.work_dir, exist_ok=True)

    vqa_eval = VQAEval()


    set_random_seed(args.seed)

    device = 'cuda:0'


    sd_model_pipe = SDPipe.from_pretrained(args.pipeline_load_from, torch_dtype=torch.float16)
    sd_model_pipe.to(device)
    sd_model_pipe.tokenizer_3.pad_token = sd_model_pipe.tokenizer_3.eos_token   # change padding token to eos token

    orig_sd_transformer = sd_model_pipe.transformer

    model = SD3JointModelFlexible(
        len(sd_model_pipe.tokenizer_3),
            **orig_sd_transformer.config
    )
    # model.pos_embed.interpolation_scale = 0.5

    ckpt_file = args.resume_from
    state_dict = torch.load(ckpt_file, map_location="cpu")

    missing, unexpect = model.load_state_dict(state_dict, strict=False)
    print(f'Missing keys: {missing}')
    print(f'Unexpected keys: {unexpect}')

    model = model.eval().to(torch.float16).to(device)
    # remove unnecessary stuff from sd3 model pipe to save memory
    # delete all the encoder/weight not used in sd model pipe
    sd_model_pipe.transformer = None
    sd_model_pipe.text_encoder_1 = None
    sd_model_pipe.text_encoder_2 = None
    sd_model_pipe.text_encoder_3.requires_grad_(False)
    with torch.no_grad():

        new_mask_token_emb = torch.load(
            os.path.join('./pretrained_diff_models/mask_token_emb.00-of-01.pth'),
            map_location="cpu",
        )
        sd_model_pipe.text_encoder_3.shared.weight.data[MASK_TOKEN_IDS] = new_mask_token_emb.to(sd_model_pipe.text_encoder_3.device)
    torch.cuda.empty_cache()
    print('Checkpoint loaded!')

    print('Preparing data')
    # the local path is hard coded
    img_processor = ImageProcessor(args.res)
    if args.dataset == 'vqav2':
        full_dataset = load_dataset("HuggingFaceM4/VQAv2", 
         trust_remote_code=True,
         cache_dir='/mnt/data/')
        dataset = full_dataset[args.split]
    else:

        raise NotImplementedError
    #     dataset = TextVQADataset(split=args.split)
    print('Dataset loaded')

    if args.exit_after < 0:
        args.exit_after = len(dataset)
    else:
        args.exit_after = min(len(dataset), args.exit_after)

    # img save setting

    save_root = args.work_dir

    model_wrapper = TextModelWrapper(model, model_pipe=sd_model_pipe)
    caption_sampler = ConditionalMaskedDiffusionSampler(32099, 128, model_wrapper)

    sample_and_save()
