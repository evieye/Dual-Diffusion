import os
import json

from PIL import Image
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import v2
from datasets import load_dataset

import torch.nn.functional as F
import torchvision
import torch

os.environ['HF_HOME'] = '/mnt/bn/us-aigc-temp/henry/'

class CenterCrop(torch.nn.Module):
    def __init__(self, size=None, ratio="1:1"):
        super().__init__()
        self.size = size
        self.ratio = ratio
    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped.

        Returns:
            PIL Image or Tensor: Cropped image.
        """
        if self.size is None:
            if isinstance(img, torch.Tensor):
                h, w = img.shape[-2:]
            else:
                w, h = img.size
            ratio = self.ratio.split(":")
            ratio = float(ratio[0]) / float(ratio[1])
            # Size must match the ratio while cropping to the edge of the image
            ratioed_w = int(h * ratio)
            ratioed_h = int(w / ratio)
            if w>=h:
                if ratioed_h <= h:
                    size = (ratioed_h, w)
                else:
                    size = (h, ratioed_w)
            else:
                if ratioed_w <= w:
                    size = (h, ratioed_w)
                else:
                    size = (ratioed_h, w)
        else:
            size = self.size
        return torchvision.transforms.functional.center_crop(img, size)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size})"
    

class OKVQADataset(Dataset):
    def __init__(self, 
                 root='/mnt/bn/us-aigc-temp/henry/data/okvqa/',
                 img_root=None,
                 transform=None, split='train'):
        self.split = split
        self.root = root
        if img_root is None:
            img_root = f'/mnt/bn/us-aigc-temp/henry/coco_2014/{split}2014/'
        self.img_root = img_root
        self.transform = transform
        
        # okvqa 
        fpath = os.path.join(self.root, f'OpenEnded_mscoco_{split}2014_questions.json')
        with open(fpath, 'r') as f:
            self.questions = json.load(f)

        fpath = os.path.join(self.root, f'mscoco_{split}2014_annotations.json')
        with open(fpath, 'r') as f:
            self.answers = json.load(f)

    def __len__(self):
        assert len(self.questions['questions']) == len(self.answers['annotations'])
        return len(self.questions['questions'])
    
    def __getitem__(self, idx):
        x = self.questions['questions'][idx]
        question = x['question']
        question_id = x['question']
        if 'answers' in self.answers['annotations'][idx]:
            answers = self.answers['annotations'][idx]['answers']
        else:
            answers = None
        image_id = x['image_id']
        image_path = os.path.join(self.img_root, f'COCO_{self.split}2014_{image_id:012d}.jpg')
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)
        
        return_dict = {
            'question': question,
            'question_id': question_id,
            'answers': answers,
            'image': image
        }
        return return_dict
        
class VizWizDataset(Dataset):
    def __init__(self, 
                 root='/mnt/bn/us-aigc-temp/henry/data/vizwiz/',
                 img_root=None,
                 transform=None, split='train'):
        self.split = split
        self.root = root
        if img_root is None:
            img_root = f'/mnt/bn/us-aigc-temp/henry/data/vizwiz/{split}/'
        self.img_root = img_root
        self.transform = transform
        
        # vizwiz 
        fpath = os.path.join(self.root, f'{split}.json')
        with open(fpath, 'r') as f:
            self.annotations = json.load(f)

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        x = self.annotations[idx]
        question = x['question']
        if 'answers' in x:
            answers = x['answers']
        else:
            answers = None
        image_path = os.path.join(self.img_root, x['image'])
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)
        
        return_dict = {
            'question': question,
            'question_id': idx,
            'answers': answers,
            'image': image
        }
        return return_dict
    
class POPEDataset(Dataset):
    def __init__(self, 
                 root='/mnt/bn/us-aigc-temp/henry/data/pope/',
                 img_root=None,
                 transform=None, split='val'):
        assert split == 'val', "ERROR: POPEDataset is not supported yet with split != val."
        self.split = split
        self.root = root
        if img_root is None:
            img_root = f'/mnt/bn/us-aigc-temp/henry/coco_2014/{split}2014/'
        self.img_root = img_root
        self.transform = transform
        
        # vizwiz 
        fpath = os.path.join(self.root, 'coco_ground_truth_segmentation_formatted.json')
        with open(fpath, 'r') as f:
            self.annotations = json.load(f)

        self.all_objects = set()
        for x in self.annotations:
            for obj in x['objects']:
                self.all_objects.add(obj)

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        x = self.annotations[idx]
        if 'objects' in x:
            present_objects = set([x for x in x['objects']])
            if np.random.randint(2) == 0 and len(present_objects):
                #positive
                object_list = present_objects
                answers = [{'answer': 'yes', 'confidence': 'yes', 'answer_id': i + 1} for i in range(10)]
            else:
                # negative
                object_list = self.all_objects.difference(present_objects)
                answers = [{'answer': 'no', 'confidence': 'yes', 'answer_id': i + 1} for i in range(10)]
        else:
            answers = None

        object_idx = np.random.randint(len(object_list))
        object = list(object_list)[object_idx]

        # add grammar
        if object[0] in ['a', 'e', 'i', 'o', 'u']:
            object = 'n ' + object
        else:
            object = ' ' + object
        
        question = f'Is there a{object} in the image?'
        image_path = os.path.join(self.img_root, x['image'])
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)
        
        return_dict = {
            'question': question,
            'question_id': idx,
            'answers': answers,
            'image': image
        }
        return return_dict
    
class TextVQADataset(Dataset):
    def __init__(self, 
                 root='/mnt/bn/us-aigc-temp/henry/data/textvqa/',
                 img_root=None,
                 transform=None, split='train'):
        self.split = split
        self.root = root
        if img_root is None:
            if split == 'val':
                img_root = os.path.join(root, f'train_images/')
            else:
                img_root = os.path.join(root, f'{split}_images/')
        self.img_root = img_root
        self.transform = transform
        
        # okvqa 
        fpath = os.path.join(self.root, f'TextVQA_0.5.1_{split}.json')
        with open(fpath, 'r') as f:
            self.annotations = json.load(f)['data']

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        x = self.annotations[idx]
        question = x['question']
        question_id = x['question_id']
        if 'answers' in x:
            answers = [{'answer': ans, 'answer_id': i + 1} for i, ans in enumerate(x['answers'])]
        else:
            assert False, [k for k in x]
            answers = None
        image_id = x['image_id']
        image_path = os.path.join(self.img_root, f'{image_id}.jpg')
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)
        
        return_dict = {
            'question': question,
            'question_id': question_id,
            'answers': answers,
            'image': image
        }
        return return_dict
    
class GQADataset(Dataset):
    def __init__(self, 
                 root='/mnt/bn/us-aigc-temp/henry/data/gqa/',
                 img_root=None,
                 transform=None, 
                 split='val'):
        self.split = split
        self.root = root
        if img_root is None:
            img_root = f'/mnt/bn/us-aigc-temp/henry/data/gqa/images/'
        self.img_root = img_root
        self.transform = transform
        
        # gqa 
        fpath = os.path.join(self.root, 'testdev_all_questions.json')
        with open(fpath, 'r') as f:
            self.annotations = json.load(f)
        
        self.keys = sorted([x for x in self.annotations.keys()])

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        key = self.keys[idx]
        x = self.annotations[key]
        
        question = x['question']
        if 'answer' in x:
            answers = [{'answer': x['answer'], 'confidence': 'yes', 'answer_id': i + 1} for i in range(10)]
        else:
            assert False, [k for k in x]
            answers = None
        image_path = os.path.join(self.img_root, f"{x['imageId']}.jpg")
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)
        
        return_dict = {
            'question': question,
            'question_id': idx,
            'answers': answers,
            'image': image
        }
        return return_dict

class COCODataset(Dataset):
    def __init__(self, 
                 root='/mnt/bn/us-aigc-temp/henry/coco_2014',
                 img_root=None,
                 transform=None, 
                 split='val'):
        self.split = split
        self.root = root
        if img_root is None:
            img_root = os.path.join(self.root, f'{split}2014/')
        self.img_root = img_root
        self.transform = transform
        
        # coco 
        fpath = os.path.join(self.root, f'annotations/captions_{split}2014.json')
        with open(fpath, 'r') as f:
            self.annotations = json.load(f)

        self.images = self.annotations['images']
        self.caption_dict = {}
        for x in self.annotations['annotations']:
            id = x['image_id']
            if id not in self.caption_dict:
                self.caption_dict[id] = []
            
            self.caption_dict[id].append(x['caption'])

        self.prompts = [
            "Describe the image concisely.",
            "Provide a brief description of the given image.",
            "Offer a succinct explanation of the picture presented.",
            "Summarize the visual content of the image.",
            "Give a short and clear explanation of the subsequent image.",
            "Share a concise interpretation of the image provided.",
            "Present a compact description of the photo’s key features.",
            "Relay a brief, clear account of the picture shown.",
            "Render a clear and concise summary of the photo.",
            "Write a terse but informative summary of the picture.",
            "Create a compact narrative representing the image presented.",
        ]

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        x = self.images[idx]
        id = x['id']
        
        question = ''
        if id in self.caption_dict:
            answers = [{'answer': x, 'confidence': 'yes', 'answer_id': i + 1} for i, x in enumerate(self.caption_dict[id])]
        else:
            answers = None
        image_path = os.path.join(self.img_root, x['file_name'])
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)
        
        return_dict = {
            'question': question,
            'question_id': id,
            'answers': answers,
            'image': image,
            'captions': self.caption_dict[id]
        }
        return return_dict

class LlavaDataset:
    def __init__(
        self, 
        annotations_file='/mnt/bn/us-aigc-temp/henry/LLaVA/playground/data/llava_v1_5_mix665k_filtered.json', 
        img_dir='/mnt/bn/us-aigc-temp/henry/LLaVA/playground/data/', 
        tokenizer=None, 
        transform=None, 
        max_length=64, 
        res=256, 
        add_vqav2=False, 
        add_okvqa=False, 
        add_vizwiz=False,
        add_pope=False,
        add_textvqa=False,
        add_gqa=False,
        add_coco=False,
        no_llava=False,
        split='train',
        return_kwargs=False,
        reverse=True,
        ):
        if no_llava:
            self.img_labels = []
        else:
            with open(annotations_file, 'r') as f:
                self.img_labels = json.load(f)
        
        self.img_dir = img_dir
        if transform is None:
            transform = v2.Compose([
                CenterCrop(),
                v2.Resize(size=(res, res), antialias=True),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[.5]*3, std=[0.5]*3),
            ])
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        self.return_kwargs = return_kwargs
        self.reverse = reverse

        self.datasets = []
        self.dataset_dict = {}

        if add_vqav2:
            os.environ['HF_HOME'] = '/mnt/bn/us-aigc-temp/henry/'
            assert os.environ['HF_HOME'] == '/mnt/bn/us-aigc-temp/henry/'
            vqav2 = load_dataset("HuggingFaceM4/VQAv2", trust_remote_code=True, cache_dir='/mnt/bn/us-aigc-temp/henry/test/')[split]
            self.datasets.append(vqav2)
            self.dataset_dict['vqav2'] = vqav2

        if add_okvqa:
            okvqa = OKVQADataset(split=split)
            self.datasets.append(okvqa)
            self.dataset_dict['okvqa'] = okvqa

        if add_vizwiz:
            vizwiz = VizWizDataset(split=split)
            self.datasets.append(vizwiz)
            self.dataset_dict['vizwiz'] = vizwiz

        if add_pope:
            pope = POPEDataset(split='val')
            self.datasets.append(pope)
            self.dataset_dict['pope'] = pope

        if add_textvqa:
            textvqa = TextVQADataset(split=split)
            self.datasets.append(textvqa)
            self.dataset_dict['textvqa'] = textvqa

        if add_gqa:
            gqa = GQADataset(split=split)
            self.datasets.append(gqa)
            self.dataset_dict['gqa'] = gqa

        if add_coco:
            coco = COCODataset(split=split)
            self.datasets.append(coco)
            self.dataset_dict['coco'] = coco

    def __len__(self):
        return len(self.img_labels) + sum([len(dataset) for dataset in self.datasets])
    
    def get_dataset_idx(self, idx, dataset=None):
        if dataset is not None:
            return self.dataset_dict[dataset], idx

        if idx < len(self.img_labels):
            return 'llava', idx
        
        idx -= len(self.img_labels)
        
        for dataset in self.datasets:
            if idx < len(dataset):
                return dataset, idx
            else:
                idx -= len(dataset)

    def get_image_and_text(self, dataset, idx):
        if dataset == 'llava':
            assert 'image' in self.img_labels[idx], f"{idx}, {self.img_labels[idx]}"
            img_path = os.path.join(self.img_dir, self.img_labels[idx]['image'])
            image = read_image(img_path)

            # sample random label
            conv_idx = 2 * np.random.randint(len(self.img_labels[idx]['conversations']) // 2)
            conv_idx = 0
            human = self.img_labels[idx]['conversations'][conv_idx]['value'].replace('<image>', '')
            gpt = self.img_labels[idx]['conversations'][conv_idx + 1]['value']
            question_id = idx
            answers = gpt
        else:
            x = dataset[idx]
            image = x['image']
            image = v2.ToTensor()(image)
            human = x['question']
            question_id = x['question_id']
            if isinstance(dataset, VizWizDataset):
                human += ' Answer the question using a single word or phrase. Reply unanswerable if information is not present in the image.'
            elif isinstance(dataset, COCODataset):
                idx = np.random.randint(len(dataset.prompts) // 2)
                human += f' {dataset.prompts[idx]}'
            else:
                human += ' Answer the question using a single word or phrase.'

            if x['answers'] is not None:
                answers = x['answers']
                gpt = answers[np.random.choice(len(answers))]['answer']
            else:
                answers = None
                gpt = None

        return human, gpt, image, answers, question_id

    def __getitem__(self, idx):
        return self._getitem(idx)


    def _getitem(self, idx, dataset=None):
        dataset, idx = self.get_dataset_idx(idx, dataset=dataset)
        human, gpt, image, answers, question_id = self.get_image_and_text(dataset, idx)

        try:
            image = self.transform(image)
        except Exception as e:
            print("IMAGE SHAPE!", image.shape)
            image = self.transform(image[:3])
        
        if self.tokenizer is not None:
            if self.reverse:
                postamble = f' Q: {human}'
                
                postamble = self.tokenizer(postamble, max_length=self.max_length, truncation=True, return_tensors='pt')
                preamble = self.tokenizer(gpt, max_length=self.max_length, truncation=True, return_tensors='pt')

                input_ids = [preamble['input_ids'][:, :-1]]
                attention_mask = [preamble['attention_mask'][:, :-1]]
                label_mask = [torch.ones_like(preamble['attention_mask'][:, :-1])]
        
                pad_length = self.max_length - (len(postamble['input_ids'][0]) + len(preamble['input_ids'][0]) - 1)
                if pad_length > 0:
                    mask_pad = torch.ones(size=(1, pad_length), dtype=int)
                    input_ids.append(mask_pad * self.tokenizer.pad_token_id)
                    attention_mask.append(mask_pad)
                    label_mask.append(mask_pad)
                
                input_ids.append(postamble['input_ids'])
                attention_mask.append(postamble['attention_mask'])
                label_mask.append(torch.zeros_like(postamble['attention_mask']))

                input_ids = torch.cat(input_ids, dim=1)[0, :self.max_length]
                attention_mask = torch.cat(attention_mask, dim=1)[0, :self.max_length]
                label_mask = torch.cat(label_mask, dim=1)[0, :self.max_length]
                
                text = {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'label_mask': label_mask
                }
            else:
                preamble = f'Q: {human} A: '

                preamble = self.tokenizer(preamble, max_length=self.max_length, truncation=True, return_tensors='pt')
                gpt = self.tokenizer(gpt, max_length=self.max_length, truncation=True, return_tensors='pt')
                input_ids = torch.cat([preamble['input_ids'][:, :-1], gpt['input_ids']], dim=1)[0, :self.max_length]
                attention_mask = torch.cat([preamble['attention_mask'][:, :-1], gpt['attention_mask']], dim=1)[0, :self.max_length]
                label_mask = torch.cat([torch.zeros_like(preamble['attention_mask'][:, :-1]), torch.ones_like(gpt['attention_mask'])], dim=1)[0, :self.max_length]

                assert len(input_ids) == len(attention_mask) == len(label_mask)
                pad_length = self.max_length - len(input_ids)
                if pad_length:
                    mask_pad = torch.ones(size=(pad_length,), dtype=int)
                    input_ids = torch.cat([input_ids, mask_pad * self.tokenizer.pad_token_id])
                    attention_mask = torch.cat([attention_mask, mask_pad])
                    label_mask = torch.cat([label_mask, mask_pad])
                
                text = {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'label_mask': label_mask
                }
        else:
            text = (human, gpt)

        if self.return_kwargs:
            kwargs = {
                'answers': answers,
                'question_id': question_id,
                'question': human,
                'answer': gpt
            }
            return image, text, kwargs

        return image, text
