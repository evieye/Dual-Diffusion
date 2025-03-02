# import os
# import random
# from PIL import Image
# import json
# import numpy as np
# import torch
# from torchvision.datasets.folder import default_loader, IMG_EXTENSIONS
# from torch.utils.data import Dataset
# from diffusers.utils.torch_utils import randn_tensor
# from torchvision import transforms as T
# from torchvision.transforms.functional import InterpolationMode


# from .text_data import get_logger

# LOGGER = get_logger(__name__)


# def load_json(file_path):
#     with open(file_path, 'r') as f:
#         meta_data = json.load(f)

#     return meta_data


# class FlexibleInternalData(Dataset):
#     def __init__(self,
#                  roots,       # a list of root that has the same length as image_list_json_lst
#                  json_lst=None,   # a list of json file, each json file contains a list of dict, each dict contains the info of an image and its caption
#                  resolution=256,
#                  org_caption_key=None,
#                  re_caption_key=None,
#                  tokenizer=None, # do tokenizing on the fly
#                  max_length=None,
#                  **kwargs):


#         self.resolution = resolution

#         self.meta_data_clean = []
#         self.img_samples = []
#         self.org_captions = []
#         self.re_captions = []

#         self.org_caption_key = org_caption_key
#         self.re_caption_key = re_caption_key
#         self.tokenizer = tokenizer
#         if self.tokenizer is not None:
#             assert max_length is not None, "max_length must be provided when tokenizer is not None"
#             self.max_length = max_length

#         self.interpolate_model = InterpolationMode.BICUBIC
#         self.transform = T.Compose([
#                 T.Lambda(lambda img: img.convert('RGB')),
#                 T.Resize(self.resolution, interpolation=self.interpolate_model),  # Image.BICUBIC
#                 T.CenterCrop(self.resolution),
#                 T.ToTensor(),
#                 T.Normalize([.5], [.5]),
#             ])

#         if not json_lst:
#             # default, use meta data json
#             json_lst = [os.path.join(root, 'meta_data.json') for root in roots]

#         if org_caption_key is None:
#             org_caption_key = 'caption'

#         i = 0

#         for root, json_file in zip(roots, json_lst):

#             meta_data = load_json(os.path.join(root, json_file))
#             LOGGER.info(f"{json_file} data volume: {len(meta_data)}")
#             # enfore a reasonable aspect ratio also non-empty-caption
#             meta_data_clean = [item for item in meta_data if item['ratio'] <= 4.5 and min(item['height'], item['width']) >= 256 and len(item[org_caption_key]) > 5]
#             self.meta_data_clean.extend(meta_data_clean)
#             self.img_samples.extend([
#                 os.path.join(root, item['image_path']) for item in meta_data_clean
#             ])
            
#             self.org_captions.extend([
#                 item[org_caption_key] for item in meta_data_clean
#             ])
#             if re_caption_key is not None:
#                 self.re_captions.extend([
#                     item[re_caption_key] for item in meta_data_clean
#                 ])
#             i += 1
#         LOGGER.info(f"Total data volume: {len(self.meta_data_clean)}")

#         self.loader = default_loader

#     def getdata(self, index):
#         img_path = self.img_samples[index]
#         origin_caption = self.org_captions[index]
#         if self.re_caption_key is not None:
#             re_caption = self.re_captions[index]
#         else:
#             re_caption = origin_caption

#         data_info = {}
#         ori_h, ori_w = self.meta_data_clean[index]['height'], self.meta_data_clean[index]['width']
        
#         raw_img = self.loader(img_path)
#         h, w = (raw_img.size[1], raw_img.size[0])
#         assert h, w == (ori_h, ori_w)

#         data_info['img_hw'] = torch.tensor([ori_h, ori_w], dtype=torch.float32)

#         img_tsr = self.transform(raw_img)

#         if self.tokenizer is not None:
#             # tokenize
#             org_caption_token_info = self.tokenizer(
#                                         origin_caption,
#                                         padding="max_length",
#                                         truncation=True,
#                                         max_length=self.max_length,
#                                         return_tensors="pt",
#                                     )
            
#             if re_caption is not None:
#                 re_caption_token_info = self.tokenizer(
#                                         re_caption,
#                                         padding="max_length",
#                                         truncation=True,
#                                         max_length=self.max_length,
#                                         return_tensors="pt",
#                                     )
#             else:
#                 re_caption_token_info = org_caption_token_info

#             return img_tsr, org_caption_token_info, re_caption_token_info
        
#         else:
#             return img_tsr, origin_caption, re_caption
        
#     def __len__(self):
#         return len(self.img_samples)

#     def __getitem__(self, idx):
#         for _ in range(20):
#             try:
#                 data = self.getdata(idx)
#                 return data
#             except Exception as e:
#                 print(f"Error details: {str(e)}")
#                 idx = random.choice(self.__len__) # get a closest
#         raise RuntimeError('Too many bad data.')

import os
import random
from PIL import Image
import json
import numpy as np
import torch
from torchvision.datasets.folder import default_loader, IMG_EXTENSIONS
from torch.utils.data import Dataset
from diffusers.utils.torch_utils import randn_tensor
from torchvision import transforms as T
from torchvision.transforms.functional import InterpolationMode


from .text_data import get_logger

LOGGER = get_logger(__name__)


def load_json(file_path):
    with open(file_path, 'r') as f:
        meta_data = json.load(f)

    return meta_data


class FlexibleInternalData(Dataset):
    def __init__(self,
                 roots,       # a list of root that has the same length as image_list_json_lst
                 json_lst=None,   # a list of json file, each json file contains a list of dict, each dict contains the info of an image and its caption
                 resolution=256,
                 org_caption_key=None,
                 re_caption_key=None,
                 tokenizer=None, # do tokenizing on the fly
                 max_length=None,
                 **kwargs):


        self.resolution = resolution

        self.meta_data_clean = []
        self.img_samples = []
        self.org_captions = []
        self.re_captions = []

        self.org_caption_key = org_caption_key
        self.re_caption_key = re_caption_key
        self.tokenizer = tokenizer
        if self.tokenizer is not None:
            assert max_length is not None, "max_length must be provided when tokenizer is not None"
            self.max_length = max_length

        self.interpolate_model = InterpolationMode.BICUBIC
        self.transform = T.Compose([
                T.Lambda(lambda img: img.convert('RGB')),
                T.Resize(self.resolution, interpolation=self.interpolate_model),  # Image.BICUBIC
                T.CenterCrop(self.resolution),
                T.ToTensor(),
                T.Normalize([.5], [.5]),
            ])

        if not json_lst:
            # default, use meta data json
            json_lst = [os.path.join(root, 'meta_data.json') for root in roots]

        if org_caption_key is None:
            org_caption_key = 'caption'

        if not isinstance(org_caption_key, list):
            org_caption_key = [org_caption_key] * len(json_lst)
        if re_caption_key and not isinstance(re_caption_key, list):
            re_caption_key = [re_caption_key] * len(json_lst)

        i = 0

        for root, json_file in zip(roots, json_lst):

            meta_data = load_json(os.path.join(root, json_file))
            LOGGER.info(f"{json_file} data volume: {len(meta_data)}")
            # enfore a reasonable aspect ratio also non-empty-caption
            meta_data_clean = [item for item in meta_data if item['ratio'] <= 4.0 and min(item['height'], item['width']) >= 256]
            self.meta_data_clean.extend(meta_data_clean)
            self.img_samples.extend([
                os.path.join(root, item['image_path']) for item in meta_data_clean
            ])
            
            self.org_captions.extend([
                item[org_caption_key[i]] for item in meta_data_clean
            ])
            if re_caption_key is not None:
                self.re_captions.extend([
                    item[re_caption_key[i]] for item in meta_data_clean
                ])
            i += 1
        LOGGER.info(f"Total data volume: {len(self.meta_data_clean)}")

        self.loader = default_loader

    def getdata(self, index):
        img_path = self.img_samples[index]
        origin_caption = self.org_captions[index]
        if self.re_caption_key is not None:
            re_caption = self.re_captions[index]
        else:
            re_caption = origin_caption

        data_info = {}
        ori_h, ori_w = self.meta_data_clean[index]['height'], self.meta_data_clean[index]['width']
        
        raw_img = self.loader(img_path)
        h, w = (raw_img.size[1], raw_img.size[0])
        assert h, w == (ori_h, ori_w)

        data_info['img_hw'] = torch.tensor([ori_h, ori_w], dtype=torch.float32)

        img_tsr = self.transform(raw_img)

        if self.tokenizer is not None:
            # tokenize
            org_caption_token_info = self.tokenizer(
                                        origin_caption,
                                        padding="max_length",
                                        truncation=True,
                                        max_length=self.max_length,
                                        return_tensors="pt",
                                    )
            
            if re_caption is not None:
                re_caption_token_info = self.tokenizer(
                                        re_caption,
                                        padding="max_length",
                                        truncation=True,
                                        max_length=self.max_length,
                                        return_tensors="pt",
                                    )
            else:
                re_caption_token_info = org_caption_token_info

            return img_tsr, org_caption_token_info, re_caption_token_info
        
        else:
            return img_tsr, origin_caption, re_caption
        
    def __len__(self):
        return len(self.img_samples)

    def __getitem__(self, idx):
        for _ in range(20):
            try:
                data = self.getdata(idx)
                return data
            except Exception as e:
                print(f"Error details: {str(e)}")
                idx = random.choice(self.__len__) # get a closest
        raise RuntimeError('Too many bad data.')