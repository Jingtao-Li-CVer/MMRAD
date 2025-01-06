import os
import json
from img_io import read_img, write_img
import numpy as np
import logging
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from datasets import register
import cv2


@register('image-folder')
class ImageFolder(Dataset):
    def __init__(self, path,  split_file=None, split_key=None, first_k=None, size=None,
                 repeat=1, cache='none', mask=False):
        self.repeat = repeat
        self.cache = cache
        self.path = path
        self.Train = False
        self.split_key = split_key

        self.size = size
        self.mask = mask

        if split_file is None:
            filenames = sorted(os.listdir(path))
        else:
            with open(split_file, 'r') as f:
                filenames = json.load(f)[split_key]
        if first_k is not None:
            filenames = filenames[:first_k]

        self.files = []

        for filename in filenames:
            file = os.path.join(path, filename)
            self.append_file(file)

    def append_file(self, file):
        if self.cache == 'none':
            self.files.append(file)
        elif self.cache == 'in_memory':
            self.files.append(self.img_process(file))

    def __len__(self):
        return len(self.files) * self.repeat

    def __getitem__(self, idx):
        x = self.files[idx % len(self.files)]
        file_name = x.split('/')[-1].split('.')[0]
        if self.cache == 'none':
            return self.img_process(x), file_name
        elif self.cache == 'in_memory':
            return x

    def img_process(self, file):
        return read_img(file)


@register('paired-image-folders')
class PairedImageFolders(Dataset):

    def __init__(self, root_path_1, root_path_2, **kwargs):
        self.dataset_1 = ImageFolder(root_path_1, **kwargs)
        self.dataset_2 = ImageFolder(root_path_2, **kwargs, mask=True)
        self.FP_transform = A.Compose([
            A.ToFloat(max_value=255, p=1),
            A.Normalize(mean=[0.38812142, 0.28388125, 0.15787678],
                        std=[0.23149024, 0.16714649, 0.10265642]),
            A.Resize(height=1024, width=1024, p=1, interpolation=cv2.INTER_NEAREST),
            ToTensorV2(p=1),
        ])

    def __len__(self):
        return len(self.dataset_1)

    def __getitem__(self, idx):
        transformed = self.FP_transform(image=self.dataset_1[idx][0], mask=self.dataset_2[idx][0])
        return transformed['image'], transformed['mask'].unsqueeze(0).float()/255, self.dataset_1[idx][1]


@register('FP-OCT-Medical')
class FP_OCt_Medical(Dataset):

    def __init__(self, img_dir, is_training):
        self.img_dir = img_dir
        self.FP_transform = A.Compose([
            A.ToFloat(max_value=255, p=1),
            A.Normalize(mean=[0.38812142, 0.28388125, 0.15787678],
                        std=[0.23149024, 0.16714649, 0.10265642], max_pixel_value=1.0),
            A.Resize(height=1024, width=1024, p=1, interpolation=cv2.INTER_NEAREST),
            ToTensorV2(p=1),
        ])
        
        self.OCT_transform = A.Compose([
            A.ToFloat(max_value=255, p=1),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225], max_pixel_value=1.0),
            A.Resize(height=1024, width=1024, p=1, interpolation=cv2.INTER_NEAREST),
            ToTensorV2(p=1),
        ])

        self.shift_transform = A.Compose([A.ShiftScaleRotate(shift_limit=0.0, p=1, interpolation=cv2.INTER_NEAREST), A.IAAPiecewiseAffine (scale=(0.03, 0.05), order=0, p=1)])
        self.anomaly_augmentation = A.OneOf([A.ShiftScaleRotate(p=1.0), A.ColorJitter(p=1.0), \
                                          A.ChannelShuffle(p=0.6), A.HueSaturationValue(p=0.6)])

        self.is_training = is_training
        self.user_files = os.listdir(img_dir)

        print(f'Creating dataset with {len(self.user_files)} examples')

    def __len__(self):
        return len(self.user_files)

    def __getitem__(self, i):
        folder_name = self.user_files[i]
        user_folder = os.path.join(self.img_dir, folder_name)
        FP_img_path = os.path.join(user_folder, 'FP.jpg')
        FP_label_path = os.path.join(user_folder, 'FP_label.png')
        OCT_img_path = os.path.join(user_folder, 'OCT.tif')
        OCT_label_path = os.path.join(user_folder, 'OCT_label.png')
        

        if self.is_training:
            # Generating simulated anomaly localization samples for fundus images
            FP_img = read_img(FP_img_path)
            FP_mask =  np.zeros((FP_img.shape[0], FP_img.shape[1]))

            max_height_width = int(np.sqrt(FP_img.shape[0]*FP_img.shape[1]*0.4))
            min_height_width = int(np.sqrt(FP_img.shape[0]*FP_img.shape[1]*0.0225))
            self.FP_CoarseDropout_transform = A.CoarseDropout(max_holes=3 , min_holes=2, max_height=max_height_width, max_width=max_height_width, min_height=min_height_width, min_width=min_height_width, fill_value=0, mask_fill_value=1, p=0.7)

            FP_sample = self.FP_CoarseDropout_transform(image=FP_img, mask = FP_mask)
            FP_img2 = self.anomaly_augmentation(image=FP_img)['image']
            locs = np.where(FP_sample['mask'] == 1)
            FP_sample['image'][locs[0], locs[1], :] = FP_img2[locs[0], locs[1], :] 
            FP_sample_shift = self.shift_transform(image=FP_sample['image'], mask = FP_sample['mask'])
            locs_shift = np.where(FP_sample_shift['mask'] == 1)
            FP_img[locs_shift[0], locs_shift[1], :] = FP_sample_shift['image'][locs_shift[0], locs_shift[1], :]
            sample_FP = self.FP_transform(image=FP_img, mask = FP_sample_shift['mask'])

            # Generating simulated anomaly localization samples for OCT images
            OCT_img = read_img(OCT_img_path)
            # Removing user private information 
            if OCT_img.shape[0] == 868 and OCT_img.shape[1] == 1536:
                OCT_img = OCT_img[0:769, 769:,:]
            elif OCT_img.shape[0] == 596 and OCT_img.shape[1] == 1264:
                OCT_img = OCT_img[0:495, 495:,:]

            OCT_mask =  np.zeros((OCT_img.shape[0], OCT_img.shape[1]))
            sample_OCT = self.OCT_transform(image=OCT_img, mask=OCT_mask)

            max_height_width = int(np.sqrt(OCT_img.shape[0]*OCT_img.shape[1]*0.4))
            min_height_width = int(np.sqrt(OCT_img.shape[0]*OCT_img.shape[1]*0.0225))
            self.OCT_CoarseDropout_transform = A.CoarseDropout(max_holes=3 , min_holes=2, max_height=max_height_width, max_width=max_height_width, min_height=min_height_width, min_width=min_height_width, fill_value=1, mask_fill_value=1, p=0.3)

            OCT_sample = self.OCT_CoarseDropout_transform(image=OCT_img, mask = OCT_mask)
            OCT_img2 = self.anomaly_augmentation(image=OCT_img)['image']
            locs = np.where(OCT_sample['mask'] == 1)
            OCT_sample['image'][locs[0], locs[1], :] = OCT_img2[locs[0], locs[1], :] 
            OCT_sample_shift = self.shift_transform(image=OCT_sample['image'], mask = OCT_sample['mask'])
            locs_shift = np.where(OCT_sample_shift['mask'] == 1)
            OCT_img[locs_shift[0], locs_shift[1], :] = OCT_sample_shift['image'][locs_shift[0], locs_shift[1], :]

            OCT_sample = self.OCT_transform(image=OCT_img, mask = OCT_sample_shift['mask'])

            # Generated anomaly detection samples
            image_level_detection_gt = torch.max(torch.max(sample_FP['mask'].flatten()), torch.max(OCT_sample['mask'].flatten()))

            return sample_FP['image'].float(), sample_FP['mask'].unsqueeze(0), OCT_sample['image'].float(), \
                OCT_sample['mask'].unsqueeze(0), image_level_detection_gt
        else:
            try:
                FP_img = read_img(FP_img_path)
            except AttributeError:
                print(1)

            if not os.path.exists(FP_label_path):
                FP_label = np.zeros(FP_img.shape)
            else:
                FP_label = read_img(FP_label_path)

            if len(FP_label.shape) > 2:
                FP_label = FP_label[:,:,0]

            if not os.path.exists(OCT_img_path):
                OCT_img_path = OCT_img_path.replace(".tif", ".jpg")
            OCT_img = read_img(OCT_img_path)
            if OCT_img.shape[0] == 868 and OCT_img.shape[1] == 1536:
                OCT_img = OCT_img[0:769, 769:,:]
            elif OCT_img.shape[0] == 596 and OCT_img.shape[1] == 1264:
                OCT_img = OCT_img[0:495, 495:,:]
            
            if OCT_img.shape[2] > 3:
                OCT_img = OCT_img[:,:,:3]

            if not os.path.exists(OCT_label_path):
                OCT_label = np.zeros(OCT_img.shape)
            else:
                OCT_label = read_img(OCT_label_path)
                if OCT_label.shape[0] == 868 and OCT_label.shape[1] == 1536:
                    OCT_label = OCT_label[0:769, 769:]
                elif OCT_label.shape[0] == 596 and OCT_label.shape[1] == 1264:
                    OCT_label = OCT_label[0:495, 495:]

            if len(OCT_label.shape) > 2:
                OCT_label = OCT_label[:,:,0]

            FP_label[FP_label >= 1] = 1
            OCT_label[OCT_label >= 1] = 1

            sample_FP = self.FP_transform(image=FP_img, mask=FP_label)
            sample_OCT = self.OCT_transform(image=OCT_img, mask=OCT_label)

            image_level_detection_gt = torch.max(torch.max(sample_FP['mask'].flatten()), torch.max(sample_OCT['mask'].flatten()))
            
            return sample_FP['image'].float(), sample_FP['mask'].unsqueeze(0), sample_OCT['image'].float(), \
                sample_OCT['mask'].unsqueeze(0), image_level_detection_gt