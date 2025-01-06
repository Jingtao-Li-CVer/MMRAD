from torch.utils.data import Dataset
from torchvision import transforms
from datasets import register


def to_mask(mask):
    return transforms.ToTensor()(
        transforms.Grayscale(num_output_channels=1)(
            transforms.ToPILImage()(mask)))

def resize_fn(img, size):
    return transforms.ToTensor()(
        transforms.Resize(size)(
            transforms.ToPILImage()(img)))


@register('val')
class ValDataset(Dataset):
    def __init__(self, dataset, inp_size=None, augment=False):
        self.dataset = dataset
        self.inp_size = inp_size
        self.augment = augment

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        fp_img, fp_mask, oct_img, oct_mask, image_level_detection_gt = self.dataset[idx]

        return {
            'fp_inp': (fp_img),
            'fp_gt': (fp_mask),
            'oct_inp': (oct_img),
            'oct_gt': (oct_mask),
            'image_level_detection_gt' : image_level_detection_gt
        }



@register('train')
class TrainDataset(Dataset):
    def __init__(self, dataset, size_min=None, size_max=None, inp_size=None,
                 augment=False, gt_resize=None):
        self.dataset = dataset
        self.size_min = size_min
        if size_max is None:
            size_max = size_min
        self.size_max = size_max
        self.augment = augment
        self.gt_resize = gt_resize
        self.inp_size = inp_size


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        fp_img, simulated_fp_mask, oct_img, simulated_oct_mask, image_level_detection_gt = self.dataset[idx]

        return {
            'fp_inp': (fp_img),
            'fp_gt': (simulated_fp_mask),
            'oct_inp': (oct_img),
            'oct_gt': (simulated_oct_mask),
            'image_level_detection_gt' : image_level_detection_gt
        }