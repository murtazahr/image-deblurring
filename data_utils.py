import glob as gb
import os
import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader


def normalization(x):
    """Normalize input to [-1, 1] range"""
    return x / 127.5 - 1


def format_image(image_path, size):
    """Format and split image into full and blur versions"""
    image = Image.open(image_path)
    # slice image into full and blur images
    image_full = image.crop((0, 0, image.size[0] // 2, image.size[1]))
    image_blur = image.crop((image.size[0] // 2, 0, image.size[0], image.size[1]))

    image_full = image_full.resize((size, size), Image.Resampling.LANCZOS)
    image_blur = image_blur.resize((size, size), Image.Resampling.LANCZOS)

    # Convert to numpy arrays
    return np.array(image_full), np.array(image_blur)


def build_hdf5(jpeg_dir, size=256):
    """Build HDF5 dataset from image directory"""
    hdf5_file = os.path.join('data', 'data.h5')
    with h5py.File(hdf5_file, 'w') as f:
        for data_type in tqdm(['train', 'test'], desc='create HDF5 dataset from images'):
            data_path = jpeg_dir + '/%s/*.jpg' % data_type
            images_path = gb.glob(data_path)
            data_full = []
            data_blur = []
            for image_path in images_path:
                image_full, image_blur = format_image(image_path, size)
                data_full.append(image_full)
                data_blur.append(image_blur)

            f.create_dataset('%s_data_full' % data_type, data=data_full)
            f.create_dataset('%s_data_blur' % data_type, data=data_blur)


class ImageDataset(Dataset):
    """Dataset class for loading images"""

    def __init__(self, data_type='train'):
        with h5py.File('data/data.h5', 'r') as f:
            self.data_full = f['%s_data_full' % data_type][:].astype(np.float32)
            self.data_blur = f['%s_data_blur' % data_type][:].astype(np.float32)

        self.data_full = normalization(self.data_full)
        self.data_blur = normalization(self.data_blur)

        # Convert to PyTorch tensors and adjust channels
        self.data_full = torch.FloatTensor(self.data_full).permute(0, 3, 1, 2)
        self.data_blur = torch.FloatTensor(self.data_blur).permute(0, 3, 1, 2)

    def __len__(self):
        return len(self.data_full)

    def __getitem__(self, idx):
        return self.data_blur[idx], self.data_full[idx]


def get_loader(data_type, batch_size, shuffle=True, num_workers=4):
    """Create data loader"""
    dataset = ImageDataset(data_type)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def generate_image(full, blur, generated, path, epoch=None, index=None):
    """Generate and save comparison images"""
    # Convert from PyTorch tensors to numpy arrays
    if torch.is_tensor(full):
        full = full.cpu().detach().numpy().transpose(0, 2, 3, 1)
    if torch.is_tensor(blur):
        blur = blur.cpu().detach().numpy().transpose(0, 2, 3, 1)
    if torch.is_tensor(generated):
        generated = generated.cpu().detach().numpy().transpose(0, 2, 3, 1)

    full = full * 127.5 + 127.5
    blur = blur * 127.5 + 127.5
    generated = generated * 127.5 + 127.5

    for i in range(generated.shape[0]):
        image_full = full[i]
        image_blur = blur[i]
        image_generated = generated[i]
        image = np.concatenate((image_full, image_blur, image_generated), axis=1)
        if (epoch is not None) and (index is not None):
            Image.fromarray(image.astype(np.uint8)).save(path + f"{epoch + 1}_{index + 1}.png")
        else:
            Image.fromarray(image.astype(np.uint8)).save(path + f"{i}.png")


if __name__ == '__main__':
    format_image('data/small/test/301.jpg', size=256)
    build_hdf5('data/small')
    loader = get_loader('train', batch_size=4)
    for blur, full in loader:
        print(f"Blur shape: {blur.shape}, Full shape: {full.shape}")
        break
