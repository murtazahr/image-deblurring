import torch
import torch.nn as nn
from torch.optim import Adam
import glob as gb
import numpy as np
from PIL import Image
from tqdm import tqdm

import data_utils
from model import Generator, Discriminator
from losses import GeneratorLoss, AdversarialLoss


def train(batch_size, epoch_num):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Get data loader
    train_loader = data_utils.get_loader('train', batch_size=batch_size)

    # Initialize models
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    # Initialize losses
    generator_criterion = GeneratorLoss().to(device)
    adversarial_criterion = AdversarialLoss().to(device)
    bce_criterion = nn.BCELoss().to(device)

    # Initialize optimizers
    g_optimizer = Adam(generator.parameters())
    d_optimizer = Adam(discriminator.parameters())

    for epoch in range(epoch_num):
        print(f'Epoch: {epoch + 1}/{epoch_num}')

        for index, (blur_images, full_images) in enumerate(tqdm(train_loader)):
            # Move data to device
            blur_images = blur_images.to(device)
            full_images = full_images.to(device)

            batch_size = blur_images.size(0)

            # Ground truths
            valid = torch.ones((batch_size, 1), requires_grad=False).to(device)
            fake = torch.zeros((batch_size, 1), requires_grad=False).to(device)

            # -----------------
            #  Train Generator
            # -----------------
            g_optimizer.zero_grad()

            # Generate images
            generated_images = generator(blur_images)

            # Calculate generator loss
            g_loss = generator_criterion(full_images, generated_images)

            # Calculate adversarial loss
            pred_generated = discriminator(generated_images)
            g_adv_loss = adversarial_criterion(pred_generated)

            # Total generator loss
            g_total_loss = g_loss + g_adv_loss
            g_total_loss.backward()
            g_optimizer.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------
            d_optimizer.zero_grad()

            # Real loss
            pred_real = discriminator(full_images)
            real_loss = bce_criterion(pred_real, valid)

            # Fake loss
            pred_fake = discriminator(generated_images.detach())
            fake_loss = bce_criterion(pred_fake, fake)

            # Total discriminator loss
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            d_optimizer.step()

            # Output training stats and save images
            if index % 30 == 0:
                print(f'[Batch {index}] [D loss: {d_loss.item():.4f}] [G loss: {g_total_loss.item():.4f}]')

                # Save generated images
                data_utils.generate_image(
                    full_images.cpu(),
                    blur_images.cpu(),
                    generated_images.cpu(),
                    'result/interim/',
                    epoch,
                    index
                )

                # Save model weights
                torch.save(generator.state_dict(), 'weight/generator.pth')
                torch.save(discriminator.state_dict(), 'weight/discriminator.pth')


def test(batch_size):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load test data
    test_loader = data_utils.get_loader('test', batch_size=batch_size, shuffle=False)

    # Initialize and load trained generator
    generator = Generator().to(device)
    generator.load_state_dict(torch.load('weight/generator.pth'))
    generator.eval()

    with torch.no_grad():
        for i, (blur_images, full_images) in enumerate(test_loader):
            blur_images = blur_images.to(device)
            full_images = full_images.to(device)

            # Generate images
            generated_images = generator(blur_images)

            # Save generated images
            data_utils.generate_image(
                full_images.cpu(),
                blur_images.cpu(),
                generated_images.cpu(),
                'result/finally/'
            )


def test_pictures(batch_size):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load and process test images
    data_path = 'data/test/*.jpeg'
    images_path = gb.glob(data_path)
    data_blur = []

    for image_path in images_path:
        image = Image.open(image_path)
        image = image.convert('RGB')  # Ensure RGB format
        image = np.array(image)
        data_blur.append(image)

    data_blur = np.array(data_blur).astype(np.float32)
    data_blur = data_utils.normalization(data_blur)

    # Convert to PyTorch tensor and adjust channels
    data_blur = torch.FloatTensor(data_blur).permute(0, 3, 1, 2).to(device)

    # Load trained generator
    generator = Generator().to(device)
    generator.load_state_dict(torch.load('weight/generator.pth'))
    generator.eval()

    with torch.no_grad():
        # Generate images in batches
        for i in range(0, len(data_blur), batch_size):
            batch = data_blur[i:i + batch_size]
            generated_images = generator(batch)

            # Convert back to numpy and save
            generated = generated_images.cpu().numpy()
            generated = generated.transpose(0, 2, 3, 1)
            generated = generated * 127.5 + 127.5

            for j, img in enumerate(generated):
                Image.fromarray(img.astype(np.uint8)).save(f'result/test/{i + j}.png')


if __name__ == '__main__':
    import os

    # Create necessary directories if they don't exist
    os.makedirs('data', exist_ok=True)
    os.makedirs('result/interim', exist_ok=True)
    os.makedirs('result/finally', exist_ok=True)
    os.makedirs('result/test', exist_ok=True)
    os.makedirs('weight', exist_ok=True)

    # Check if HDF5 file exists, if not create it
    if not os.path.exists('data/data.h5'):
        print("Creating HDF5 dataset from images...")
        data_utils.build_hdf5('data/small')
        print("HDF5 dataset created!")

    # Start training
    print("Starting training...")
    train(batch_size=2, epoch_num=10)
    test(batch_size=4)
    test_pictures(batch_size=2)
