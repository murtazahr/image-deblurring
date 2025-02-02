import torch
import torch.nn as nn
import torchvision.models as models

# Constants
K_1 = 145
K_2 = 170


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg16(pretrained=True)
        # Get features up to block3_conv3 (equivalent to Keras implementation)
        self.feature_extractor = nn.Sequential(*list(vgg.features.children())[:16])
        # Freeze the network
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.feature_extractor.eval()

    def forward(self, y_true, y_pred):
        features_true = self.feature_extractor(y_true)
        features_pred = self.feature_extractor(y_pred)
        return torch.mean((features_true - features_pred) ** 2)


class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()
        self.perceptual_loss = PerceptualLoss()
        self.l1_loss = nn.L1Loss()

    def forward(self, y_true, y_pred):
        return K_1 * self.perceptual_loss(y_true, y_pred) + K_2 * self.l1_loss(y_pred, y_true)


class AdversarialLoss(nn.Module):
    def __init__(self):
        super(AdversarialLoss, self).__init__()

    @staticmethod
    def forward(y_pred):
        # Implementation of -log(y_pred)
        return -torch.mean(torch.log(y_pred + 1e-10))  # Add small epsilon to prevent log(0)


if __name__ == '__main__':
    # Test the losses
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gen_loss = GeneratorLoss().to(device)
    adv_loss = AdversarialLoss().to(device)

    # Create dummy data
    y_true = torch.randn(1, 3, 256, 256).to(device)
    y_pred = torch.randn(1, 3, 256, 256).to(device)

    # Test generator loss
    loss = gen_loss(y_true, y_pred)
    print(f"Generator Loss: {loss.item()}")

    # Test adversarial loss
    pred = torch.sigmoid(torch.randn(1, 1)).to(device)
    loss = adv_loss(pred)
    print(f"Adversarial Loss: {loss.item()}")
