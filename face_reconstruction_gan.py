import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ContextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 2, 1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class TANGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 3, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
            nn.Conv2d(256, 1, 4, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


class FaceReconstructionGAN:
    def __init__(self, device="cpu"):
        self.device = device

        self.context_encoder = ContextEncoder().to(device)
        self.tan_generator = TANGenerator().to(device)
        self.discriminator = Discriminator().to(device)

        self.identity_model = models.resnet50(weights="DEFAULT").to(device)
        self.identity_model.fc = nn.Identity()
        self.identity_model.eval()
        for p in self.identity_model.parameters():
            p.requires_grad = False

        self.l1 = nn.L1Loss()
        self.bce = nn.BCELoss()
        self.mse = nn.MSELoss()

        self.g_optimizer = torch.optim.Adam(
            list(self.context_encoder.parameters()) +
            list(self.tan_generator.parameters()),
            lr=2e-4, betas=(0.5, 0.999)
        )

        self.d_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=5e-5, betas=(0.5, 0.999)
        )

    def forward_generators(self, masked):
        ce_out = self.context_encoder(masked)
        tan_out = self.tan_generator(masked)
        return torch.clamp(ce_out + tan_out, -1, 1)

    def train_step(self, real, masked, mask):

        self.d_optimizer.zero_grad()

        fake = self.forward_generators(masked)

        real_pred = self.discriminator(real)
        fake_pred = self.discriminator(fake.detach())

        real_labels = torch.ones_like(real_pred)
        fake_labels = torch.zeros_like(fake_pred)

        d_loss = (self.bce(real_pred, real_labels) +
                  self.bce(fake_pred, fake_labels)) / 2

        d_loss.backward()
        self.d_optimizer.step()

        self.g_optimizer.zero_grad()

        fake = self.forward_generators(masked)
        fake_pred = self.discriminator(fake)

        l_adv = self.bce(fake_pred, real_labels)
        l_cc = self.l1(fake * (1 - mask), real * (1 - mask))

        with torch.no_grad():
            emb_real = self.identity_model(real)
        emb_fake = self.identity_model(fake)

        l_id = self.mse(F.normalize(emb_fake),
                        F.normalize(emb_real))

        g_loss = l_cc + 0.1 * l_adv + 0.5 * l_id
        g_loss.backward()
        self.g_optimizer.step()

        return {
            "d_loss": d_loss.item(),
            "g_loss": g_loss.item(),
            "l_adv": l_adv.item(),
            "l_cc": l_cc.item(),
            "l_id": l_id.item()
        }

    def reconstruct(self, masked):
        self.context_encoder.eval()
        self.tan_generator.eval()
        with torch.no_grad():
            return self.forward_generators(masked)
