import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, Callback
import albumentations as A
import numpy as np
from tqdm import tqdm
import pytorch_lightning as pl
from torch import optim
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, mean_absolute_error
import cv2
import torchbnn as bnn


class BayesianMobNet(pl.LightningModule):
    def __init__(self, k=21*2):
        super(BayesianMobNet, self).__init__()
        torch.cuda.empty_cache()

        # Load a pre-trained model
        self.model = models.mobilenet_v2(pretrained=True)
        self.model.classifier[1] = bnn.BayesLinear(prior_mu=0, prior_sigma=1e-5, in_features=self.model.last_channel, out_features=k)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        imgs, kpts = batch
        preds = self(imgs)

        # Compute projection errors and trust scores
        projection_errors = compute_projection_errors(preds, kpts)
        trust_scores = compute_trust(projection_errors)

        # Weighted MSE loss
        loss = self.weighted_mse_loss(preds, kpts, trust_scores)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, kpts = batch
        preds = self(imgs)

        # Compute projection errors and trust scores
        projection_errors = compute_projection_errors(preds, kpts)
        trust_scores = compute_trust(projection_errors)

        # Weighted MSE loss
        loss = self.weighted_mse_loss(preds, kpts, trust_scores)
        self.log('val_loss', loss)
        return loss

    def weighted_mse_loss(self, pred, target, weight):
        weight = torch.tensor(weight, device=pred.device)
        weight = weight.unsqueeze(0).unsqueeze(0).expand_as(target)
        loss = (weight * (pred - target) ** 2).mean()
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

def tensor_to_numpy(img):
    img = img.detach().cpu().numpy()
    img = np.transpose(img, (1, 2, 0))  # Convert from (C, H, W) to (H, W, C)
    img = img * 255  # Scale to [0, 255]
    img = img.astype(np.uint8)
    return img

def assess_image_clarity(img):
    img = tensor_to_numpy(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    variance_of_laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance_of_laplacian

def estimate_noise(img):
    img = tensor_to_numpy(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    noise = gray - cv2.medianBlur(gray, 3)
    noise_level = np.var(noise)
    return noise_level

def detect_occlusions(kpts, threshold=0.5):
    if kpts.ndim == 1:
        kpts = kpts.unsqueeze(0)
    occlusions = (kpts[:, 2] < threshold).sum().item()
    return occlusions

def evaluate_historical_performance(preds, targets):
    preds_flat = preds.view(-1, preds.shape[-1]).detach().cpu().numpy()
    targets_flat = targets.view(-1, targets.shape[-1]).detach().cpu().numpy()

    mse = mean_squared_error(targets_flat, preds_flat)
    mae = mean_absolute_error(targets_flat, preds_flat)

    return mse, mae

def compute_prediction_confidence(preds):
    confidence = preds.var(dim=0).mean().item()
    return confidence

def compute_cross_view_consistency(preds_views):
    # compute the variance of the predictions across views
    consistency = preds_views.var(dim=0).mean().item()
    return consistency

def compute_trust(metrics):
    mse, mae, confidence, consistency = metrics
    # Combine metrics to compute trust score; adjust weights as needed
    trust_score = mse + mae + confidence + consistency
    return trust_score

def compute_projection_errors(preds, kpts):
    # preds and kpts have shape (batch_size, num_keypoints * 2)
    preds = preds.view(preds.size(0), -1, 2)  # Reshape to (batch_size, num_keypoints, 2)
    kpts = kpts.view(kpts.size(0), -1, 2)  # Reshape to (batch_size, num_keypoints, 2)
    # Compute the Euclidean distance between predicted and true keypoints
    errors = torch.sqrt(torch.sum((preds - kpts) ** 2, dim=-1))  # Shape: (batch_size, num_keypoints)
    # Average error per image
    mean_errors = errors.mean(dim=-1)  # Shape: (batch_size,)

    return mean_errors

class TrustAwareBayesianMobNet(BayesianMobNet):
    def training_step(self, batch, batch_idx):
        imgs, kpts = batch
        preds = self(imgs)

        # Quality Assessment
        clarity_scores = [assess_image_clarity(img) for img in imgs]
        noise_levels = [estimate_noise(img) for img in imgs]
        occlusions = [detect_occlusions(kpt) for kpt in kpts]

        # Historical Performance
        mse, mae = evaluate_historical_performance(preds, kpts)

        # Prediction Confidence
        confidence = compute_prediction_confidence(preds)

        # Cross-View Consistency (assume multiple views are passed in batch)
        consistency = compute_cross_view_consistency(preds)

        # Combine to compute trust scores
        trust_scores = compute_trust([mse, mae, confidence, consistency])

        # Weighted MSE loss
        loss = self.weighted_mse_loss(preds, kpts, trust_scores)
        self.log('train_loss', loss)
        self.log('train_mse', mse)
        self.log('train_mae', mae)
        self.log('train_clarity', np.mean(clarity_scores))
        self.log('train_noise', np.mean(noise_levels))
        self.log('train_occlusions', np.mean(occlusions))
        self.log('train_confidence', confidence)
        self.log('train_consistency', consistency)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, kpts = batch
        preds = self(imgs)

        # Quality Assessment
        clarity_scores = [assess_image_clarity(img) for img in imgs]
        noise_levels = [estimate_noise(img) for img in imgs]
        occlusions = [detect_occlusions(kpt) for kpt in kpts]

        # Historical Performance
        mse, mae = evaluate_historical_performance(preds, kpts)

        # Prediction Confidence
        confidence = compute_prediction_confidence(preds)

        # Cross-View Consistency (assume multiple views are passed in batch)
        consistency = compute_cross_view_consistency(preds)

        # Combine to compute trust scores
        trust_scores = compute_trust([mse, mae, confidence, consistency])

        # Weighted MSE loss
        loss = self.weighted_mse_loss(preds, kpts, trust_scores)
        self.log('val_loss', loss)
        self.log('val_mse', mse)
        self.log('val_mae', mae)
        self.log('val_clarity', np.mean(clarity_scores))
        self.log('val_noise', np.mean(noise_levels))
        self.log('val_occlusions', np.mean(occlusions))
        self.log('val_confidence', confidence)
        self.log('val_consistency', consistency)
        return loss
