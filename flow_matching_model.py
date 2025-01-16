import argparse
from typing import Optional, Tuple
from flow_matching.path import CondOTProbPath

import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import pytorch_lightning as pl

# Import flow_matching library components
# Adjust the import paths based on the actual library structure
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath
from flow_matching.solver import Solver, ODESolver
# If the library has different components, adjust accordingly
from unet import UNet

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class ImageFlowMatcher(pl.LightningModule):
    """
    Flow Matching model for MNIST image generation.
    Utilizes the flow_matching library for path sampling and loss computation.
    """

    def __init__(
        self,
        lr: float = 1e-4,
        num_train_timesteps: int = 1000,
        sigma: float = 0.1,
        step_size: float = 0.05,
        num_steps: int = 1000,
        use_cuda: bool = True
    ):
        super(ImageFlowMatcher, self).__init__()
        self.save_hyperparameters()  # Saves hyperparameters for easy access later

        self.lr = lr
        self.num_train_timesteps = num_train_timesteps
        self.sigma = sigma
        self.step_size = step_size
        self.num_steps = num_steps
        self.normalize_data = True  # Assumes normalization in training_step

        # Initialize the CNN model as the velocity field
        self.model = UNet(1, 1, c=32)

        # Initialize the probability path for flow matching
        self.prob_path = CondOTProbPath()

        self.criterion = nn.MSELoss()
        
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the CNN model.

        Args:
            xt (torch.Tensor): Current state tensor.
            t (torch.Tensor): Time scalar.

        Returns:
            torch.Tensor: Velocity field tensor.
        """
        return self.model(x, t)

    def generate(
        self,
        batch_size: int = 32,
        sample_image_size: Tuple[int, int, int] = (1, 28, 28),
        step_size: float = 0.05,
        num_steps: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate images using the trained flow model via ODE integration.

        Args:
            batch_size (int): Number of images to generate.
            sample_image_size (Tuple[int, int, int]): Size of each sample image (C, H, W).
            step_size (float): Integration step size.
            num_steps (Optional[int]): Number of integration steps. Defaults to 2.

        Returns:
            torch.Tensor: Generated images tensor.
        """
        if num_steps is None:
            num_steps = 2
        solver = ODESolver(velocity_model=self.model)
        # Initialize from Gaussian noise
        x_init = torch.randn(batch_size, *sample_image_size, device=self.device)

        # Define a uniform time grid
        time_grid = torch.linspace(0, 1, steps=num_steps, device=self.device)

        # Perform ODE integration using the solver
        generated = solver.sample(time_grid=time_grid, x_init=x_init, method='dopri5', step_size=step_size, atol=1e-4, rtol=1e-4)

        # Denormalize if needed
        if self.normalize_data:
            generated = (generated + 1) / 2  # transpose from [-1, 1] to [0, 1]
            generated = torch.clamp(generated, 0.0, 1.0)

        return generated

    def training_step(self, batch, batch_idx):
        """
        Training step for flow matching.

        Args:
            batch (tuple): A batch of data containing images and labels.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Training loss.
        """
        images = batch[0]  # MNIST returns (images, labels)

        # Normalize images to [-1, 1]
        if self.normalize_data:
            images = images.float() / 255.
            images = images * 2 - 1  # Normalize [0,1] to [-1,1]

        # Sample initial noise x0
        x0 = torch.randn_like(images)

        # Sample a random t in [0,1] for each sample in the batch
        t = torch.rand(images.size(0), device=self.device)

        # Sample paths for flow matching
        path_batch = self.prob_path.sample(x0, images, t)

        # Predict velocity field at x_t

        # Predict velocity field at x_t
        vt = self.forward(path_batch.x_t, path_batch.t)  # Unconditional generation

        # Compute flow matching loss: MSE between predicted velocity and true velocity
        loss = self.criterion(vt, path_batch.dx_t)
        
        # Log training loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step for flow matching.

        Args:
            batch (tuple): A batch of data containing images and labels.
            batch_idx (int): Index of the batch.
        """
        images  = batch[0]

        # Normalize images to [-1, 1]
        if self.normalize_data:
            images = images.float() / 255.
            images = images * 2 - 1 # Normalize [0,1] to [-1,1]

        # Sample initial noise x0
        x0 = torch.randn_like(images)

        # Sample a random t in [0,1] for each sample in the batch
        t = torch.rand(images.size(0), device=self.device)

        # Sample paths for flow matching
        path_batch = self.prob_path.sample(x0, images, t)

        # Predict velocity field at x_t
        vt = self.forward(path_batch.x_t, path_batch.t)

        # Compute flow matching loss
        loss = self.criterion(vt, path_batch.dx_t)
        
        # Log validation loss
        self.log('val_loss', loss, prog_bar=True, logger=True)

        # Optionally, generate and log sample images
        if batch_idx == 0:
            sample_images = self.generate(batch_size=16)
            grid = torchvision.utils.make_grid(sample_images, nrow=4, normalize=False, value_range=(0,1))
            self.logger.experiment.add_image('generated_images', grid, self.current_epoch)
            plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
            plt.show()
            plt.close()

    def configure_optimizers(self):
        """
        Configure the optimizer for training.

        Returns:
            torch.optim.Optimizer: Configured optimizer.
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer

    @staticmethod
    def add_model_specific_args(parser):
        """
        Add model-specific command-line arguments.

        Args:
            parser (argparse.ArgumentParser): Argument parser.

        Returns:
            argparse.ArgumentParser: Updated parser with model-specific arguments.
        """
        group_parser = parser.add_argument_group(title='FlowMatcher')
        group_parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
        group_parser.add_argument('--num_train_timesteps', type=int, default=1000, help='Number of ODE time steps')
        group_parser.add_argument('--sigma', type=float, default=0.1, help='Sigma for the Flow Matcher')
        group_parser.add_argument('--step_size', type=float, default=0.05, help='ODE integration step size')
        group_parser.add_argument('--num_steps', type=int, default=1000, help='Number of ODE integration steps')
        return parser