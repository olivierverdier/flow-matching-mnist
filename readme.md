# Flow Matching for MNIST
![flow matching](https://github.com/user-attachments/assets/78d56f06-29e7-461c-9f9d-0d024549f853)


This project implements a flow matching model for image generation using PyTorch Lightning. The model is trained on the MNIST dataset and can generate new images using a learned flow.

## Installation

```bash
pip install -r requirements.txt
```
## Jupyter Notebook

The project includes a Jupyter notebook (`flow-matching-with-mnist-dataset.ipynb`) that demonstrates the flow matching process and provides the complete workflow of the training on kaggle

## Training

To train the model, run:

```bash
python train.py --data_dir data --batch_size 32 --max_epochs 10
```

Key training arguments:
- `--data_dir`: Directory to store the datasets (default: 'data')
- `--batch_size`: Batch size for training and testing (default: 32)
- `--max_epochs`: Maximum number of training epochs (default: 10)
- `--devices`: Number of GPUs to use (default: 1)
- `--ckpt_dir`: Directory to save checkpoints (default: 'checkpoints')
- `--early_stopping`: Enable early stopping (flag)
- `--patience`: Early stopping patience (default: 3)

## Generation
![output_mnist](https://github.com/user-attachments/assets/a0c66b52-49f4-40bc-bec4-132f8ef2df35)

To generate new images using a trained model, run:

```bash
python generate.py --checkpoint path/to/checkpoint.ckpt --num_samples 16
```

Key generation arguments:
- `--checkpoint`: Path to model checkpoint (required)
- `--num_samples`: Number of images to generate (default: 32)
- `--batch_size`: Batch size for generation (default: 32)
- `--output_dir`: Directory to save generated images (default: 'generated')
- `--num_steps`: Number of steps for generation (default: 2)
- `--channels`: Number of channels in generated images (default: 1)
- `--height`: Height of generated images (default: 28)
- `--width`: Width of generated images (default: 28)
- `--seed`: Random seed for reproducibility (optional)

## Model Architecture

The flow matching model is implemented in `flow_matching_model.py`. It uses a UNet architecture to learn the velocity fields for the flow matching process.

## Project Structure

```
.
├── README.md
├── requirements.txt
├── train.py
├── generate.py
└── flow_matching_model.py
```

## Example Usage

1. Train the model:
```bash
python train.py --batch_size 64 --max_epochs 20 --early_stopping
```

2. Generate images:
```bash
python generate.py --checkpoint checkpoints/flow-best.ckpt --num_samples 100 --num_steps 4
```

Generated images will be saved in the specified output directory.
