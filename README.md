# Task-04: Image-to-Image Translation with cGAN

**Author:** Renuka Gosavi  
**Organization:** Comillas Negras / ProDigy Infotech  

## Objective

Perform **Image-to-Image Translation** using **Conditional Generative Adversarial Networks (cGANs)**.  

This task involves training a neural network to **translate input images** (e.g., simple shapes or gradients) into **target images** with desired features.  

**Example:**  
- Input: Circle with gradient  
- Output: Stylized or transformed circle image  

## How It Works

1. **Prepare Input-Target Pairs**:  
   - Input images (e.g., gradient shapes)  
   - Corresponding target images (desired outputs)  

2. **Conditional GAN (cGAN) Architecture**:  
   - Generator: Generates images conditioned on input  
   - Discriminator: Distinguishes real vs generated images  

3. **Training Process**:  
   - Feed input-target pairs to the cGAN  
   - Generator learns to produce outputs similar to target  
   - Discriminator provides feedback to improve generator
   - 

## Files

- `cgan_image_translation.py` – Python script to train or run cGAN for image-to-image translation  
- `input/` – Folder containing input images  
- `target/` – Folder containing target images  
- `output/` – Folder where translated/generated images will be saved  


## How to Run

1. Install required Python packages:

```bash
pip install torch torchvision matplotlib pillow
python cgan_image_translation.py
