# Face Reconstruction GAN

Identity-Preserving Face Reconstruction under Occlusion using Generative Adversarial Networks.

This project implements a GAN-based framework for reconstructing occluded face images while preserving identity consistency.

---

## 📌 Overview

Face recognition systems significantly degrade under occlusion (masks, objects, hands, etc.).

This repository presents a **Face Reconstruction GAN** that:

- Restores missing facial regions  
- Preserves identity-specific features  
- Improves PSNR and SSIM  
- Supports downstream recognition pipelines  

---

## 🏗 Model Architecture

### Generator + Discriminator Architecture

![GAN Architecture](images/gan_architecture.png)

---

## 🧪 Example Reconstruction

Sample output after training:

![Reconstruction Example](images/epoch_0025.png)

---

## 📊 Training Curves

### Discriminator Loss

![Discriminator Loss](output/1_discriminator_loss.png)

---

### Generator Total Loss

![Generator Total Loss](output/2_generator_total_loss.png)

---

### Generator Adversarial Loss

![Generator Adversarial Loss](output/3_generator_adversarial_loss.png)

---

### Generator Context Loss

![Generator Context Loss](output/4_generator_context_loss.png)

---

### Generator Identity Loss

![Generator Identity Loss](output/5_generator_identity_loss.png)

---

### PSNR Curve

![PSNR](output/6_psnr.png)

---

### SSIM Curve

![SSIM](output/7_ssim.png)

---

## 📂 Project Structure

