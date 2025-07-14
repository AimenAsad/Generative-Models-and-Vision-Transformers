# 🤖 Generative Models and Vision Transformers

This repository contains implementations and analyses of advanced deep learning models for **image generation** and **natural language processing (NLP)**, specifically focusing on **Generative Adversarial Networks (GANs)**, **Variational Autoencoders (VAEs)**, and **Transformer architectures** for both machine translation and image classification. The project provides a comparative study of different generative approaches and explores the efficacy of Vision Transformers against traditional CNNs.

---

## 🎯 Project Objectives

* **Image Generation:** Train and compare GANs and VAEs for generating "cat" and "dog" images from the CIFAR-10 dataset, evaluating their performance based on image quality, convergence, and quantitative metrics (e.g., SSIM).
* **Machine Translation:** Implement and fine-tune Transformer-based models (custom and mBART) for English-to-Urdu neural machine translation, assessing translation quality using BLEU scores.
* **Image Classification (ViT vs. CNN):** Compare the performance of Vision Transformers (ViT) with traditional Convolutional Neural Networks (CNNs) on the CIFAR-10 image classification task, analyzing accuracy, training time, and confusion matrices.

---
## 🚀 Task 1: Image Generation using GAN and VAE on CIFAR-10

This section details the training and comparison of two prominent generative models, GANs and VAEs, for conditional image generation.

### 1.1 Objective

To train and compare a **Generative Adversarial Network (GAN)** and a **Variational Autoencoder (VAE)** on a filtered CIFAR-10 dataset (only "cat" and "dog" classes). The objective is to evaluate their performance regarding image quality, convergence behavior, and suitable quantitative metrics.

### 1.2 Dataset Description

* **CIFAR-10 Subset:** The original CIFAR-10 dataset (60,000 32x32 color images across 10 classes) was filtered to include only "cat" and "dog" categories.
* **Split:** 80% training, 20% testing split.
* **Normalization:** All images were normalized for improved model convergence.

### 1.3 Model Architectures and Methodology

* **Custom GAN with Similarity Discriminator:**
    * **Generator:** Standard GAN generator architecture.
    * **Discriminator:** A custom, similarity-based discriminator (inspired by Siamese Networks) that produces a similarity score between a real and a generated image, rather than a binary classification. The generator aims to minimize this score for generated images.
    * **Loss Functions:**
        * **Generator Loss ($\mathcal{L}_G$):** $E[1 - S(G(z), x)]$
        * **Discriminator Loss ($\mathcal{L}_D$):** $E[|S(G(z), x) - 0| + |S(x, x) - 1|]$
    * **Diversity Improvement:** Mini-batch discrimination was implemented within the generator's training pipeline to enhance diversity and mitigate mode collapse.

* **Variational Autoencoder (VAE):**
    * **Encoder:** Maps input images to a latent distribution (mean $\mu$, log-variance $\log\sigma$).
    * **Decoder:** Reconstructs images from sampled latent vectors.
    * **Loss Function ($\mathcal{L}_{VAE}$):** $E_{q(z|x)} [\log p(x|z)] - D_{KL} (q(z|x)||p(z))$
        * The KL divergence term regularizes the latent space to follow a standard Gaussian distribution.

### 1.4 Training Details

* **Epochs:** Both models trained for 50 epochs.
* **Optimizer:** Adam optimizer for both.
* **Learning Rates:**
    * GAN: 0.0002 with beta values of (0.5, 0.999).
    * VAE: 0.001.
* **Batch Size:** 64 for both models.
* **Logging:** Training curves were logged for generator/discriminator losses and VAE loss.

### 1.5 Results and Analysis

* **GAN Performance:**
    * Generator loss increased gradually, suggesting improving realism in generated images.
    * Discriminator loss remained near-zero due to the custom similarity scoring method, indicating the discriminator was consistently confident in its similarity scores.
    * **Structural Similarity Index (SSIM):** A low SSIM score (0.0492) between real and generated images suggests room for improvement in generating fine details.
    * *Visualizations:* Sample GAN images and training loss curves are available in the `results/` directory.

* **VAE Performance:**
    * Showed stable reconstruction loss.
    * Generated slightly blurry outputs, which is a common characteristic of VAEs due to the Gaussian sampling in the latent space.
    * *Visualizations:* Sample VAE reconstructions and training loss curves are available in the `results/` directory.

### 1.6 Conclusion

Both models successfully learned representations of "cat" and "dog" images. GAN outputs were generally more visually coherent, but lacked structural similarity to real images (low SSIM). VAEs offered consistent convergence but produced blurrier samples. Future work could explore conditional GANs or VAE-GAN hybrids to leverage the strengths of both architectures.

---

## 🚀 Task 2: Machine Translation using Transformers (English to Urdu)

This section focuses on implementing and fine-tuning Transformer-based sequence-to-sequence models for English-to-Urdu machine translation.

### 2.1 Objective

To train a **Transformer-based sequence-to-sequence model** for translating English text into Urdu. This includes utilizing both a custom-implemented Transformer architecture and fine-tuning a pre-trained multilingual model (mBART).

### 2.2 Dataset Description

Two datasets were utilized:

* **Parallel Corpus from Kaggle:** Contains over 24,000 aligned English-Urdu sentence pairs.
* **UMC005 Corpus:** Includes religious and formal texts, supplementing the parallel corpus.
* **Augmentation:** Data augmentation techniques were applied, increasing the combined dataset to 29,430 sentence pairs.

### 2.3 Model Architecture

* **Custom Transformer:** Implemented the original Transformer architecture.
    * **Positional Encodings:** Used sine and cosine functions to inject positional information into the embeddings:
        * $PE_{(pos,2i)} = \sin(pos/10000^{2i/d_{model}})$
        * $PE_{(pos,2i+1)} = \cos(pos/10000^{2i/d_{model}})$
    * *Visualizations:* Positional encoding visualization is available in the `results/` directory.

### 2.4 Training and Evaluation

* **Optimizer:** Adam optimizer.
* **Learning Rate:** Employed a learning rate warmup schedule.
* **Evaluation Metric:** **BLEU (Bilingual Evaluation Understudy) score**.
* **Performance:** Final BLEU-4 scores exceeded 30, indicating strong translation quality.
* **Comparison:** Fine-tuning the pre-trained **mBART** model consistently outperformed the custom Transformer models, demonstrating the benefits of transfer learning for low-resource languages or domain-specific tasks.
* *Visualizations:* Sample mBART translations are available in the `results/` directory.

### 2.5 Conclusion

Transformer-based models proved highly effective for English-to-Urdu machine translation, with significant performance gains observed when fine-tuning pre-trained models like mBART. Future improvements could involve attention visualization, back-translation techniques, or more extensive multilingual training.

---

## 🚀 Task 3: Vision Transformer vs CNN on CIFAR-10

This section presents a comparative analysis of Vision Transformers (ViT) against traditional Convolutional Neural Networks (CNNs) for image classification.

### 3.1 Objective

To compare the performance of **Vision Transformers (ViT)** with traditional **Convolutional Neural Networks (CNNs)** on the CIFAR-10 image classification task.

### 3.2 Model Description

* **CNN Baseline:**
    * A ResNet-like CNN architecture incorporating convolution, ReLU activation, pooling, batch normalization, and dropout layers.
* **Vision Transformer (ViT):**
    * The input image was split into smaller patches.
    * Each patch was linearly embedded and combined with learnable positional encodings.
    * The sequence of embedded patches was then processed through multiple self-attention blocks within the Transformer encoder.

### 3.3 Training and Evaluation

* **Dataset:** Both models were trained on the CIFAR-10 dataset.
* **Data Augmentation:** Data augmentation techniques were applied during training to both models.
* **Metrics Evaluated:**
    * Accuracy, Precision, Recall, F1-score
    * Training time and convergence behavior
    * Confusion matrix and sample predictions
* *Visualizations:* Accuracy and loss comparison curves, confusion matrices, class performance plots, and sample predictions from both models are available in the `results/` directory.

### 3.4 Conclusion

* **Accuracy:** ViT demonstrated competitive accuracy compared to the CNN baseline.
* **Parameters & Training Time:** ViT achieved this with fewer parameters but generally required longer training times.
* **Convergence:** CNNs exhibited faster convergence during training.
* **Dataset Size:** ViT models typically benefit more significantly from larger datasets, suggesting their full potential might be realized with more extensive training data.
* **Future Work:** Hybrid models combining the strengths of CNNs (local feature extraction) and Transformers (global context) may offer better trade-offs in future work.

---

## 🌐 Socials:

[![LinkedIn](https://img.shields.io/badge/LinkedIn-%230077B5.svg?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/aimen-asad-536496299/)
[![Email](https://img.shields.io/badge/Email-D14836?logo=gmail&logoColor=white)](mailto:aimenasad42@gmail.com)

---

## 💻 Tech Stack:

![Java](https://img.shields.io/badge/java-%23ED8B00.svg?style=for-the-badge&logo=openjdk&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Redis](https://img.shields.io/badge/redis-%23DD0031.svg?style=for-the-badge&logo=redis&logoColor=white)
![SQLite](https://img.shields.io/badge/sqlite-%2307405e.svg?style=for-the-badge&logo=sqlite&logoColor=white)
![MySQL](https://img.shields.io/badge/mysql-4479A1.svg?style=for-the-badge&logo=mysql&logoColor=white)
![Postgres](https://img.shields.io/badge/postgres-%23316192.svg?style=for-the-badge&logo=postgresql&logoColor=white)
![Adobe Illustrator](https://img.shields.io/badge/adobe%20illustrator-%23FF9A00.svg?style=for-the-badge&logo=adobe%20illustrator&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![mlflow](https://img.shields.io/badge/mlflow-%23d9ead3.svg?style=for-the-badge&logo=numpy&logoColor=blue)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Scipy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%230C55A5)
![Git](https://img.shields.io/badge/git-%23F05033.svg?style=for-the-badge&logo=git&logoColor=white)
![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)
![Trello](https://img.shields.io/badge/Trello-%23026AA7.svg?style=for-the-badge&logo=Trello&logoColor=white)
![JavaScript](https://img.shields.io/badge/javascript-%23323330.svg?style=for-the-badge&logo=javascript&logoColor=%23F7DF1E)
![Canva](https://img.shields.io/badge/Canva-%2300C4CC.svg?style=for-the-badge&logo=Canva&logoColor=white)
![Power Bi](https://img.shields.io/badge/power_bi-F2C811?style=for-the-badge&logo=powerbi&logoColor=black)

---
