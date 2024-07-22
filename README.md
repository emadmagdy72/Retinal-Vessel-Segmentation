# Retinal Vessel Segmentation

This project focuses on segmenting retinal blood vessels in fundus images using advanced deep learning models. Accurate segmentation of retinal vessels is crucial for diagnosing and monitoring various ophthalmic conditions, such as diabetic retinopathy and glaucoma.

## Project Overview

Retinal vessel segmentation involves distinguishing blood vessels from the background in retinal images, a key step in automated ophthalmic diagnostics. This project leverages two powerful neural network architectures, **U-Net** and **Sagenet**, to achieve high precision and accuracy in vessel detection.

### U-Net

U-Net is a convolutional neural network architecture designed for biomedical image segmentation. It is particularly effective due to its encoder-decoder structure, which captures context and precise localization through skip connections.

### Sagenet

Sagenet is another deep learning architecture known for its robustness in handling complex image segmentation tasks. It complements U-Net by providing enhanced performance in detecting intricate vessel structures.

## Dataset

The project uses the **[Digital Retinal Images for Vessel Extraction (DRIVE)](https://drive.grand-challenge.org/)** dataset, which includes 40 color fundus photographs. Each image is manually labeled for training and testing purposes, ensuring high-quality ground truth data for model evaluation.

## Results

### Model Performance Comparison

#### Overall Performance Metrics

| Metric   | U-Net | Sagenet |
|----------|-------|---------|
| Accuracy | 0.97  | 0.98    |

#### Class-wise Performance Metrics

**Class Background**

| Metric    | U-Net | Sagenet |
|-----------|-------|---------|
| Precision | 0.97  | 0.98    |
| Recall    | 1.00  | 0.99    |
| F1-Score  | 0.98  | 0.99    |

**Class Object**

| Metric    | U-Net | Sagenet |
|-----------|-------|---------|
| Precision | 0.95  | 0.92    |
| Recall    | 0.67  | 0.82    |
| F1-Score  | 0.78  | 0.87    |


The results demonstrate that both models are highly effective in segmenting retinal vessels, with Sagenet achieving slightly higher accuracy and F1 score, indicating its superior ability to capture complex vessel structures.

## Video Demo
A video demonstration showcasing the segmentation results using both U-Net and Sagenet is available. The demo highlights the real-time processing of retinal images and the effectiveness of these models in accurately identifying and segmenting blood vessels.

https://github.com/user-attachments/assets/7d790b91-e71b-4b4a-8220-06545b148f5b




[Watch the video demo here](#) <!-- Replace with the actual link to the video demo -->

## Conclusion

The successful implementation of U-Net and Sagenet for retinal vessel segmentation highlights the potential of deep learning models in enhancing diagnostic tools for ophthalmic diseases. These models can assist healthcare professionals by providing automated, accurate vessel segmentation, leading to improved patient outcomes.
