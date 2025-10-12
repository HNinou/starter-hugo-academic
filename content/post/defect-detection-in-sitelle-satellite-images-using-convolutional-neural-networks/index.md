---
title: Defect Detection in SITELLE Satellite Images Using Convolutional Neural
  Networks
date: 2025-10-12T17:22:36.096Z
draft: true
featured: false
image:
  filename: featured
  focal_point: Smart
  preview_only: false
---

## Introduction: The Challenge of Defect Detection in Astronomical Imaging

In modern astronomy, instruments like **SITELLE**—an imaging Fourier transform spectrometer—generate vast **data cubes** composed of hundreds of individual images. Even a single defect, such as a cosmic ray or satellite trail, can compromise an entire dataset, necessitating costly reshoots. Manual inspection is impractical due to the sheer volume of data, making **automated defect detection** essential.

This blog post details my **end-to-end project** to build a **Convolutional Neural Network (CNN)** for defect detection in SITELLE images. I led every stage: **dataset creation, architectural design, model training, and performance evaluation**. The goal was to develop a **scalable, unified solution** capable of identifying multiple defect types simultaneously, outperforming classical algorithms in both speed and accuracy.

---

## Why Neural Networks? The Case for CNNs Over Classical Methods

### Limitations of Classical Algorithms
Traditional image processing techniques are often:
- **Time-consuming** to develop and execute.
- **Specialized** for single defect types, requiring multiple algorithms.
- **Inflexible** with varying image dimensions.

### Advantages of Convolutional Neural Networks
CNNs address these challenges by offering:
- **Unified detection**: A single model can identify multiple defect types.
- **Adaptability**: Handles images of diverse shapes and sizes.
- **Efficiency**: Faster implementation and execution compared to classical methods.
- **Superior performance**: Proven effectiveness in image classification tasks.

By leveraging CNNs, we can **automate defect detection**, reducing manual effort and improving data reliability.

---

## Understanding CNNs: Mimicking Biological Vision

### Biological Inspiration
CNNs are inspired by the human visual system, where neurons process local visual information hierarchically. Similarly, CNNs use layers to extract increasingly abstract features from images.

### Core Components of CNNs
1. **Convolutional Layers**: Apply filters (kernels) to detect features like edges, textures, or defects. Example kernels:
   - **Horizontal Edge**: `[0, 0, 0; 0, 0, 1; 1, 1, 1]`
   - **Vertical Edge**: `[0, 0, 0; 1, 1, 1; 0, 0, 0]`
   - **Diagonal Edge**: `[1, 1, 1; 0, 0, 0; 0, 0, 0]`

2. **ReLU Activation**: Introduces non-linearity, enabling the network to model complex patterns.

3. **Pooling Layers**: Reduce dimensionality, retaining essential features while lowering computational cost.

4. **Fully Connected Layers**: Aggregate features for final classification.

### Key Properties
- **Translation Invariance**: Detects features regardless of their position in the image.
- **Scalability**: Works with images of varying sizes, ideal for SITELLE’s diverse data.

---

## Dataset Preparation: Simulating Real-World Defects

### Dataset Composition
- **50,000 training images** and **10,000 verification images**.
- Defects were **artificially injected** into clean SITELLE images to mimic real-world scenarios.

### Defect Types
| Defect Type       | Description                                                                 |
|-------------------|-----------------------------------------------------------------------------|
| **Cosmic Rays**   | Bright spots caused by high-energy particles hitting the sensor.           |
| **Satellite Trails** | Long streaks from satellites passing through the field of view.           |
| **Background**    | Non-defective regions.                                                      |

### Realism and Generalization
To ensure robustness, defects were simulated with **varied intensities and shapes**. The dataset was split into:
- **Training set**: Used to teach the model defect patterns.
- **Verification set**: Evaluated generalization to unseen data.

---

## Architectural Design: Semantic Segmentation for Defect Detection

### Model Workflow
1. **Input**: Raw SITELLE image (potentially containing defects).
2. **Ground Truth**: Manually labeled images for supervised learning.
3. **Feature Extraction**: Convolutional and pooling layers identify defect features.
4. **Classification**: Fully connected layers assign pixels to defect classes.
5. **Output**: Segmented image highlighting defects.

### Loss Function
The model optimizes a **pixel-wise loss function**, comparing predictions to ground truth labels. This guides weight adjustments during training.

---

## Training the Network: Balancing Speed and Accuracy

### Training Process
- **Duration**: ~4 hours on a GPU.
- **Inference Time**: <5 seconds per full-size SITELLE image.
- **Metrics**:
  - **Precision**: Accuracy in defect identification.
  - **Purity**: Minimizing false positives.

### Performance Tuning
A **probability threshold** balances precision and purity. For example:
- **High threshold**: Fewer false positives but potential missed defects.
- **Low threshold**: Higher recall but increased false positives.

---

## Results: Strengths and Limitations

### Verification Set Performance
- **Cosmic rays and satellite trails** were detected with high accuracy.
- **Low-intensity satellites** posed challenges due to faint signals.

### Real-World Application
- **Effective** in operational conditions but occasionally misclassified **saturated star centers** as cosmic rays.

### Limitations
| Challenge                     | Proposed Solution                          |
|-------------------------------|--------------------------------------------|
| Faint satellite detection     | Augment dataset with low-intensity examples. |
| Star misclassification        | Introduce a dedicated "star" class.        |
| Overlapping defect classes    | Enforce mutually exclusive pixel labels.    |

---

## Future Improvements: Toward a Robust System

1. **Dataset Augmentation**: Include more low-intensity satellites and diverse defect patterns.
2. **Class Refinement**: Add a "star" class to reduce misclassification.
3. **Architectural Enhancements**: Explore attention mechanisms or transformer-based models for better feature extraction.

---

## Conclusion: CNNs as a Game-Changer for Astronomical Imaging

This project demonstrates the **power of CNNs** in automating defect detection for SITELLE images. By replacing manual inspection with a **fast, scalable, and accurate** model, we enable more efficient astronomical data analysis.

### Key Takeaways
- CNNs outperform classical algorithms in **speed, adaptability, and accuracy**.
- **End-to-end ownership**—from dataset creation to model deployment—ensures a tailored solution.
- Future work will focus on **dataset expansion** and **architectural refinements** to address edge cases.

By integrating CNNs into the SITELLE pipeline, astronomers can **focus on discovery** rather than data cleaning, unlocking new possibilities in celestial observation.

---
*This project showcases my expertise in **machine learning tools**, **dataset engineering**, and **CNN architecture design** for real-world applications.*
