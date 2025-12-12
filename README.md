# Automated Deep Learning Approach For Flood Detection Using Satellite Imagery
<p align="justify"> This project demonstrates an automated flood detection system using Deep Learning techniques . Flood detection is crucial for disaster management, urban planning, and early warning systems. In this project, satellite and aerial images are processed, preprocessed, and segmented using advanced deep learning models like UNet, Deep Lab V3+ and Attention U-net  to identify water bodies and flood-affected areas. This project was developed as part of my final-year project focusing on real-time flood monitoring and disaster mitigation. </p>

## Table of Contents
- [Project Overview](#project-overview)
- [Why I Chose This Project](#why-i-chose-this-project)
- [Problem This Project Solves](#problem-this-project-solves)
- [Dataset](#dataset)
- [Flow of the Project](#flow-of-the-project)
- [Files in This Repository](#files-in-this-repository)
- [Tech Stack Used and Why](#tech-stack-used-and-why)
- [Usage Instructions](#usage-instructions)
- [Results and Insights](#results-and-insights)
- [Author](#author)
- [Contact](#contact)

## Project Overview
<p align="justify"> The objective of this project is to automatically detect flood-affected areas from satellite and aerial images using semantic segmentation. The project leverages deep learning models like <b>UNet</b> for pixel-wise segmentation, <b>Deep Lab V3+</b> and <b>Attention U-Net</b> modules for improved feature representation, and <b>Attention Units</b> to focus on relevant regions. The workflow includes data preprocessing, augmentation, model building, training, evaluation, and prediction, enabling accurate flood detection for real-world applications. </p>

## Why Did I Choose This Project?
<p align="justify"> Floods are one of the most destructive natural disasters, causing loss of life, property damage, and environmental degradation. I chose this project to contribute to disaster management by leveraging deep learning and satellite imagery for automated flood detection. Working on this project enhanced my understanding of semantic segmentation, encoder-decoder architectures, attention mechanisms, and real-time image analysis for critical applications. </p>

## Problem This Project Solves
<p align="justify"> Traditional flood detection methods rely on manual analysis of images or sensor readings, which is time-consuming and prone to errors. Rapid and accurate identification of flood-affected areas is essential for emergency response, evacuation planning, and resource allocation. This project provides an automated solution to detect water bodies and flooded regions from satellite images, helping authorities respond efficiently to disasters. </p>

## Dataset
<p align="justify"> The dataset consists of high-resolution satellite and aerial images of flood-prone regions. Each image is paired with a corresponding mask indicating water bodies and flood-affected areas. The dataset includes diverse environmental conditions, seasons, and varying water levels to make the model robust. </p> <p align="justify"><b>Dataset Link:</b> <a href="https://www.kaggle.com/datasets/sovitrath/water-bodies-segmentation-dataset-with-split" target="_blank">Download Flood Detection Dataset from Kaggle</a></p> <p align="justify"> Each image undergoes preprocessing steps including resizing, normalization, and augmentation. These images are then fed into semantic segmentation models to extract pixel-level flood predictions. </p>

## Flow of the Project
<p align="justify"> The workflow of this project transforms raw satellite images into segmented flood maps. The main steps include: </p>

 Load Dataset
Images and corresponding masks are loaded from training and testing directories.

Preprocessing and Augmentation

Resize images to a standard resolution

Normalize pixel values

Apply data augmentations such as flips, rotations, and brightness adjustments

Feature Extraction and Model Inputs

Images are converted into suitable tensor formats

Relevant channels and features are extracted for model input

Model Building Using U-Net,Attention U-Net and DeepLb V3+.

Model Training
The model is trained using loss functions like Dice Loss and Binary Cross-Entropy for accurate segmentation.

Evaluation

Metrics: Intersection over Union (IoU), Dice Coefficient, Precision, Recall

Visualization: Predicted masks overlaid on original images

Prediction
The trained model generates flood maps for unseen images, highlighting water bodies and flooded areas.

## Files in This Repository

flood_detection.ipynb – Jupyter Notebook with full implementation

Project in Ieee format – project documentation

requirements.txt – List of dependencies for running the project

README.md – Project documentation

## Tech Stack Used and Why

Python – Core programming language for data processing and model development

TensorFlow / Keras – Deep learning framework for building segmentation models

OpenCV & PIL – Image preprocessing and augmentation

Albumentations – Advanced image augmentation library

NumPy & Pandas – Numerical computations and dataset handling

Matplotlib & Seaborn – Visualization of images, masks, and evaluation metrics

Scikit-learn – Dataset splitting and evaluation metrics

<p align="justify"> This tech stack provides a comprehensive ecosystem for image preprocessing, model development, training, evaluation, and visualization, enabling robust flood detection. </p>

## Usage Instructions

Clone the repository

git clone https://github.com/JAHNAVIDINGARI/FLOOD-DETECTION-PROJECT.git

Navigate to the project directory

cd flood-detection


Install dependencies

pip install -r requirements.txt


Run the notebook
Open flood_detection.ipynb, update dataset paths if required, and execute all cells to train, evaluate, and generate predictions.

## Results and Insights
<p align="justify"> The deep learning models trained on the Flood Detection dataset achieved strong performance in segmenting water bodies and flood-affected regions. Key observations include: </p>

Deep Lab v3+ outperformed in generating accurate water masks.

Data augmentation improved model generalization across varying environmental conditions.

Predicted masks closely align with ground truth, enabling precise flood mapping.

The approach can be deployed in real-time disaster management systems for early warning and monitoring.

<p align="justify"> These results confirm that the system is robust for real-world flood detection, helping authorities identify and respond to affected areas efficiently. </p>

## Authors

Jahnavi Dingari

Sandeep Pulata

Sountharrajan S

## Contact
<p align="justify"> For queries, collaboration, or further discussion regarding this project, please reach out via <b>LinkedIn</b> or <b>Email</b>: </p>

LinkedIn: https://www.linkedin.com/in/jahnavi-dingari

Email: <a href="mailto:jahnavidingari04@gmail.com">jahnavidingari04@gmail.com
</a>
