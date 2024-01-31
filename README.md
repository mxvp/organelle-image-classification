# Organelle Image Classification

## Introduction

This project aims to reach protein classification through advanced image analysis techniques. The primary objective is to create a machine learning model capable of predicting the presence of specific proteins within a given image, addressing the complexity of a multi-label classification challenge. The metric of choice for assessing model performance is the mean F1-score, ensuring precision and recall are both accounted for.

## Data

Dataset contains 15389 128 x 128 resolution images of 3 channels (RGB) in PNG format. There are in total 10 different labels present in the dataset. The dataset is acquired in a highly standardized way using one imaging modality (confocal microscopy). However, the dataset comprises 10 different cell types of highly different morphology, which affect the protein patterns of the different organelles. Each image can have 1 or more labels associated to them.

![example](https://github.com/mxvp/organelle-image-classification/blob/main/images/samples.png)

## Methodology

The multilabel image classification machine learning project is performed using the Densenet121 CNN.

![densenet](https://github.com/mxvp/organelle-image-classification/blob/main/images/densenet.png)

## Results

An accuracy of 90% and F1-score of 0.59 was achieved.

![confusion](https://github.com/mxvp/organelle-image-classification/blob/main/images/confusion.png)