# CDS 5950- Computer Vision Capstone Project

This reposistory contains code for my capstone project for the Computational Data Science major at Hamline University. The project is a series of computer vision algorithms that are used to classify images of beer by style. The project evaluates the efficacy and efficiency differences between Sharpened Cosine Similarity (SimCSE) and Convolutions for image classification.

Presented 2022-05-11 for CDS-5950 CDS Capstone Project

Blog post: [Andrew's Blog](https://www.andrewargeros.com/post/image-classification-efficiency-for-beers)

## Data 

The data for this project come from [Untapped](https://untapped.com/) Top Beers by Style. This creates a download of roughly 129,000 images. Code to download the images can be found is the `/GetData` directory. NOTE: The data is not available for use in the project, as it is too large to be hosted on GitHub.

## Data Cleaning

To clean the images, I used CLIP to sort the images based on whether or not they contain an image of a glass of beer. The images that do not contain a beer are removed. The images that do contain a beer are then sorted based on the style of beer. The images are then sorted based on the style of beer, to then create train and test datasets. 
 
## Models

*Note: 6 Classes

|Model Name        | Parameters | Training Time | Accuracy | Latency |
|------------------|------------|---------------|----------|---------|
| CNN              | 7,656,632  | 03:17:22      | 49.14%   | 0.09    |
| SimCSE           | 39,261     | 02:01:48      | 39.52%   | 0.06    |
| CoAtNet          | 16,054,392 | 04:46:13      | 58.88%   | 0.12    |
| CoAtNet w/ SimCSE| 13,107,902 | 06:27:27      | 48.26%   | 0.09    |
