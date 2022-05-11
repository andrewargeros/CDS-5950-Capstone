# CDS 5950- Computer Vision Capstone Project


## Data 

The data for this project come from [Untapped](https://untapped.com/) Top Beers by Style. Images are then parsed into pictures of beers vs pictures of cans, and resized for use in training. 
 

## Models

*Note: 7 Classes

|Model Name        | Parameters | Training Time | Accuracy | Latency |
|------------------|------------|---------------|----------|---------|
| CNN              | 7,656,632  | 03:17:22      | 49.14%   | 0.09    |
| SimCSE           | 39,261     | 02:01:48      | 39.52%   | 0.06    |
| CoAtNet          | 16,054,392 | 04:46:13      | 58.88%   | 0.12    |
| CoAtNet w/ SimCSE| 13,107,902 | 06:27:27      | 48.26%   | 0.09    |
