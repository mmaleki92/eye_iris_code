# Iris Recognition System

This is a simplified implementation of an iris recognition system, inspired by the techniques used to identify the "Afghan Girl" (Sharbat Gula) 18 years after her famous National Geographic photograph.

## How It Works

1. **Image Preprocessing**: Loads and converts eye images to grayscale
2. **Iris Localization**: Detects the boundaries of the iris and pupil
3. **Normalization**: Converts the circular iris to a rectangular form (Daugman's "rubber sheet" model)
4. **Feature Extraction**: Creates a binary code representing iris patterns
5. **Matching**: Compares two iris codes using Hamming Distance

## Requirements

- Python 3.6+
- OpenCV
- NumPy
- Matplotlib
- scikit-image
- SciPy

# Reference
- Eyes data: CASIA1

data taken from:
https://www.kaggle.com/datasets/sondosaabed/casia-iris-thousand