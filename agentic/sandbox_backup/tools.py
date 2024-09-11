import numpy as np
import cv2
from matplotlib import pyplot as plt


def generate_gaussian_noise_image():
    """
    Generate a 400x400 image filled with Gaussian noise.

    Gaussian noise is a type of statistical noise that has a probability density function (PDF) 
    equal to that of the normal distribution, also known as the Gaussian distribution. In other 
    words, the values of the noise follow a bell curve.

    The mean and standard deviation of the Gaussian noise can be adjusted, but in this example, 
    we will use a mean of 0 and a standard deviation of 25.

    Returns:
        np.ndarray: A 2D numpy array representing a 400x400 grayscale image filled with Gaussian noise.
    """
    # Image dimensions
    width, height = 400, 400
    
    # Mean and standard deviation
    mean = 0
    std_dev = 25
    
    # Generate Gaussian noise
    gauss_noise = np.random.normal(mean, std_dev, (height, width))

    # Clipping to ensure values are between 0 and 255
    gauss_noise = np.clip(gauss_noise, 0, 255).astype(np.uint8)
    
    # Displaying the image using matplotlib
    plt.imshow(gauss_noise, cmap='gray')
    plt.title("Gaussian Noise Image")
    plt.axis('off')  # Hide the axis
    plt.show()

    return gauss_noise