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


import requests
import json
def get_weather_data(city, api_key, units='metric'):
    """
    Retrieves the current weather data for a given city using the OpenWeatherMap API.

    Args:
    city (str): The city name for which to retrieve the weather data.
    api_key (str): The OpenWeatherMap API key.
    units (str, optional): The unit system to use for temperature values. Defaults to 'metric'.

    Returns:
    dict: A dictionary containing the retrieved weather data.
    """

    # Construct the API URL
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units={units}"

    try:
        # Send a GET request to the API
        response = requests.get(url)

        # Check if the request was successful
        if response.status_code == 200:
            # Parse the JSON response
            weather_data = response.json()

            return weather_data
        else:
            # Return an error message if the request was not successful
            return {"error": f"Failed to retrieve weather data for {city}. Status code: {response.status_code}"}
    except requests.exceptions.RequestException as e:
        # Return an error message if a request exception occurred
        return {"error": f"An error occurred while retrieving weather data for {city}: {e}"
}