import numpy as np
from PIL import Image
import argparse

# Conversion matrices for sRGB to LMS and vice-versa
# These matrices are commonly used in color blindness simulation literature.
# Source: http://www.daltonize.org/
# sRGB to LMS (assuming D65 white point, 2 degree observer)
RGB_TO_LMS_MATRIX = np.array([
    [17.8824, 43.5161, 4.11935],
    [3.45565, 27.1554, 3.86714],
    [0.0299566, 0.184309, 1.46709]
])

LMS_TO_RGB_MATRIX = np.array([
    [0.0809, -0.1305, 0.1167],
    [-0.0102, 0.0540, -0.1136],
    [-0.0004, -0.0041, 0.6935]
])

# Protanopia simulation matrix (loss of L cone)
PROTANOPIA_MATRIX = np.array([
    [0, 2.02344, -2.52581],
    [0, 1, 0],
    [0, 0, 1]
])

# Deuteranopia simulation matrix (loss of M cone)
DEUTERANOPIA_MATRIX = np.array([
    [1, 0, 0],
    [0.494207, 0, 1.24827],
    [0, 0, 1]
])

# Tritanopia simulation matrix (loss of S cone)
TRITANOPIA_MATRIX = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [-0.395913, 0.821107, 0]
])

def simulate_color_blindness(image_path: str, blindness_type: str) -> Image.Image:
    """
    Simulates different types of color blindness on an image.

    Args:
        image_path (str): Path to the input image.
        blindness_type (str): Type of color blindness to simulate
                              ('protanopia', 'deuteranopia', 'tritanopia').

    Returns:
        PIL.Image.Image: The simulated image.
    """
    img = Image.open(image_path).convert("RGB")
    pixels = np.array(img).astype(float) / 255.0  # Normalize to [0, 1]

    # Convert RGB to LMS
    # Flatten pixels for matrix multiplication
    flat_pixels = pixels.reshape(-1, 3)
    lms_pixels = np.dot(flat_pixels, RGB_TO_LMS_MATRIX.T)

    # Apply color blindness simulation matrix
    if blindness_type == 'protanopia':
        simulated_lms = np.dot(lms_pixels, PROTANOPIA_MATRIX.T)
    elif blindness_type == 'deuteranopia':
        simulated_lms = np.dot(lms_pixels, DEUTERANOPIA_MATRIX.T)
    elif blindness_type == 'tritanopia':
        simulated_lms = np.dot(lms_pixels, TRITANOPIA_MATRIX.T)
    else:
        raise ValueError(f"Unknown blindness type: {blindness_type}")

    # Convert LMS back to RGB
    simulated_rgb = np.dot(simulated_lms, LMS_TO_RGB_MATRIX.T)

    # Clip values to [0, 1] and convert back to 0-255 for image saving
    simulated_rgb = np.clip(simulated_rgb, 0, 1) * 255.0
    simulated_pixels = simulated_rgb.reshape(pixels.shape).astype(np.uint8)

    return Image.fromarray(simulated_pixels)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simulate different types of color blindness on an image."
    )
    parser.add_argument(
        "input_image",
        type=str,
        help="Path to the input image file."
    )
    parser.add_argument(
        "output_image",
        type=str,
        help="Path to save the simulated output image."
    )
    parser.add_argument(
        "blindness_type",
        type=str,
        choices=['protanopia', 'deuteranopia', 'tritanopia'],
        help="Type of color blindness to simulate (protanopia, deuteranopia, tritanopia)."
    )

    args = parser.parse_args()

    try:
        print(f"Simulating {args.blindness_type} on {args.input_image}...")
        simulated_image = simulate_color_blindness(args.input_image, args.blindness_type)
        simulated_image.save(args.output_image)
        print(f"Simulated image saved to {args.output_image}")
    except FileNotFoundError:
        print(f"Error: Input image file '{args.input_image}' not found.")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
