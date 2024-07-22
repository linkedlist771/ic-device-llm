import numpy as np
import cv2
from loguru import logger
from PIL import Image
import os


def visualize_data(file_path, output_dir, fps=10, start_time=0, end_time=None, title_font_size=1,
                   title_color=(255, 255, 255),
                   use_title: bool = False, save: bool = False):
    """
    Visualize data from a text file and save as grayscale images and a GIF.

    Args:
    file_path (str): Path to the input text file.
    output_dir (str): Directory to save output images and GIF.
    fps (int): Frames per second for the GIF. Default is 10.
    start_time (int): Start time in seconds. Default is 0.
    end_time (int): End time in seconds. If None, process all data. Default is None.
    title_font_size (float): Font size for the title. Default is 1.
    title_color (tuple): RGB color for the title. Default is (255, 255, 255) (white).

    Returns:
    List of grayscale numpy arrays representing the images.
    """
    # Create output directory if it doesn't exist
    if save:
        os.makedirs(output_dir, exist_ok=True)

    # Load data
    data = np.loadtxt(file_path)

    # Create a list to store all images
    images = []
    images_array = []
    __new_data_matrix = np.zeros(data.shape)
    substract_masks_idx = np.where((np.flipud(data) <= 0))
    substract_masks = np.zeros(data.shape)

    # Set all values at these indices to 0.5
    substract_masks[substract_masks_idx] = 0.5

    # Determine end time if not provided
    if end_time is None:
        end_time = int(np.max(data)) + 1

    # Process data for each second
    for second in range(start_time, end_time):
        lower_time = second
        upper_time = second + 1

        # Select data within current time range
        idx = np.where((data >= lower_time) & (data < upper_time))
        new_data_matrix = np.zeros(data.shape)

        new_data_matrix[idx] = 1
        new_data_matrix = np.flipud(new_data_matrix)
        __new_data_matrix += new_data_matrix

        # Normalize data to 0-255 range
        normalized_data = cv2.normalize(__new_data_matrix, None, 0, 255, cv2.NORM_MINMAX)

        # Convert to 8-bit unsigned integer
        img = np.uint8(normalized_data)

        img[substract_masks_idx] = 127

        # Keep the image in grayscale
        if use_title:
            # Create a color image only for adding text
            img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            # Add time title
            cv2.putText(img_color, f"Time: {second}-{second + 1} seconds", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, title_font_size, title_color, 2)
            # Convert back to grayscale
            img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

        if save:
            # Save image
            cv2.imwrite(os.path.join(output_dir, f"frame_{second:03d}.png"), img)

        # Convert to PIL Image
        pil_img = Image.fromarray(img)
        images.append(pil_img)
        images_array.append(img)

    # Save as GIF
    if save:
        gif_path = os.path.join(output_dir, "data_visualization.gif")
        images[0].save(gif_path, save_all=True, append_images=images[1:],
                       optimize=False, duration=int(1000 / fps), loop=0)
        logger.info(f"GIF saved as {gif_path}")

    logger.info(f"Data shape: {data.shape}")
    return images_array


if __name__ == "__main__":
    from src.ic_device_llm.configs.path_config import RESOURCES_DIR_PATH

    input_txt_path = RESOURCES_DIR_PATH / 'deposition' / 'all data0921' / 'sub1' / 'results_txt' / 't1.txt'

    res = visualize_data(input_txt_path, "output_folder", fps=15, start_time=0, end_time=156, title_font_size=1.2,
                         title_color=(255, 0, 0), save=True)
    print(res[0].shape)  # This will now print the shape of a 2D grayscale array
