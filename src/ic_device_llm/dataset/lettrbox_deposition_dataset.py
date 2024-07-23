import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from loguru import logger
import argparse

from src.ic_device_llm.configs.path_config import RESOURCES_DIR_PATH
from src.ic_device_llm.utils.data.augmentation import letterbox

PROCESSED_DATA_DIR_PATH = RESOURCES_DIR_PATH / "deposition" / "processed_data"
LETTERBOXED_DATA_SAVE_DIR_PATH = RESOURCES_DIR_PATH / "deposition" / "letter_box_384x512_processed_data"


def process_npz(npz_file_path: Path):
    name = npz_file_path.stem
    with np.load(npz_file_path) as data:
        images = data["images"]  # sequence_length x height x width
        target_size = (384, 512)
        sequence_length, height, width = images.shape
        letterboxed_images = np.zeros((sequence_length, *target_size), dtype=np.uint8)
        for i in tqdm(range(sequence_length), desc=f"Processing {name}"):
            letterboxed_image, _, _ = letterbox(images[i], new_shape=target_size, auto=True, scaleup=True, stride=32)
            letterboxed_images[i] = letterboxed_image

        # Create a new dictionary with modified data
        new_data = {key: data[key] for key in data.keys()}
        new_data["images"] = letterboxed_images

    # Save letterboxed images
    save_path = LETTERBOXED_DATA_SAVE_DIR_PATH / f"{name}.npz"
    np.savez(save_path, **new_data)
    logger.info(f"Saved letterboxed images to {save_path}")


def test_letterbox():
    # Test letterbox function
    test_file_path = Path(
        'C:/Users/23174/Desktop/GitHub Project/ic-device-llm/resources/deposition/processed_data/sub1_5.npz')
    if not test_file_path.exists():
        logger.error(f"Test file not found: {test_file_path}")
        return

    data = np.load(test_file_path)
    input_image = data["images"][40]
    logger.info(f"Input image shape: {input_image.shape}")
    target_size = (128 + 256, 512)
    cv2.imshow("input_image", input_image)
    letterboxed_image, ratio, pad = letterbox(input_image, new_shape=target_size, auto=True, scaleup=True, stride=32)
    logger.info(f"Letterboxed image shape: {letterboxed_image.shape}")
    cv2.imshow("letterboxed_image", letterboxed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main(input_dir: Path, output_dir: Path):
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    for npz_file in tqdm(input_dir.glob("*.npz"), desc="Processing NPZ files"):
        try:
            process_npz(npz_file)
        except Exception as e:
            logger.error(f"Error processing {npz_file}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process NPZ files with letterbox.")
    parser.add_argument("--input", type=Path, default=PROCESSED_DATA_DIR_PATH,
                        help="Input directory containing NPZ files")
    parser.add_argument("--output", type=Path, default=LETTERBOXED_DATA_SAVE_DIR_PATH,
                        help="Output directory for processed NPZ files")
    args = parser.parse_args()


    main(args.input, args.output)