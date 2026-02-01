import numpy as np
import cv2

def prepare_image(image_16bit):
    # Step 1: 16-bit to 8-bit Normalization [cite: 21]
    # We take the 2nd and 98th percentile to avoid 'bright' noise
    low, high = np.percentile(image_16bit, (2, 98))
    image_8bit = np.clip((image_16bit - low) / (high - low) * 255, 0, 255).astype(np.uint8)
    return image_8bit

def create_tiles(image, tile_size=128):
    # Step 2: Tiling logic to save RAM [cite: 49]
    tiles = []
    h, w, _ = image.shape
    for i in range(0, h, tile_size):
        for j in range(0, w, tile_size):
            tile = image[i:i+tile_size, j:j+tile_size]
            if tile.shape[:2] == (tile_size, tile_size):
                tiles.append(tile)
    return tiles