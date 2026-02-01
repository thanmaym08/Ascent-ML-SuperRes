import numpy as np

def normalize_16bit_to_8bit(image):
    """
    Standardizes satellite data. 
    1. Slices multi-band data to 3-channel RGB.
    2. Normalizes 16-bit values to 8-bit using percentiles.
    """
    # Slice to RGB if the image has many bands (Sentinel-2 usually has 12+)
    if image.shape[-1] > 3:
        # Taking bands 3, 2, 1 (Red, Green, Blue)
        image = image[:, :, [3, 2, 1]]
        
    # Percentile scaling (removes glare/clouds from the math)
    low, high = np.percentile(image, (2, 98))
    image_8bit = np.clip((image - low) / (high - low) * 255, 0, 255).astype(np.uint8)
    return image_8bit

def get_tiles(image, tile_size=128):
    """Splits large satellite imagery into small patches for the AI."""
    tiles = []
    h, w, _ = image.shape
    for i in range(0, h, tile_size):
        for j in range(0, w, tile_size):
            tile = image[i:i+tile_size, j:j+tile_size]
            if tile.shape[:2] == (tile_size, tile_size):
                tiles.append(tile)
    return tiles