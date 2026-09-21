import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

def output_path(read_path, suffix):
    return read_path.rsplit(".", 1)[0] + suffix + ".png"

def save_power(read_path, power_spectrum):
    power_spectrum = np.uint8(255 * power_spectrum / np.max(power_spectrum))  # Normalize to 0-255
    cv2.imwrite(output_path(read_path, "_power"), power_spectrum)

def save_angles(read_path, angle_bins, direction_power):
    plt.figure()
    plt.plot(angle_bins[:-1], direction_power)
    plt.xlabel("direction of variation (degrees)")
    plt.ylabel("power")
    plt.tight_layout()
    plt.savefig(output_path(read_path, "_directions"))
    plt.close()

def get_power_spectrum(image):
    fft_shifted = np.fft.fftshift(np.fft.fft2(image - np.mean(image)))
    return np.abs(fft_shifted) ** 2

def get_direction_power(power_spectrum):
    h, w = power_spectrum.shape
    y, x = np.indices((h, w))
    dx = x - w // 2
    dy = y - h // 2
    radius = np.sqrt(dx ** 2 + dy ** 2)
    angles = (np.degrees(np.arctan2(dy, dx)) + 180) % 180
    mask = radius > min(5, max(h, w) / 4)
    angle_bins = np.arange(181)
    direction_power, _ = np.histogram(angles[mask], bins=angle_bins, weights=power_spectrum[mask])
    return angle_bins, direction_power

def get_strongest_angle(direction_power):
    if np.max(direction_power) == 0:
        return 0
    return np.argmax(direction_power)

def sine_tile(shape, angle_degrees, avg_color, max_color):
    h, w = shape
    sine_image = np.full((h, w, 3), avg_color, dtype=np.uint8)
    angle = np.radians(angle_degrees)
    corners_x = np.array([-w / 2, w / 2, -w / 2, w / 2])
    corners_y = np.array([-h / 2, -h / 2, h / 2, h / 2])
    corner_along = corners_x * np.cos(angle) + corners_y * np.sin(angle)
    along_min = corner_along.min()
    along_max = corner_along.max()
    along_range = along_max - along_min
    amplitude = min(h, w) * 0.80
    periods = 4
    thickness = 2
    curve_along = np.linspace(along_min, along_max, max(h, w) * 20)
    curve_across = amplitude * np.sin(2 * np.pi * periods * (curve_along - along_min) / along_range)
    curve_x = curve_along * np.cos(angle) - curve_across * np.sin(angle) + w / 2
    curve_y = curve_along * np.sin(angle) + curve_across * np.cos(angle) + h / 2
    points = np.column_stack((curve_x, curve_y)).round().astype(np.int32)
    cv2.polylines(sine_image, [points], False, max_color.tolist(), thickness, lineType=cv2.LINE_AA)
    return sine_image

def save_sine(read_path, image, tile_size):
    if tile_size <= 0:
        raise ValueError("tile_size must be greater than 0")

    h, w = image.shape
    color_image = cv2.cvtColor(cv2.imread(read_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    sine_image = np.zeros((h, w, 3), dtype=np.uint8)

    for y0 in range(0, h, tile_size):
        y1 = min(y0 + tile_size, h)
        for x0 in range(0, w, tile_size):
            x1 = min(x0 + tile_size, w)
            image_tile = image[y0:y1, x0:x1]
            color_tile = color_image[y0:y1, x0:x1]
            avg_color = np.sqrt(np.mean(color_tile.astype(float) ** 2, axis=(0, 1))).astype(np.uint8)
            # half of avg color (dimmer)
            half_avg_color = (avg_color / 2).astype(np.uint8)
            max_color = np.max(color_tile, axis=(0, 1)).astype(np.uint8)
            tile_power = get_power_spectrum(image_tile)
            _, tile_direction_power = get_direction_power(tile_power)
            tile_angle = get_strongest_angle(tile_direction_power)
            sine_image[y0:y1, x0:x1] = sine_tile(image_tile.shape, tile_angle, avg_color, half_avg_color)

    cv2.imwrite(output_path(read_path, "_sine"), cv2.cvtColor(sine_image, cv2.COLOR_RGB2BGR))

def transform_and_save(read_path, tile_size):
    """
    Reads an image from the specified file path.

    Args:
        read_path (str): The path to the image file to read.
    """
    image = cv2.imread(read_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {read_path}")

    # get transform
    power_spectrum = get_power_spectrum(image)
    angle_bins, direction_power = get_direction_power(power_spectrum)
    strongest_angles = np.argsort(direction_power)[-5:][::-1]
    print(f"{read_path}: strongest variation directions: {strongest_angles} degrees")

    save_power(read_path, power_spectrum)
    save_angles(read_path, angle_bins, direction_power)
    save_sine(read_path, image, tile_size)
    return None

def main():
    """
    Reads an image and saves its fourier transform.
    """
    if len(sys.argv) < 2:
        print("Usage: python3 fft.py IMAGE_PATH [TILE_SIZE]")
        sys.exit(1)

    read_path = sys.argv[1]
    if not os.path.isabs(read_path):
        read_path = os.path.join(os.getcwd(), read_path)

    tile_size = 16
    if len(sys.argv) >= 3:
        tile_size = int(sys.argv[2])

    transform_and_save(read_path, tile_size)

if __name__ == "__main__":
    main()
