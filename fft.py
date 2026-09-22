import os
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt

def output_path(read_path, suffix, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.splitext(os.path.basename(read_path))[0] + suffix + ".png"
    return os.path.join(output_dir, filename)

def save_power(read_path, power_spectrum):
    power_spectrum = np.uint8(255 * power_spectrum / np.max(power_spectrum))  # Normalize to 0-255
    cv2.imwrite(output_path(read_path, "_power", "debugging"), power_spectrum)

def save_angles(read_path, angle_bins, direction_power):
    plt.figure()
    plt.plot(angle_bins[:-1], direction_power)
    plt.xlabel("direction of variation (degrees)")
    plt.ylabel("power")
    plt.tight_layout()
    plt.savefig(output_path(read_path, "_directions", "debugging"))
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

def keep_dominant_rgb_channel(colors):
    original_shape = colors.shape
    colors = np.reshape(colors, (-1, 3))
    result = colors.copy()
    max_channel = np.argmax(colors, axis=-1)
    green_mask = max_channel == 1
    result[green_mask] = 0
    result[green_mask, 1] = colors[green_mask, 1]
    result[~green_mask, 1] = 0
    return np.reshape(result, original_shape).astype(np.uint8)

def rgb_to_cmy(rgb):
    rgb = rgb.astype(float) / 255
    return 1 - rgb

def cmy_to_rgb(cmy):
    c = cmy[..., 0]
    m = cmy[..., 1]
    y = cmy[..., 2]
    rgb = np.stack(
        (
            1 - c,
            1 - m,
            1 - y,
        ),
        axis=-1,
    )
    return np.clip(rgb * 255, 0, 255).astype(np.uint8)

def keep_dominant_cmy_channel(colors):
    cmy = rgb_to_cmy(colors)
    result = np.zeros_like(cmy)
    max_channel = np.argmax(cmy, axis=-1)
    max_value = np.take_along_axis(cmy, np.expand_dims(max_channel, -1), axis=-1)
    np.put_along_axis(result, np.expand_dims(max_channel, -1), max_value, axis=-1)
    return cmy_to_rgb(result)

def apply_color_mode(colors, color_mode):
    if color_mode == "rgb":
        return keep_dominant_rgb_channel(colors)
    if color_mode == "cmy":
        return keep_dominant_cmy_channel(colors)
    return colors

def sine_tile(shape, angle_degrees, avg_color, wave_color, wave_tile, wave_type, thickness, periods):
    h, w = shape
    sine_image = np.full((h, w, 3), avg_color, dtype=np.uint8)
    angle = np.radians(angle_degrees)
    corners_x = np.array([-w / 2, w / 2, -w / 2, w / 2])
    corners_y = np.array([-h / 2, -h / 2, h / 2, h / 2])
    corner_along = corners_x * np.cos(angle) + corners_y * np.sin(angle)
    along_min = corner_along.min()
    along_max = corner_along.max()
    along_range = along_max - along_min
    amplitude = min(h, w) * 0.6
    curve_along = np.linspace(along_min, along_max, max(h, w) * 20)
    curve_across = amplitude * np.sin(2 * np.pi * periods * (curve_along - along_min) / along_range)
    curve_x = curve_along * np.cos(angle) - curve_across * np.sin(angle) + w / 2
    curve_y = curve_along * np.sin(angle) + curve_across * np.cos(angle) + h / 2
    points = np.column_stack((curve_x, curve_y)).round().astype(np.int32)

    if wave_type in ("original", "small"):
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.polylines(mask, [points], False, 255, thickness, lineType=cv2.LINE_AA)
        sine_image[mask > 0] = wave_tile[mask > 0]
    else:
        cv2.polylines(sine_image, [points], False, wave_color.tolist(), thickness, lineType=cv2.LINE_AA)

    return sine_image

def save_sine(read_path, image, tile_size, wave_type, thickness, periods, upscale_ratio, color_mode):
    if tile_size <= 0:
        raise ValueError("tile_size must be greater than 0")
    if upscale_ratio <= 0:
        raise ValueError("upscale_ratio must be greater than 0")

    h, w = image.shape
    color_image = cv2.cvtColor(cv2.imread(read_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    color_image = apply_color_mode(color_image, color_mode)
    small_image = None
    if wave_type == "small":
        small_path = output_path(read_path, "_sine", os.path.join("results", "small"))
        small_image = cv2.imread(small_path, cv2.IMREAD_COLOR)
        if small_image is None:
            raise FileNotFoundError(f"Could not read small sine image: {small_path}")
        small_image = cv2.cvtColor(small_image, cv2.COLOR_BGR2RGB)
        if small_image.shape[:2] != image.shape:
            small_image = cv2.resize(small_image, (w, h), interpolation=cv2.INTER_LINEAR)
        small_image = apply_color_mode(small_image, color_mode)

    output_ratio = upscale_ratio if wave_type == "flat" else 1
    sine_image = np.zeros((h * output_ratio, w * output_ratio, 3), dtype=np.uint8)

    for y0 in range(0, h, tile_size):
        y1 = min(y0 + tile_size, h)
        for x0 in range(0, w, tile_size):
            x1 = min(x0 + tile_size, w)
            image_tile = image[y0:y1, x0:x1]
            color_tile = color_image[y0:y1, x0:x1]
            wave_tile = color_tile
            if wave_type == "small":
                wave_tile = small_image[y0:y1, x0:x1]
            avg_color = np.sqrt(np.mean(color_tile.astype(float) ** 2, axis=(0, 1))).astype(np.uint8)
            # half of avg color (dimmer)
            half_avg_color = (avg_color / 2).astype(np.uint8)
            # max of avg color (brighter)
            max_avg_color = np.clip(avg_color * 1.5, 0, 255).astype(np.uint8)
            tile_power = get_power_spectrum(image_tile)
            _, tile_direction_power = get_direction_power(tile_power)
            tile_angle = get_strongest_angle(tile_direction_power)
            output_y0 = y0 * output_ratio
            output_y1 = y1 * output_ratio
            output_x0 = x0 * output_ratio
            output_x1 = x1 * output_ratio
            output_shape = (output_y1 - output_y0, output_x1 - output_x0)
            sine_image[output_y0:output_y1, output_x0:output_x1] = sine_tile(
                output_shape,
                tile_angle,
                half_avg_color,
                avg_color,
                wave_tile,
                wave_type,
                round(thickness * output_ratio) if wave_type == "flat" else round(thickness),
                periods,
            )

    cv2.imwrite(output_path(read_path, "_sine", "results"), cv2.cvtColor(sine_image, cv2.COLOR_RGB2BGR))

def transform_and_save(read_path, tile_size, wave_type, thickness, periods, upscale_ratio, color_mode):
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
    save_sine(read_path, image, tile_size, wave_type, thickness, periods, upscale_ratio, color_mode)
    return None

def main():
    """
    Reads an image and saves its fourier transform.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", default="belltower.jpg")
    parser.add_argument("-s", "--size", type=int, default=100)
    parser.add_argument("-w", "--wave", choices=["flat", "original", "small"], default="flat")
    parser.add_argument("-t", "--thickness", type=float, default=3)
    parser.add_argument("-p", "--periodicity", type=int, default=6)
    parser.add_argument("-r", "--ratio", type=int, default=1)
    parser.add_argument("-c", "--color-mode", choices=["none", "rgb", "cmy"], default="none")
    args = parser.parse_args()

    read_path = args.image
    if not os.path.isabs(read_path):
        read_path = os.path.join(os.getcwd(), "data", read_path)

    transform_and_save(read_path, args.size, args.wave, args.thickness, args.periodicity, args.ratio, args.color_mode)

if __name__ == "__main__":
    main()
