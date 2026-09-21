import cv2
import numpy as np

def read_image(image_path):
    """
    Reads an image from the specified file path.

    Args:
        image_path (str): The path to the image file.
    """
    image = cv2.imread(image_path)
    return image

def main():
    """
    Reads an image and saves its fourier transform.
    """
    read_img_path = "/Users/mridu/lawson_art_proj/brick.jpeg"
    write_img_path = "/Users/mridu/lawson_art_proj/test_brick.jpg"
    image = read_image(read_img_path)

    # get transform
    f_transform = np.fft.fftshift(np.fft.fft2(image))
    # save the f_transform as an image
    f_transform_magnitude = np.abs(f_transform)
    f_transform_magnitude = np.log(f_transform_magnitude + 1)  # Log
    f_transform_magnitude = np.uint8(255 * f_transform_magnitude / np.max(f_transform_magnitude))  # Normalize to 0-255
    cv2.imwrite(write_img_path, f_transform_magnitude)

if __name__ == "__main__":
    main()