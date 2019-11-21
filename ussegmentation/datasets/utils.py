import numpy as np

class_to_color_mapping = {
    0: (0, 255, 0),  # other
    1: (0, 0, 255),  # sky
    2: (128, 128, 128),  # road
}


def remap_classes_to_colors(src_image):
    dst_image = np.zeros(
        (src_image.shape[0], src_image.shape[1], 3), dtype=np.uint8
    )
    for class_id, color in class_to_color_mapping.items():
        dest_r, dest_g, dest_b = color
        mask = src_image[:, :, 0] == class_id
        dst_image[:, :, :3][mask] = [dest_r, dest_g, dest_b]

    return dst_image


def remap_color_to_classes(src_image, src_color, dst_image, dest_class):
    src_r, src_g, src_b = src_color
    mask = (
        (src_image[:, :, 0] == src_r)
        & (src_image[:, :, 1] == src_g)
        & (src_image[:, :, 2] == src_b)
    )
    dst_image[:, :, 0][mask] = dest_class
    return dst_image

