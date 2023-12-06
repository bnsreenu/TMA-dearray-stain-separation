"""
This module contains functions for color separation, nuclei segmentation, and analysis of TMA core images.

Functions:
- `analyze_TMA_core`: Performs color separation, nuclei segmentation, and intensity analysis on a TMA core image.
- `color_separate_only`: Performs color separation without nuclei segmentation, useful for previewing channels.
- `color_separate`: Separates H, E, D stains and optionally combines H and D for fluorescence effect.
- `segment_nuclei`: Segments nuclei in the H component using StarDist.
- `filter_objects`: Filters segmented nuclei based on size.
- `brown_intensity_from_nuclei`: Extracts brown intensity from nuclei locations.

"""



import numpy as np
from skimage import img_as_ubyte
from skimage.color import rgb2hed, hed2rgb
from skimage.exposure import rescale_intensity
from stardist.models import StarDist2D
from stardist.plot import render_label
from csbdeep.utils import normalize
from skimage import measure
import pandas as pd

def analyze_TMA_core(image, min_size=30, max_size=1000):
    """
    Analyzes a TMA core image by performing color separation, nuclei segmentation, and filtering based on size.

    Args:
    - image: Input TMA core image.
    - min_size: Minimum size of nuclei to consider.
    - max_size: Maximum size of nuclei to consider.

    Returns:
    - Tuple of image, H component, brown component, filtered segmentation image, means, and standard deviations.
    """
    print("Now color separating ...")
    H, _, brown, _ = color_separate(image)

    print("Now segmenting nuclei using StarDist ...")
    nuclei_segm = segment_nuclei(H)

    print("Now filtering nuclei based on user-defined min and max ...")
    filtered_segm_image = filter_objects(nuclei_segm, image, min_size, max_size)

    print("Now extracting mean and std dev. brown intensity values from Nuclei ...")
    means, stdevs = brown_intensity_from_nuclei(filtered_segm_image, brown)

    return image, H, brown, filtered_segm_image, means, stdevs


def color_separate_only(image):
    """
    Performs color separation without segmentation, useful for previewing separated channels.

    Args:
    - image: Input RGB image.

    Returns:
    - Tuple of input image, H component, and brown component.
    """
    print("Now color separating ...")
    H, _, brown, _ = color_separate(image)

    return image, H, brown


def color_separate(image_rgb):
    """
    Performs color separation on an RGB image.

    Args:
    - image_rgb: Input RGB image.

    Returns:
    - Tuple of H, E, D components, and a combined image for visualization.
    """
    # Convert the RGB image to HED using the prebuilt skimage method
    image_hed = rgb2hed(image_rgb)

    # Create an RGB image for each of the separated stains
    # Convert them to ubyte for easy saving to drive as an image
    null = np.zeros_like(image_hed[:, :, 0])
    image_h = img_as_ubyte(hed2rgb(np.stack((image_hed[:, :, 0], null, null), axis=-1)))
    image_e = img_as_ubyte(hed2rgb(np.stack((null, image_hed[:, :, 1], null), axis=-1)))
    image_d = img_as_ubyte(hed2rgb(np.stack((null, null, image_hed[:, :, 2]), axis=-1)))

    # Optional fun exercise of combining H and DAB stains into a single image with fluorescence look
    h = rescale_intensity(image_hed[:, :, 0], out_range=(0, 1),
                          in_range=(0, np.percentile(image_hed[:, :, 0], 99)))
    d = rescale_intensity(image_hed[:, :, 2], out_range=(0, 1),
                          in_range=(0, np.percentile(image_hed[:, :, 2], 99)))

    # Cast the two channels into an RGB image, as the blue and green channels
    # Convert to ubyte for easy saving as image to local drive
    zdh = img_as_ubyte(np.dstack((null, d, h)))  # DAB in green and H in Blue

    return image_h, image_e, image_d, zdh


def segment_nuclei(H_img):
    """
    Segments nuclei using StarDist.

    Args:
    - H_img: Input H component image.

    Returns:
    - Nuclei segmentation labels.
    """
    # Define a pretrained model to segment nuclei in fluorescence images (download from pretrained)
    model = StarDist2D.from_pretrained('2D_versatile_fluo')

    H_inverted = np.invert(H_img)
    H_gray = H_inverted[:, :, 0]
    H_labels, _ = model.predict_instances(normalize(H_gray), nms_thresh=0.3, prob_thresh=0.6)

    return H_labels


def filter_objects(H_labels, intensity_image, min_size=30, max_size=1000):
    """
    Filters segmented objects based on size.

    Args:
    - H_labels: Nuclei segmentation labels.
    - intensity_image: Original intensity image.
    - min_size: Minimum size of nuclei to consider.
    - max_size: Maximum size of nuclei to consider.

    Returns:
    - Filtered segmentation labels.
    """
    props = measure.regionprops_table(H_labels, intensity_image,
                                      properties=['label',
                                                  'area', 'equivalent_diameter',
                                                  'mean_intensity', 'solidity', 'centroid'])

    df = pd.DataFrame(props)

    # Filter objects by size
    df_filtered = df[df.area > min_size]
    df_filtered = df_filtered[df_filtered.area < max_size]

    # Filter objects from the labeled image
    useful_labels = df_filtered.label.values  # Labels of objects that passed our filter criteria
    filtered_H_labels = np.zeros_like(H_labels)  # Array same size as labeled image but with 0s, to be filled later
    for i in range(H_labels.shape[0]):
        for j in range(H_labels.shape[1]):
            if (H_labels[i, j] in useful_labels) == True:
                filtered_H_labels[i, j] = H_labels[i, j]
            else:
                filtered_H_labels[i, j] = 0

    return filtered_H_labels


def brown_intensity_from_nuclei(filtered_H_labels, brown_image):
    """
    Extracts brown intensity from nuclei locations.

    Args:
    - filtered_H_labels: Filtered nuclei segmentation labels.
    - brown_image: Brown intensity image.

    Returns:
    - Tuple of means and standard deviations of brown intensity.
    """
    DAB_intensity_from_H_filtered = measure.regionprops_table(filtered_H_labels, brown_image,
                                                             properties=['label', 'mean_intensity'])
    df_DAB_intensity = pd.DataFrame(DAB_intensity_from_H_filtered)

    mean_R = df_DAB_intensity["mean_intensity-0"].mean()
    mean_G = df_DAB_intensity["mean_intensity-1"].mean()
    mean_B = df_DAB_intensity["mean_intensity-2"].mean()

    std_R = df_DAB_intensity["mean_intensity-0"].std()
    std_G = df_DAB_intensity["mean_intensity-1"].std()
    std_B = df_DAB_intensity["mean_intensity-2"].std()

    means = [mean_R, mean_G, mean_B]
    stdevs = [std_R, std_G, std_B]

    return means, stdevs
