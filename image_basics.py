import numpy as np
import SimpleITK as sitk


# --- DO NOT CHANGE ---
def _get_registration_method(atlas_img, img) -> sitk.ImageRegistrationMethod:
    registration_method = sitk.ImageRegistrationMethod()

    # Similarity metric settings.
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.REGULAR)
    registration_method.SetMetricSamplingPercentage(0.2)

    registration_method.SetMetricUseFixedImageGradientFilter(False)
    registration_method.SetMetricUseMovingImageGradientFilter(False)

    registration_method.SetInterpolator(sitk.sitkLinear)

    # Optimizer settings.
    registration_method.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=100,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10,
    )
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # Setup for the multi-resolution framework.
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Set initial transform
    initial_transform = sitk.CenteredTransformInitializer(
        atlas_img,
        img,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )
    registration_method.SetInitialTransform(initial_transform, inPlace=False)
    return registration_method
# --- DO NOT CHANGE ---


def load_image(img_path, is_label_img):
    """
    Args:
        img_path: Path to the image file to be loaded.
        is_label_img: Boolean flag to indicate if the image is a label image.

    Returns:
        The loaded image as a SimpleITK image object.
    """
    pixel_type = sitk.sitkUInt8 if is_label_img else sitk.sitkFloat32  # todo: modify here
    img = sitk.ReadImage(img_path, outputPixelType=pixel_type)  # todo: modify here

    return img


def to_numpy_array(img):
    """
    Converts a SimpleITK image to a NumPy array.

    Args:
        img: A SimpleITK Image object.

    Returns:
        A NumPy array representation of the input image.
    """
    np_img = sitk.GetArrayFromImage(img)

    return np_img


def to_sitk_image(np_image, reference_img):
    """
    Args:
        np_image: numpy array representation of an image.
        reference_img: A SimpleITK image used as a reference for copying image information.

    Returns:
        A SimpleITK image with the same spatial information as the reference image.
    """

    img = sitk.GetImageFromArray(np_image)
    img.CopyInformation(reference_img)

    return img


def preprocess_rescale_numpy(np_img, new_min_val, new_max_val):
    """
    Args:
        np_img: The NumPy array representing the image to be rescaled.
        new_min_val: The new minimum value for the rescaled image.
        new_max_val: The new maximum value for the rescaled image.

    Returns:
        A NumPy array with the image values rescaled to the specified range.
    """
    max_val = np_img.max()
    min_val = np_img.min()

    normalized_np_img = (np_img - min_val) / (max_val - min_val)
    rescaled_np_img = normalized_np_img * (new_max_val - new_min_val) + new_min_val

    return rescaled_np_img


def preprocess_rescale_sitk(img, new_min_val, new_max_val):
    """
    Args:
        img: Input image to be rescaled.
        new_min_val: New minimum intensity value.
        new_max_val: New maximum intensity value.

    Returns:
        Image with rescaled intensity values.
    """
    # Ensure image is in correct format
    img = sitk.Cast(img, sitk.sitkFloat32)

    # Rescale intensity
    rescaled_img = sitk.RescaleIntensity(img, newMinimum=new_min_val, newMaximum=new_max_val)

    return rescaled_img


def register_images(img, label_img, atlas_img):
    """
    Args:
        img: The floating image that needs to be registered.
        label_img: The label image associated with the floating image.
        atlas_img: The fixed atlas image to which the floating and label images will be registered.

    Returns:
        The registered floating image and the registered label image.

    """
    registration_method = _get_registration_method(
        atlas_img, img
    )  # type: sitk.ImageRegistrationMethod
    transform = registration_method.Execute(atlas_img, img)  # todo: modify here

    # todo: apply the obtained transform to register the image (img) to the atlas image (atlas_img)
    # hint: 'Resample' (with referenceImage=atlas_img, transform=transform, interpolator=sitkLinear,
    # defaultPixelValue=0.0, outputPixelType=img.GetPixelIDValue())
    registered_img = sitk.Resample(label_img, atlas_img, transform, sitk.sitkLinear, 0.0, img.GetPixelIDValue())

    # todo: apply the obtained transform to register the label image (label_img) to the atlas image (atlas_img), too
    # be careful with the interpolator type for label images!
    # hint: 'Resample' (with interpolator=sitkNearestNeighbor, defaultPixelValue=0.0,
    # outputPixelType=label_img.GetPixelIDValue())
    registered_label = sitk.Resample(label_img, atlas_img, transform, sitk.sitkNearestNeighbor, 0.0, label_img.GetPixelIDValue())

    return registered_img, registered_label


def extract_feature_median(img):
    """
    Args:
        img: An image in SimpleITK format from which the median feature is to be extracted.

    Returns:
        A new image with the median filter applied, in SimpleITK format.
    """
    median_img = sitk.Median(img)

    return median_img


def postprocess_largest_component(label_img):
    """
    Args:
        label_img: A SimpleITK image representing the labeled image for which
                   the largest connected component needs to be extracted.

    Returns:
        A SimpleITK image where only the largest connected component is retained,
        other components are set to the background.
    """
    connected_components = sitk.ConnectedComponent(label_img)

    # todo: order the component by ascending component size (hint: 'RelabelComponent')
    relabeled_components = sitk.RelabelComponent(connected_components, sortByObjectSize=True)

    largest_component = relabeled_components == 1  # zero is background
    return largest_component
