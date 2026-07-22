"""
Grad-CAM (Gradient-weighted Class Activation Mapping) for the
Alzheimer's detection CNN.

Produces a heatmap showing which regions of the MRI scan most
influenced the model's prediction, plus a colorized heatmap and
an overlay blended with the original image.
"""

import numpy as np
import tensorflow as tf
from PIL import Image


def find_last_conv_layer(model):
    """
    Auto-detect the name of the last convolutional layer in the model.
    Grad-CAM needs a conv layer's feature maps (spatial information
    is lost after flattening / pooling into dense layers).
    """
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found in the model.")


def make_gradcam_heatmap(model, img_array, last_conv_layer_name=None):
    """
    Compute the Grad-CAM heatmap for a single preprocessed image.

    Parameters
    ----------
    model : tf.keras.Model
        The trained (loaded) Keras model.
    img_array : np.ndarray
        Preprocessed image, shape (1, H, W, 3), already normalized
        the same way it was for prediction.
    last_conv_layer_name : str, optional
        Name of the conv layer to explain. Auto-detected if omitted.

    Returns
    -------
    heatmap : np.ndarray
        2D array in [0, 1], shape (conv_h, conv_w).
    """

    if last_conv_layer_name is None:
        last_conv_layer_name = find_last_conv_layer(model)

    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)

    # Manually forward-pass layer by layer inside the tape so we can
    # grab the intermediate conv activations. Building a separate
    # functional tf.keras.Model(inputs=..., outputs=...) on top of a
    # re-loaded Sequential model is unreliable in Keras 3, so this
    # explicit layer walk is used instead.
    with tf.GradientTape() as tape:
        x = img_tensor
        conv_output = None

        for layer in model.layers:
            x = layer(x)
            if layer.name == last_conv_layer_name:
                conv_output = x
                tape.watch(conv_output)

        predictions = x
        # Single sigmoid output neuron (binary classification)
        class_channel = predictions[:, 0]

    grads = tape.gradient(class_channel, conv_output)

    # Global-average-pool the gradients -> one importance weight per
    # feature-map channel.
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output = conv_output[0]
    heatmap = tf.reduce_sum(conv_output * pooled_grads, axis=-1)

    # Keep only the features that positively influenced the prediction.
    heatmap = tf.nn.relu(heatmap)

    max_val = tf.reduce_max(heatmap)
    heatmap = heatmap / (max_val + 1e-8)

    return heatmap.numpy()


def _colorize_heatmap(heatmap):
    """
    Map a [0, 1] heatmap to an RGB "jet"-style colormap without
    depending on matplotlib or opencv.

    Returns an (H, W, 3) uint8 array.
    """

    h = np.clip(heatmap, 0.0, 1.0)

    # Piecewise-linear approximation of the classic "jet" colormap.
    r = np.clip(1.5 - np.abs(4 * h - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * h - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * h - 1), 0, 1)

    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255).astype(np.uint8)


def generate_gradcam_images(model, img_array, display_image, alpha=0.45):
    """
    Full Grad-CAM pipeline for display in the app.

    Parameters
    ----------
    model : tf.keras.Model
    img_array : np.ndarray
        Preprocessed model input, shape (1, H, W, 3).
    display_image : PIL.Image
        The original, full-resolution uploaded image.
    alpha : float
        Strength of the heatmap overlay blended onto the original image.

    Returns
    -------
    dict with:
        heatmap_image : PIL.Image  -- colorized heatmap alone
        overlay_image : PIL.Image  -- heatmap blended over the original
    """

    heatmap = make_gradcam_heatmap(model, img_array)

    target_size = display_image.size  # (width, height)

    heatmap_img = Image.fromarray((heatmap * 255).astype(np.uint8))
    heatmap_img = heatmap_img.resize(target_size, resample=Image.BILINEAR)
    heatmap_resized = np.array(heatmap_img).astype(np.float32) / 255.0

    colored_heatmap = _colorize_heatmap(heatmap_resized)
    heatmap_image = Image.fromarray(colored_heatmap)

    original_rgb = np.array(display_image.convert("RGB")).astype(np.float32)
    blended = (1 - alpha) * original_rgb + alpha * colored_heatmap.astype(np.float32)
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    overlay_image = Image.fromarray(blended)

    return {
        "heatmap_image": heatmap_image,
        "overlay_image": overlay_image,
    }
