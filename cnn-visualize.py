from keras import Model
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
from numpy import expand_dims
import sys

"""
Code to visualize some filters and feature maps for a CNN
"""

# to allow downloading the model
# issue: https://github.com/fchollet/deep-learning-models/issues/33#issuecomment-397257502
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

model = VGG16()  # import pre-trained VGG16 model
model.summary()

# extract conv layers
conv_layers = []
for layer in model.layers:
    if 'conv' not in layer.name:
        continue
    w, _ = layer.get_weights()
    print('layer name: {}, conv shape: {}'.format(layer.name, w.shape))
    conv_layers.append(layer)


def visualize_weights(layer_idx, num_filters):
    assert 0 <= layer_idx < len(conv_layers), 'layer_idx out of bounds'
    layer = conv_layers[layer_idx]
    weight, _ = layer.get_weights()
    # normalize
    weight = weight / weight.max()  # [H,W,in_channels,out_channels]
    print(weight.shape)
    in_channels = min(4, weight.shape[2])
    num_filters = min(num_filters, weight.shape[-1])
    fig, axes = plt.subplots(num_filters, in_channels)
    for i in range(num_filters):
        w = weight[:, :, :, i]  # [H,W,in_channels]
        # loop over channels
        for channel in range(in_channels):
            axes[i, channel].imshow(w[:, :, channel], cmap='gray')
    plt.show()


def visualize_feat_maps(layer_idx, image_path):
    assert 0 <= layer_idx < len(conv_layers), 'layer_idx out of bounds'
    layer = conv_layers[layer_idx]
    # create a model such that its output is the output of the defined layer by index
    curr_model = Model(inputs=model.inputs, outputs=layer.output)
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)  # convert to numpy array
    img = expand_dims(img, axis=0)  # add batch axis to have 1 sample as input
    img = preprocess_input(img)  # prepare the image for VGG16 network
    feat_maps = curr_model.predict(img)  # [B,H,W,out_channels]
    out_channels = min(64, feat_maps.shape[-1])
    rows, cols = out_channels // 8, out_channels // 8
    print(rows, cols)
    _, axes = plt.subplots(rows, cols)
    for i in range(out_channels):
        r = i // cols
        c = i % cols
        axes[r, c].imshow(feat_maps[0, :, :, i], cmap='gray')
        axes[r, c].set_xticks([])
        axes[r, c].set_yticks([])
    plt.show()


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print('usage: %s <img>' % sys.argv[0])
        sys.exit(-1)
    img = sys.argv[1]
    #visualize_weights(1, 6)
    visualize_feat_maps(10, img)
