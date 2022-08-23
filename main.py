import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class GradCAM:
    def __init__(
            self, model, last_convlayer_name='block14_sepconv2_act', cmap='jet',
            input_size=(256, 256)
    ):
        self.grad_model = tf.keras.models.Model(
            [model.inputs], [model.get_layer(last_convlayer_name).output, model.output]
        )
        jet = plt.cm.get_cmap(cmap)
        self.colors = jet(np.arange(256))[:, :3]
        self.input_size = input_size

    def compute_cam(self, img_tensor, pred_index):
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = self.grad_model(img_tensor)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]
        grads = tape.gradient(class_channel, last_conv_layer_output)

        conv_output = last_conv_layer_output[0]
        grad_val = grads[0]

        weights = tf.reduce_mean(grad_val, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, conv_output), axis=-1)

        cam = tf.nn.relu(cam) / tf.reduce_max(cam)
        return cam, preds

    def __call__(self, img_tensor, pred_index=None, alpha=.4):
        if len(tf.shape(img_tensor)) == 3:
            img_tensor = img_tensor[tf.newaxis, ...]
        resized = tf.image.resize(img_tensor, self.input_size)
        cam, preds = self.compute_cam(resized, pred_index)
        cam = np.uint8(255 * cam)
        jet_heatmap = self.colors[cam]
        jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((tf.shape(img_tensor)[2], tf.shape(img_tensor)[1]))
        jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap) / 255

        output = tf.clip_by_value(jet_heatmap * alpha + img_tensor, 0., 1.)
        return output, preds