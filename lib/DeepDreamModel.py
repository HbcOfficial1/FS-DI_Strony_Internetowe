import tensorflow as tf
import PIL


def random_roll(img, maxroll):
    # Randomly shift the image to avoid tiled boundaries.
    shift = tf.random.uniform(shape=[2], minval=-maxroll, maxval=maxroll,
                              dtype=tf.int32)
    img_rolled = tf.roll(img, shift=shift, axis=[0, 1])
    return shift, img_rolled


class TiledGradients(tf.Module):
    def __init__(self, model):
        self.model = model

    @tf.function(
        input_signature=(
                tf.TensorSpec(shape=[None, None, 3], dtype=tf.float32),
                tf.TensorSpec(shape=[], dtype=tf.int32),)
    )
    def __call__(self, img, tile_size=512):
        shift, img_rolled = random_roll(img, tile_size)

        # Initialize the image gradients to zero.
        gradients = tf.zeros_like(img_rolled)

        # Skip the last tile, unless there's only one tile.
        xs = tf.range(0, img_rolled.shape[0], tile_size)[:-1]
        if not tf.cast(len(xs), bool):
            xs = tf.constant([0])
        ys = tf.range(0, img_rolled.shape[1], tile_size)[:-1]
        if not tf.cast(len(ys), bool):
            ys = tf.constant([0])

        for x in xs:
            for y in ys:
                # Calculate the gradients for this tile.
                with tf.GradientTape() as tape:
                    # This needs gradients relative to `img_rolled`.
                    # `GradientTape` only watches `tf.Variable`s by default.
                    tape.watch(img_rolled)

                    # Extract a tile out of the image.
                    img_tile = img_rolled[x:x + tile_size, y:y + tile_size]
                    loss = calc_loss(img_tile, self.model)

                # Update the image gradients for this tile.
                gradients = gradients + tape.gradient(loss, img_rolled)

        # Undo the random shift applied to the image and its gradients.
        gradients = tf.roll(gradients, shift=-shift, axis=[0, 1])

        # Normalize the gradients.
        gradients /= tf.math.reduce_std(gradients) + 1e-8

        return gradients


def calc_loss(img, model):
    # Pass forward the image through the model to retrieve the activations.
    # Converts the image into a batch of size 1.
    img_batch = tf.expand_dims(img, axis=0)
    layer_activations = model(img_batch)
    if len(layer_activations) == 1:
        layer_activations = [layer_activations]

    losses = []
    for act in layer_activations:
        loss = tf.math.reduce_mean(act)
        losses.append(loss)

    return tf.reduce_sum(losses)


def run_deepdream(img, steps_per_octave=100, step_size=0.02,
                  octaves=range(-2, 3), octave_scale=1.3):
    # Adam optimizer params
    lr = tf.constant(step_size)
    beta1 = tf.constant(0.9)
    beta2 = tf.constant(0.999)
    epsilon = tf.constant(1e-8)

    base_shape = tf.shape(img)
    img = tf.keras.preprocessing.image.img_to_array(img)[:, :, :3]  # drop alpha
    img = tf.keras.applications.inception_v3.preprocess_input(img)

    initial_shape = img.shape[:-1]
    img = tf.image.resize(img, initial_shape)
    for octave in octaves:
        # Scale the image based on the octave
        new_size = tf.cast(tf.convert_to_tensor(base_shape[:-1]), tf.float32)\
                   * \
                   (octave_scale ** octave)
        img = tf.image.resize(img, tf.cast(new_size, tf.int32))
        t = 1
        m = 0
        v = 0

        for step in range(steps_per_octave):
            gradients = get_tiled_gradients(img)

            m = beta1 * m + (1 - beta1) * gradients
            v = beta2 * v + (1 - beta2) * (gradients ** 2)

            m_reg = m / (1 - beta1 ** t)
            v_reg = v / (1 - beta2 ** t)
            new_grads = (lr * m_reg) / (tf.sqrt(v_reg + epsilon))

            t += 1

            img = img + new_grads
            img = tf.clip_by_value(img, -1, 1)

    result = tf.cast(255 * (img + 1.0) / 2.0, tf.uint8)
    result = tf.image.convert_image_dtype(result, dtype=tf.uint8).numpy()
    result = PIL.Image.fromarray(result)
    return result


dream_model = tf.keras.models.load_model('models/dream_model.h5')
get_tiled_gradients = TiledGradients(dream_model)
