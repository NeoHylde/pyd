import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['figure.figsize'] = (8,8)
mpl.rcParams['axes.grid'] = False

# Load pretrained model
pretrained_model = tf.keras.applications.MobileNetV2(include_top=True,
                                                     weights='imagenet')
pretrained_model.trainable = False

# ImageNet labels
decode_predictions = tf.keras.applications.mobilenet_v2.decode_predictions

def preprocess(image):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (224,224))
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    image = image[None, ...]
    return image

def get_imagenet_label(probs):
    return decode_predictions(probs, top=1)[0][0]

loss_object = tf.keras.losses.CategoricalCrossentropy()

def create_adversarial_pattern(input_image, input_label):
    with tf.GradientTape() as tape:
        tape.watch(input_image)
        predication = pretrained_model(input_image)
        loss = loss_object(input_label, predication)

    gradient = tape.gradient(loss, input_image)
    signed_grad = tf.sign(gradient)
    return signed_grad

def display_images(image):
  _, label, confidence = get_imagenet_label(pretrained_model.predict(image))
  plt.figure()
  plt.imshow(image[0]*0.5+0.5)
  plt.title('{} : {:.2f}% Confidence'.format(label, confidence*100))
  plt.show()

def fgsm(image_path, epsilon):
    image_raw = tf.io.read_file(image_path)
    image = tf.image.decode_image(image_raw)

    image = preprocess(image)
    image_probs = pretrained_model.predict(image)

    # Get the input label of the image.
    labrador_retriever_index = 208
    label = tf.one_hot(labrador_retriever_index, image_probs.shape[-1])
    label = tf.reshape(label, (1, image_probs.shape[-1]))

    perturbations = create_adversarial_pattern(image, label)
    plt.imshow(perturbations[0] * 0.5 + 0.5);  # To change [-1, 1] to [0,1]

    adv_x = image + epsilon*perturbations
    adv_x = tf.clip_by_value(adv_x, -1, 1)
    tf.keras.utils.save_img("fgsm_img.jpg", adv_x[0])
    
    return adv_x

if __name__ == "__main__":
    img = fgsm("YellowLabradorLooking_new.jpg", 0.1)
    display_images(img)