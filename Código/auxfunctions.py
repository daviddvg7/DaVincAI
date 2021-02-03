# SISTEMAS INTELIGENTES
# Ingeniería Informática - 4º Año
# Proyecto de Transferencia de Estilos
#   Jorge Acevedo de León - alu0101123622
#   Rafael Cala González - alu0101121901
#   David Valverde Gómez - alu0101100296
#
# Fichero: auxfunctions.py
#   En este fichero se definen funciones necesarias para
#   la preparación y ejecución del modelo de transferencia de
#   estilos

from keras import backend as be
import time
import cv2
import functools
from PIL import Image
import PIL.Image
import numpy as np
from IPython import get_ipython
import matplotlib as mpl
import matplotlib.pyplot as plt
import IPython.display as display
import tensorflow as tf
import os
import sys

# MUY IMPORTANTE DEJAR AQUÍ, ANTES DE LA CARGA DE TENSORFLOW
# Cargar modelos comprimidos de tensorflow_hub

os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


# Ajustes de matplotlib
def set_matplotlib():
  mpl.rcParams['figure.figsize'] = (12, 12)
  mpl.rcParams['axes.grid'] = False


# Función encargada de transformar tensores en imágenes con Pillow
def tensor_to_image(tensor):
  tensor = tensor*255
  tensor = np.array(tensor, dtype=np.uint8)
  if np.ndim(tensor) > 3:
    assert tensor.shape[0] == 1
    tensor = tensor[0]
  return PIL.Image.fromarray(tensor)


# Función para cargar imagenes y transformarlas al formato necesario (512x512, tres canales (RGB), array tf.float32)
def load_img(path_to_img):
  max_dim = 512
  img = tf.io.read_file(path_to_img)
  img = tf.image.decode_image(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float32)

  shape = tf.cast(tf.shape(img)[:-1], tf.float32)
  long_dim = max(shape)
  scale = max_dim / long_dim

  new_shape = tf.cast(shape * scale, tf.int32)

  img = tf.image.resize(img, new_shape)
  img = img[tf.newaxis, :]

  return img

# Función para mostrar varias imágenes de manera simultánea
def imshow(images, titles=None, rows=1, cols=2):

  ax = plt.subplots(nrows=rows, ncols=cols)

  for i in range(len(images)):
    if len(images[i].shape) > 3:
        images[i] = tf.squeeze(images[i], axis=0)
    ax[1].ravel()[i].imshow(images[i])
    ax[1].ravel()[i].set_title(titles[i])
    ax[1].ravel()[i].set_axis_off()
  plt.tight_layout()
  plt.show()


def vgg_layers(layer_names):
  # Cargamos nuestro modelo y una red preentrenada VGG19
  vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
  vgg.trainable = False

  outputs = [vgg.get_layer(name).output for name in layer_names]

  model = tf.keras.Model([vgg.input], outputs)
  return model


# La matriz de Gram es una relación entre las capas de estilo de una imagen que nos permite
# obtener los datos necesarios para describir el estilo de la imagen
def gram_matrix(input_tensor):
  result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
  input_shape = tf.shape(input_tensor)
  num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
  return result/(num_locations)

# Esta función normaliza los valores de los pixels de la imagen entre 0 y 1
def clip_0_1(image):
  return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

# Función que calcula la pérdida de estilo y de contenido de la imagen resultado con respecto a las originales
def style_content_loss(outputs, style_targets,
                       style_weight, content_weight, num_style_layers, num_content_layers, content_targets):
  style_outputs = outputs['style']
  content_outputs = outputs['content']
  style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_targets[name])**2)
                          for name in style_outputs.keys()])
  style_loss *= style_weight / num_style_layers

  content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2)
                            for name in content_outputs.keys()])
  content_loss *= content_weight / num_content_layers
  loss = style_loss + content_loss
  return loss


# Función encargada de generar un collage con las 3 imágenes recibidas
def generate_collage(content_image_name, style_image_name, file_name):

  # Se abren las imágenes
  images = [Image.open(x) for x in [content_image_name,
                                    style_image_name, file_name]]

  # Se ajusta el tamaño de las imágenes
  widths, heights = zip(*(i.size for i in images))

  total_width = widths[2] * 2 + widths[1]
  max_height = heights[2]

  new_im = Image.new('RGB', (total_width, max_height))
  images[0] = images[0].resize((widths[2], heights[2]))
  images[1] = images[1].resize((widths[1], heights[2]))

  # Se crea una nueva imagen donde se añaden las 3 especificadas
  # para formar el collage
  x_offset = 0
  for im in images:
    new_im.paste(im, (x_offset, 0))
    x_offset += im.size[0]

  # Se establece un nuevo nombre para la imagen y se guarda
  file_name = file_name[:-4] + "_collage.jpg"
  new_im.save(file_name)
