# SISTEMAS INTELIGENTES
# Ingeniería Informática - 4º Año
# Proyecto de Transferencia de Estilos
#   Jorge Acevedo de León - alu0101123622
#   Rafael Cala González - alu0101121901
#   David Valverde Gómez - alu0101100296
#
# Fichero: style-transfer.py
#   En este fichero tiene lugar la ejecución principal
#   de la transferencia de estilos: se preparan el modelo
#   y los datos a utilizar, se define una función que establece
#   cada paso del entrenamiento y otra función que ejecuta estos
#   pasos un número establecido de veces, tras lo que se guarda la imagen
#   final. Por último se cuenta con una función principal que recibe y carga
#   las imágenes antes de ejecutar la transferencia

from StyleContentModel import *
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
import platform

# MUY IMPORTANTE DEJAR AQUÍ, ANTES DE LA CARGA DE TENSORFLOW
# Cargar modelos comprimidos de tensorflow_hub

os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


# Funcion que 'prepara el terreno' para realizar la transferencia de estilos

def style_transfer(content_image, style_image, file_name):
  be.clear_session()
  set_matplotlib()

  # Seleccionamos capas intermedias de la red para representar el estilo y el contenido de la imagen:
  content_layers = ['block5_conv2']

  style_layers = ['block1_conv1',
                  'block2_conv1',
                  'block3_conv1',
                  'block4_conv1',
                  'block5_conv1']

  num_content_layers = len(content_layers)
  num_style_layers = len(style_layers)

  style_extractor = vgg_layers(style_layers)
  #style_outputs = style_extractor(style_image*255)

  # Cuando se llama a una imagen, este modelo devuelve la matriz gramatical (estilo) de style_layers y el contenido de content_layers:
  extractor = StyleContentModel(style_layers, content_layers)
  #results = extractor(tf.constant(content_image))

  # Establecemos nuestros valores de estilo y contenido:
  style_targets = extractor(style_image)['style']
  content_targets = extractor(content_image)['content']

  # Definimos una tf.Variable para contener la imagen a optimizar:
  image = tf.Variable(content_image)

  # Creamos una función de optimización; optamos por Adam:
  opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

  # Asignamos pesos a estilo y contenido:
  style_weight = 1e-2
  content_weight = 1e4

  # Procedemos a la ejecución del modelo
  train_model_jpg_output(image, extractor, style_targets, style_weight, content_weight,
                          num_style_layers, num_content_layers, content_targets, opt, file_name)


# Nueva función de entrenamiento utilizando la variación total
@tf.function()
def train_step(image, extractor, style_targets,
               style_weight, content_weight, num_style_layers, num_content_layers, content_targets, opt):
  # Una desventaja de esta implementación básica es que produce muchos artefactos de alta frecuencia.
  # Disminuya estos usando un término de regularización explícito en los componentes de alta frecuencia de la imagen.
  total_variation_weight = 30
  with tf.GradientTape() as tape:
      outputs = extractor(image)
      loss = style_content_loss(outputs, style_targets,
                                style_weight, content_weight, num_style_layers, num_content_layers, content_targets)
      loss += total_variation_weight*tf.image.total_variation(image)

  grad = tape.gradient(loss, image)
  opt.apply_gradients([(grad, image)])
  image.assign(clip_0_1(image))
  return loss

# Se entrena el modelo con 1000 iteraciones en rangos de 100 y se guarda la imagen final
def train_model_jpg_output(image, extractor, style_targets,
                           style_weight, content_weight, num_style_layers, num_content_layers, content_targets, opt, file_name):
  start = time.time()
  epochs = 10
  steps_per_epoch = 100
  step = 0
  for n in range(epochs):
    for m in range(steps_per_epoch):
      step += 1
      loss = train_step(image, extractor, style_targets,
                        style_weight, content_weight, num_style_layers, num_content_layers, content_targets, opt)
      print(".", end='')
    # display.clear_output(wait=True)
    # display.display(tensor_to_image(image))
    print("Train step: {}".format(step))
    current = time.time()
    print("Train time: {:.1f}".format(current-start))
    tf.print("Current loss: ", loss)

  end = time.time()
  print("---- Process Succeeded ---- ")
  print("Total time: {:.1f}".format(end-start))
  tf.print("Total loss: ", loss)

  tensor_to_image(image).save(file_name)

# Función principal
def main(content_image_name, style_image_name, result_image_name, collage, chain):

  # Carga de imágenes
  content_image = load_img(content_image_name)
  style_image = load_img(style_image_name)

  # Si no se va a ejecutar una cadena de estilos, se muestran las imágenes inicialmente
  if chain == "0":
    imshow([content_image, style_image], ["Content Image", "Style Image"])
  
  
  if platform.system() == "Linux":
    os.system(f"clear")
  elif platform.system() == "Windows":
    os.system(f"cls")


  # Llamada a la transferencia de estilos
  style_transfer(content_image, style_image, result_image_name)

  # Si no se va a ejecutar una cadena de estilos, se muestran las imágenes inciales y el resultado final
  if chain == "0":
    result_image = load_img(result_image_name)
    imshow([content_image, style_image, result_image], [
            "Content Image", "Style Image", "Result Image"], 1, 3)

  # Si se indica, se realiza un collage con las 3 imágenes
  if collage == "y":
    generate_collage(content_image_name,
                      style_image_name, result_image_name)


main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])