# SISTEMAS INTELIGENTES
# Ingeniería Informática - 4º Año
# Proyecto de Transferencia de Estilos
#   Jorge Acevedo de León - alu0101123622
#   Rafael Cala González - alu0101121901
#   David Valverde Gómez - alu0101100296
#
# Fichero: StyleContentModel.py
#   En este fichero se define la clase StyleContentModel
#   representativa de un modelo que nos permitirá
#   calcular los valores de estilo y contenido para
#   nuestras imágenes. Deberá importar todas las funciones
#   definidas en el fichero auxfunctions.py

from auxfunctions import *
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

# Carga de modelos comprimidos de tensorflow
os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# Construimos un modelo que devuelva los valores de estilo y contenido:
class StyleContentModel(tf.keras.models.Model):
  def __init__(self, style_layers, content_layers):
    super(StyleContentModel, self).__init__()
    self.vgg = vgg_layers(style_layers + content_layers)
    self.style_layers = style_layers
    self.content_layers = content_layers
    self.num_style_layers = len(style_layers)
    self.vgg.trainable = False

  def call(self, inputs):
    inputs = inputs*255.0
    preprocessed_input = tf.keras.applications.vgg19.preprocess_input(
        inputs)
    outputs = self.vgg(preprocessed_input)
    style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                      outputs[self.num_style_layers:])

    style_outputs = [gram_matrix(style_output)
                      for style_output in style_outputs]

    content_dict = {content_name: value
                    for content_name, value
                    in zip(self.content_layers, content_outputs)}

    style_dict = {style_name: value
                  for style_name, value
                  in zip(self.style_layers, style_outputs)}

    return {'content': content_dict, 'style': style_dict}
