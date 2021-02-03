# SISTEMAS INTELIGENTES
# Ingeniería Informática - 4º Año
# Proyecto de Transferencia de Estilos
#   Jorge Acevedo de León - alu0101123622
#   Rafael Cala González - alu0101121901
#   David Valverde Gómez - alu0101100296
#
# Fichero: DaVincAI.py
#   Fichero principal del proyecto, que deberá ser ejecutado desde
#   terminal. Contiene la interfaz de usuario establecida a modo
#   de diferentes menús mostrados en terminal, y se encarga de
#   establecer las rutas a las imágenes y de indicar el modo de
#   ejecución

import tensorflow as tf
import time
import os
import sys
import platform
import six

os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# Diferentes SSOO usan diferentes slashes para las rutas
if platform.system() == "Windows":
  slash = "\\"
else:
  slash = "/"

# Función para limpiar la consola en diferentes SSOO
def limpiar_consola():
  if platform.system() == "Linux":
    os.system(f"clear")
  elif platform.system() == "Windows":
    os.system(f"cls")


# Menú para las opciones de toma de una imagen de contenido
def content_image_menu():
  print("---- Content Image ----")
  print("1. Take from the Internet")
  print("2. Take from Local")
  choice = input("Your choice: ")
  limpiar_consola()

  if choice == "1":
      content_name = "content_image"
      content_link = input("Link to the Content Image: ")
      content_path = tf.keras.utils.get_file(content_name, content_link)
      limpiar_consola()

  elif choice == "2":
      content_path = input("Path to Content Image: ")
      limpiar_consola()

  else:
      print("Option Not Recognized")
      return

  return content_path

# Menú para las opciones de toma de una imagen de estilo
def style_image_menu():
  print("---- Style Image ----")
  print("1. Take from the Internet")
  print("2. Take from Local")
  print("3. Take a Predefined Style")
  print("4. Use a chain of different Styles")
  choice = input("Your choice: ")
  limpiar_consola()

  if choice == "1":
    style_name = "style_image"
    style_link = input("Link to the Style Image: ")
    style_path = tf.keras.utils.get_file(style_name, style_link)
    limpiar_consola()

  elif choice == "2":
    style_path = input("Path to the Style Image: ")
    limpiar_consola()

  elif choice == "3":
    print("---- Choose a Style ----")
    print("1. Abstract Style")
    print("2. El Grito Style")
    print("3. Gold Style")
    print("4. Guernica Style")
    print("5. Mona Lisa Style")
    print("6. Psychedelic Style")
    print("7. Puntillism Style")
    print("8. Waves Style")
    print("9. Yellow Thunder Style")

    style_choice = input("Your choice: ")
    limpiar_consola()

    if style_choice == "1":
      style_path = "estilo" + slash + "abstract.jpg"

    elif style_choice == "2":
      style_path = "estilo" + slash + "el_grito.jpg"

    elif style_choice == "3":
      style_path = "estilo" + slash + "gold_pattern.jpg"

    elif style_choice == "4":
      style_path = "estilo" + slash + "guernica-picasso.jpg"

    elif style_choice == "5":
      style_path = "estilo" + slash + "mona_lisa.jpg"

    elif style_choice == "6":
      style_path = "estilo" + slash + "psychedelic_landscape.jpg"

    elif style_choice == "7":
      style_path = "estilo" + slash + "puntillismo.jpg"

    elif style_choice == "8":
      style_path = "estilo" + slash + "waves.png"

    elif style_choice == "9":
      style_path = "estilo" + slash + "yellow_thunder.jpg"

    else:
      print("Option Not Recognized")
      return

  elif choice == "4":
    style_path = ""
    limpiar_consola()

  else:
    print("Option Not Recognized")
    return

  return style_path

# Menú para indicar si se desea realizar un collage
def collage_menu():
  print("---- Collage ----")
  collage = input("Generate a Collage with the Result [y/n]: ")
  limpiar_consola()
  return collage

# Menú para especificar la ruta donde guardar el resultado
def result_image_menu():
  print("---- Result Image ----")
  result_path = input("Path to save the result: ")
  limpiar_consola()
  return result_path

# Función que realiza la transferencia de estilos para todas las imágenes de estilos
# guardadas en la carpeta 'estilo'
def style_chain(content_path, collage):
  content_name = input("Name of the Content Image (without extension): ")
  # Directorio con las imagenes de estilo
  style_dir = "estilo"
  style_list = os.listdir(style_dir)
  new_dir = "resultados" + slash + "chain_" + content_name

  # Si no existe el directorio especificado se crea uno nuevo
  if not os.path.exists(new_dir):
    os.makedirs(new_dir)
  # Contadores para el número de estilos aplicados y el tiempo de ejecución
  style_number = len(style_list) - 1
  styles_applied = 0
  start = time.time()
  chain = "1"

  # Para cada estilo se crea una nueva imagen dentro del directorio especificado
  # Y se aplica la transferencia de estilos
  for style in style_list:
    style_path = "estilo\\" + str(style)
    result_path = new_dir + slash + content_name + "_" + \
        str(style).replace('.jpg', '') + "_style" + ".jpg"
    os.system(
        f"python style-transfer.py {content_path} {style_path} {result_path} {collage} {chain}")    
    current = time.time()
    styles_applied += 1
    print("Tiempo total actual: {:.1f}".format(current-start) + "s")
    print(f"Progeso total: {styles_applied}/{style_number}")

  end = time.time()
  print("Tiempo total de ejecucion: {:.1f}".format(
      (end-start)/60) + " minutos.")


# Menú principal
def main_menu():
  limpiar_consola()
  print("---- Image Style Transfer ----")
  content_path = content_image_menu()
  style_path = style_image_menu()
  collage = collage_menu()

  if style_path == "":
    style_chain(content_path, collage)

  else:
    result_path = result_image_menu()
    chain = "0"
    os.system(
        f"python style-transfer.py {content_path} {style_path} {result_path} {collage} {chain}")



main_menu()