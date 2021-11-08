from keras.models import load_model
from keras.preprocessing.image import img_to_array
import cv2
import numpy as np
import PySimpleGUI as sg
import os.path

# load model
model = load_model('autoencoder_color.h5')

def colorize_image(path):
  grayscale = cv2.imread(path)
  h = grayscale.shape[0]
  w = grayscale.shape[1]
  SIZE=160
  #resizing image
  grayscale = cv2.cvtColor(grayscale, cv2.COLOR_BGR2RGB)
  grayscale = cv2.resize(grayscale, (SIZE, SIZE))
  grayscale = grayscale.astype('float32') / 255.0

  colorized = np.clip(model.predict(img_to_array(grayscale).reshape(1,SIZE, SIZE,3)),0.0,1.0).reshape(SIZE, SIZE,3)
  colorized = cv2.resize(colorized, (w,h))
  colorized = cv2.cvtColor(colorized, cv2.COLOR_RGB2BGR)
  colorized = np.clip(colorized, 0, 1)
  # the current colorized image is represented as a floating point data type in the range [0, 1] -- let's convert to an unsigned 8-bit integer representation in the range [0, 255]
  colorized = (255 * colorized).astype("uint8")
  return grayscale, colorized



left_col = [[sg.Text('Folder'), sg.In(size=(25,1), enable_events=True ,key='-FOLDER-'), sg.FolderBrowse()],
           [sg.Listbox(values=[], enable_events=True, size=(40,20),key='-FILE LIST-')]]

images_col = [[sg.Text('Input file:'), sg.In(enable_events=True, key='-IN FILE-'), sg.FileBrowse()],
             [sg.Button('Colorize Photo', key='-PHOTO-'), sg.Button('Save File', key='-SAVE-'), sg.Button('Exit')],
             [sg.Image(filename='', key='-IN-'), sg.Image(filename='', key='-OUT-')],]
	layout = [[sg.Column(left_col), sg.VSeperator(), sg.Column(images_col)]]

# ----- Make the window -----
window = sg.Window('Photo Colorizer', layout, grab_anywhere=True)

# ----- Run the Event Loop -----
prev_filename = colorized = cap = None
while True:
   event, values = window.read()
   if event in (None, 'Exit'):
       break
   if event == '-FOLDER-':         # Folder name was filled in, make a list of files in the folder
       folder = values['-FOLDER-']
       img_types = (".png", ".jpg", "jpeg", ".tiff", ".bmp")
       # get list of files in folder
       try:
           flist0 = os.listdir(folder)
       except:
           continue
       fnames = [f for f in flist0 if os.path.isfile(
           os.path.join(folder, f)) and f.lower().endswith(img_types)]
       window['-FILE LIST-'].update(fnames)
   elif event == '-FILE LIST-':    # A file was chosen from the listbox
       try:
           filename = os.path.join(values['-FOLDER-'], values['-FILE LIST-'][0])
           image = cv2.imread(filename)
           window['-IN-'].update(data=cv2.imencode('.png', image)[1].tobytes())
           window['-OUT-'].update(data='')
           window['-IN FILE-'].update('')


           image, colorized = colorize_image(filename)

           window['-OUT-'].update(data=cv2.imencode('.png', colorized)[1].tobytes())
       except:
           continue
   elif event == '-PHOTO-':        # Colorize photo button clicked
       try:
           if values['-IN FILE-']:
               filename = values['-IN FILE-']
           elif values['-FILE LIST-']:
               filename = os.path.join(values['-FOLDER-'], values['-FILE LIST-'][0])
           else:
               continue

           image, colorized = colorize_image(filename)
           window['-IN-'].update(data=cv2.imencode('.png', image)[1].tobytes())
           window['-OUT-'].update(data=cv2.imencode('.png', colorized)[1].tobytes())
       except:
           continue
   elif event == '-IN FILE-':      # A single filename was chosen
       filename = values['-IN FILE-']
       if filename != prev_filename:
           prev_filename = filename
           try:
               image = cv2.imread(filename)
               window['-IN-'].update(data=cv2.imencode('.png', image)[1].tobytes())
           except:
               continue
   elif event == '-SAVE-' and colorized is not None:   # Clicked the Save File button
       filename = sg.popup_get_file('Colorized image should be saved in (.png, .jpg, jpeg).\nEnter file name:', save_as=True)
       try:
           if filename:
               cv2.imwrite(filename, colorized)
               sg.popup_quick_message('Image save complete', background_color='green', text_color='white', font='Any 16')
       except:
           sg.popup_quick_message('ERROR - Image is NOT saved!', background_color='red', text_color='white', font='Any 16')
# ----- Exit program -----
window.close()
