import numpy as np
import cv2
import PySimpleGUI as sg
import os.path

# pre-trained model
prototxt_path = r'model/colorization_deploy_v2.prototxt'
model_path = r'model/colorization_release_v2.caffemodel'
kernel_path = r'model/pts_in_hull.npy'

net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
points = np.load(kernel_path)

points = points.transpose().reshape(2, 313, 1, 1)
net.getLayer(net.getLayerId("class8_ab")).blobs = [points.astype("float32")]
net.getLayer(net.getLayerId("conv8_313_rh")).blobs = [np.full([1,313],2.606, dtype="float32")]

def colorize_image(image_filename=None):
    image = cv2.imread(image_filename)
    image_scaled = image.astype("float32") / 255.0
    image_lab = cv2.cvtColor(image_scaled, cv2.COLOR_BGR2LAB)

    image_resized = cv2.resize(image_lab, (224, 224))
    L = cv2.split(image_resized)[0]
    L -= 50

    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab = cv2.resize(ab, (image.shape[1], image.shape[0]))
    L = cv2.split(image_lab)[0]
    image_colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
    image_colorized = cv2.cvtColor(image_colorized, cv2.COLOR_LAB2BGR)
    image_colorized = np.clip(image_colorized, 0, 1)
    image_colorized = (255 * image_colorized).astype("uint8")
    return image, image_colorized

# The GUI
left_col = [[sg.Text('Folder'), sg.In(size=(25,1), enable_events=True ,key='-FOLDER-'), sg.FolderBrowse()],
            [sg.Listbox(values=[], enable_events=True, size=(40,20),key='-FILE LIST-')]]

images_col = [[sg.Text('Input file:'), sg.In(enable_events=True, key='-IN FILE-'), sg.FileBrowse()],
              [sg.Button('Colorize Photo', key='-PHOTO-'), sg.Button('Save File', key='-SAVE-'), sg.Button('Exit')],
              [sg.Image(filename='', key='-IN-'), sg.Image(filename='', key='-OUT-')],]

layout = [[sg.Column(left_col), sg.VSeperator(), sg.Column(images_col)]]

window = sg.Window('Photo Colorizer', layout)

prev_filename = image_colorized = cap = None
while True:
    event, values = window.read()
    if event in (None, 'Exit'):
        break
    if event == '-FOLDER-':
        folder = values['-FOLDER-']
        img_types = (".png", ".jpg", "jpeg")
        try:
            file_list = os.listdir(folder)
        except:
            continue

        fnames = []
        for f in file_list:
            if os.path.isfile(os.path.join(folder, f)) and f.lower().endswith(img_types):
                fnames.append(f)
        window['-FILE LIST-'].update(fnames)
    elif event == '-FILE LIST-':
        try:
            filename = os.path.join(values['-FOLDER-'], values['-FILE LIST-'][0])
            image = cv2.imread(filename)
            window['-IN-'].update(data=cv2.imencode('.png', image)[1].tobytes())
            window['-OUT-'].update(data='')
            window['-IN FILE-'].update('')
            image, image_colorized = colorize_image(filename)
            window['-OUT-'].update(data=cv2.imencode('.png', image_colorized)[1].tobytes())
        except:
            continue
    elif event == '-PHOTO-':
        try:
            if values['-IN FILE-']:
                filename = values['-IN FILE-']
            elif values['-FILE LIST-']:
                filename = os.path.join(values['-FOLDER-'], values['-FILE LIST-'][0])
            else:
                continue
            image, image_colorized = colorize_image(filename)
            window['-IN-'].update(data=cv2.imencode('.png', image)[1].tobytes())
            window['-OUT-'].update(data=cv2.imencode('.png', image_colorized)[1].tobytes())
        except:
            continue
    elif event == '-IN FILE-':
        filename = values['-IN FILE-']
        if filename != prev_filename:
            prev_filename = filename
            try:
                image = cv2.imread(filename)
                window['-IN-'].update(data=cv2.imencode('.png', image)[1].tobytes())
            except:
                continue
    elif event == '-SAVE-' and image_colorized is not None:
        filename = sg.popup_get_file('Colorized image should be saved in (.png, .jpg, jpeg).\nEnter file name:', save_as=True)
        try:
            if filename:
                cv2.imwrite(filename, image_colorized)
                sg.popup_quick_message('Image is saved complete', background_color='green', text_color='white', font='Any 16')
        except:
            sg.popup_quick_message('Image is NOT saved!', background_color='red', text_color='white', font='Any 16')

window.close()
