from os import listdir
from os.path import isdir
from PIL import Image
from numpy import savez_compressed
from numpy import asarray
from numpy import load
from numpy import expand_dims
from keras.models import load_model
import cv2
import numpy as np

# load face detection model
detector = cv2.dnn.readNetFromCaffe('config.prototxt', 'weight.caffemodel')

# Face detection using new model
def extract_face(filename, required_size=(160, 160)):
    # load image from file
    image = cv2.imread(filename)
    # preprocess image for face detection
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    # set the blob as input to the network
    detector.setInput(blob)
    # forward pass through the network to detect faces
    detections = detector.forward()
    # check if face was detected
    if detections.shape[2] > 0:
        # extract the bounding box from the first face
        box = detections[0, 0, 0, 3:7] * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
        x1, y1, x2, y2 = box.astype('int')
        # extract the face
        face = image[y1:y2, x1:x2]
        if face.size == 0:
            return None
        # resize pixels to the model size
        face = cv2.resize(face, required_size)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        return face
    else:
        return None


def load_faces(directory):
    faces = []
    # enumerate files
    for filename in listdir(directory):
        # path
        path = directory + filename
        # load image from file
        image = cv2.imread(path)
        # check if image was loaded
        if image is None:
            continue
        # get face
        face = extract_face(path)
        # check if face was detected
        if face is not None:
            # store
            faces.append(face)
    return faces



# load a dataset that contains one subdir for each class that in turn contains images
def load_dataset(directory):
    X, y = [], []
    # enumerate folders, on per class
    for subdir in listdir(directory):
        # path
        path = directory + subdir + '/'
        # skip any files that might be in the dir
        if not isdir(path):
            continue
        # load all faces in the subdirectory
        faces = load_faces(path)
        # create labels
        labels = [subdir for _ in range(len(faces))]
        # summarize progress
        print('>loaded %d examples for class: %s' % (len(faces), subdir))
        # store
        X.extend(faces)
        y.extend(labels)
    return asarray(X), asarray(y)

# load train dataset
trainX, trainy = load_dataset('/home/jawabreh/Desktop/masked-face/data/train/') 
print(trainX.shape, trainy.shape)
# save arrays to one file in compressed format
savez_compressed('/home/jawabreh/Desktop/masked-face/masked-detected-faces.npz', trainX, trainy)

# Face feature extraction using FaceNet
# get the face embedding for one face
def get_embedding(model, face_pixels):
	# scale pixel values
	face_pixels = face_pixels.astype('float32')
	# standardize pixel values across channels (global)
	mean, std = face_pixels.mean(), face_pixels.std()
	face_pixels = (face_pixels - mean) / std
	# transform face into one sample
	samples = expand_dims(face_pixels, axis=0)
	# make prediction to get embedding
	yhat = model.predict(samples)
	return yhat[0]

# load the face dataset
data = load('/home/jawabreh/Desktop/masked-face/masked-detected-faces.npz')
trainX, trainy  = data['arr_0'], data['arr_1'],
print('Loaded: ', trainX.shape, trainy.shape, )
# load the facenet model
model = load_model('/home/jawabreh/Desktop/masked-face/facenet_keras.h5')
print('Loaded Model')
# convert each face in the train set to an embedding
newTrainX = list()
for face_pixels in trainX:
	embedding = get_embedding(model, face_pixels)
	newTrainX.append(embedding)
newTrainX = asarray(newTrainX)
print(newTrainX.shape)
# save arrays to one file in compressed format
savez_compressed('/home/jawabreh/Desktop/masked-face/masked-embeddings.npz', newTrainX, trainy)
print("\n\n\tFace Feature Extraction Done Successfuly\n\n")