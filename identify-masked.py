import cv2
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import Normalizer, LabelEncoder
from sklearn.svm import SVC
import matplotlib.pyplot as plt


# Load face detection model
proto_path = '/home/jawabreh/Desktop/masked-face/detector/config.prototxt'
model_path = '/home/jawabreh/Desktop/masked-face/detector/weight.caffemodel'
net = cv2.dnn.readNetFromCaffe(proto_path, model_path)


# Load face recognition model
facenet_model = load_model('/home/jawabreh/Desktop/masked-face/facenet_keras.h5')

# Load face embeddings
data = np.load('/home/jawabreh/Desktop/masked-face/masked-embeddings.npz')
trainX, trainy = data['arr_0'], data['arr_1']

# Normalize input vectors
in_encoder = Normalizer(norm='l2')
trainX = in_encoder.transform(trainX)

# Label encode targets
out_encoder = LabelEncoder()
out_encoder.fit(trainy)
trainy = out_encoder.transform(trainy)

# Define the classes
class_names = out_encoder.classes_
class_names = np.append(class_names, 'unknown')

# Train SVM classifier
model = SVC(kernel='linear', probability=True)
model.fit(trainX, trainy)

# Define function to extract face embeddings
def extract_face_embeddings(image):
    # Detect faces in the image
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    if detections.shape[2] == 0:
        return None
    # Extract the first face only
    box = detections[0, 0, 0, 3:7] * np.array([w, h, w, h])
    (x1, y1, x2, y2) = box.astype('int')
    face = image[y1:y2, x1:x2]
    # Resize face to the size required by facenet model
    face = cv2.resize(face, (160, 160))
    # Preprocess the face for facenet model
    face = face.astype('float32')
    mean, std = face.mean(), face.std()
    face = (face - mean) / std
    face = np.expand_dims(face, axis=0)
    # Generate embeddings using facenet model
    embeddings = facenet_model.predict(face)
    return embeddings[0]

# Define function to identify the identity of an input image
def identify_person(image):
    # Extract face embeddings from input image
    embeddings = extract_face_embeddings(image)
    if embeddings is None:
        return None, None
    # Normalize embeddings
    embeddings = in_encoder.transform([embeddings])
    # Predict the identity and confidence using SVM classifier
    prediction = model.predict(embeddings)
    confidence = model.predict_proba(embeddings)[0][prediction] * 100
    prediction = out_encoder.inverse_transform(prediction)
    return prediction[0].item(), confidence

# Define function to identify the identity and confidence of an input image
def identify_person_with_unknown(image, threshold=2):
    # Extract face embeddings from input image
    embeddings = extract_face_embeddings(image)
    if embeddings is None:
        return None, None
    # Normalize embeddings
    embeddings = in_encoder.transform([embeddings])
    # Predict the identity and confidence using SVM classifier
    predictions = model.predict_proba(embeddings)[0]
    max_idx = np.argmax(predictions)
    if predictions[max_idx] >= threshold:
        prediction = out_encoder.inverse_transform([max_idx])
        confidence = predictions[max_idx] * 100
        return prediction[0].item(), confidence
    else:
        return "unknown", None

# Example usage
image = cv2.imread('/home/jawabreh/Desktop/masked-face/data/test/Unknown/3.jpeg')

person, confidence = identify_person_with_unknown(image)
if person is None:
    print('No face detected in the input image!')
elif person == "unknown":
    text = "Unknown person"
    print(text)
else:
    print("SS")
    # Display the predicted name and confidence probability
    text = f'Predicted: {str(person)} ({confidence:.2f}%)'
    print(text)
