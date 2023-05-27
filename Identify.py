import cv2
import numpy as np
from mtcnn import MTCNN
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.preprocessing import Normalizer, LabelEncoder
from sklearn.svm import SVC

# Load masked face detection model
proto_path = '/home/jawabreh/Desktop/masked-face/detector/config.prototxt'
model_path = '/home/jawabreh/Desktop/masked-face/detector/weight.caffemodel'
net = cv2.dnn.readNetFromCaffe(proto_path, model_path)

# Load unmasked face detection model
detector = MTCNN()

# Load masked and unmasked face embeddings extraction model
facenet_model = load_model('/home/jawabreh/Desktop/masked-face/facenet_keras.h5')

# Load the classification model
classification_model = load_model("/home/jawabreh/Desktop/masked-face/classification_models/ResNet/ResNet50V2_model")

# Define the input image size and the confidence threshold
IMG_SIZE = (224, 224)
CONF_THRESH = 0.5

# Load the input image
image_path = "/home/jawabreh/Desktop/masked-face/data/test/Unknown/14.jpeg"
image = cv2.imread(image_path)

# Perform face detection
(h, w) = image.shape[:2]
# construct a blob from the image
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
# pass the blob through the network and obtain the face detections
net.setInput(blob)
detections = net.forward()

# Extract the first face only
if detections.shape[2] > 0:
    box = detections[0, 0, 0, 3:7] * np.array([w, h, w, h])
    (x1, y1, x2, y2) = box.astype('int')
    face = image[y1:y2, x1:x2]
    
    # Resize and preprocess the face for classification
    face = cv2.resize(face, IMG_SIZE)
    face = img_to_array(face)
    face = preprocess_input(face)
    
    # Perform face classification
    predictions = classification_model.predict(np.expand_dims(face, axis=0))[0]
    mask_prob = predictions[0]
    without_mask_prob = predictions[1]

    # Determine the predicted class
    if mask_prob > without_mask_prob and mask_prob > CONF_THRESH:
        pred_label = "Mask"
    elif without_mask_prob > mask_prob and without_mask_prob > CONF_THRESH:
        pred_label = "No Mask"
    else:
        pred_label = "Uncertain"
        
        
    # masked face recognition if the detected face is masked, or start unmasked face recognition if the face is unmasked 
    if pred_label == "Mask":
        # Perform face recognition on the masked face
        # Extract face embeddings from the masked face
        face = cv2.resize(face, (160, 160))
        face = face.astype('float32')
        mean, std = face.mean(), face.std()
        face = (face - mean) / std
        face = np.expand_dims(face, axis=0)
        embeddings = facenet_model.predict(face)
        
        # Normalize embeddings
        in_encoder = Normalizer(norm='l2')
        embeddings = in_encoder.transform(embeddings)
        
        # Load face embeddings for classification
        data = np.load('/home/jawabreh/Desktop/masked-face/masked-embeddings.npz')
        trainX, trainy = data['arr_0'], data['arr_1']
        
        # Train SVM classifier
        out_encoder = LabelEncoder()
        out_encoder.fit(trainy)
        trainy = out_encoder.transform(trainy)
        model = SVC(kernel='linear', probability=True)
        model.fit(trainX, trainy)
        
        # Predict the identity and confidence using SVM classifier
        predictions = model.predict_proba(embeddings)[0]
        max_idx = np.argmax(predictions)
        
        if predictions[max_idx] >= CONF_THRESH:
            prediction = out_encoder.inverse_transform([max_idx])
            prediction_name = prediction[0]
            confidence = predictions[max_idx] * 100
            print(f"Predicted Masked Face: {prediction_name} ({confidence:.2f}%)")
        else:
            print("Unknown Masked Face")
            
            
    elif pred_label == "No Mask":
        # Load face embeddings
        data = np.load('/home/jawabreh/Desktop/HumaneX/face-recognition/embeddings/unmasked-embeddings.npz')
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
            faces = detector.detect_faces(image)
            if not faces:
                return None
            # Extract the first face only
            x1, y1, width, height = faces[0]['box']
            x2, y2 = x1 + width, y1 + height
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
        def identify_person_with_unknown(image, threshold=0.9):
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

        person, confidence = identify_person_with_unknown(image)
        if person is None:
            print('No face detected in the input image!')
        elif person == "unknown":
            text = "Unknown Unmasked Face"
            print(text)
        else:
            # Display the predicted name and confidence probability
            text = f'Predicted Unmasked Face: {str(person)} ({confidence:.2f}%)'
            print(text)
