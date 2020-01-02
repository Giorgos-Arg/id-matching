import sys
from os import path, listdir, environ
from numpy import asarray, array, expand_dims
from matplotlib import pyplot
from PIL import Image
from scipy.spatial.distance import cosine
from operator import itemgetter
from mtcnn.mtcnn import MTCNN
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input

# Disable the AVX2 Tensorflow warning
environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# extract one face from an image
def extract_face(filepath):
    # open image from file
    img = Image.open(filepath)
    # convert to rgb
    rgb_img = img.convert('RGB')
    pixel_array = asarray(rgb_img)
    # create the face detector
    detector = MTCNN()
    # detect faces in the image
    results = detector.detect_faces(pixel_array)
    # define the  boundaries of the first face
    x, y, width, height = results[0]['box']
    # extract the face
    face_array = pixel_array[y:y+height, x:x + width]
    # resize face_array to the model size
    img = Image.fromarray(face_array)
    img = img.resize((224, 224))
    face_array = asarray(img)
    return face_array


# calculate face features
def get_face_features(model, face):
    # prepare the face for the model
    face = face.astype('float32')
    face = expand_dims(face, axis=0)
    face = preprocess_input(face, version=2)
    # perform prediction
    features = model.predict(face)
    return features


# verify if a candidate face matches the known face and return the similarity score
def verify_identity(known_face_features, candidate_face_features):
    # threshold for two faces to be considered as matching
    threshold = 0.5
    # calculate distance between face features i.e. their similarity
    score = cosine(known_face_features, candidate_face_features)
    if score <= threshold:
        return score
    else:
        return -1


def main(argv):
    if len(sys.argv) != 3:
        print('py matchid.py <input_image> <documents_folder>')
        sys.exit(0)
    else:
        input_img = sys.argv[1]
        documents_folder = sys.argv[2]

    documents = [path.join(documents_folder, x) for x in listdir(documents_folder) if '.png' in x or '.jpg' in x]
    # list for storing all the documents that match the input image
    matched_list = []

    # create a vggface model
    model = VGGFace(include_top=False, model='resnet50', input_shape=(224, 224, 3), pooling='avg')
    # extract input face
    input_face_array = extract_face(input_img)
    # get input face features
    input_face_features = get_face_features(model, input_face_array)

    for i in range(len(documents)-1):
        # extract document face
        document_face_array = extract_face(documents[i])
        # get document face features
        document_face_features = get_face_features(model, document_face_array)
        # verify identity
        score = verify_identity(input_face_features, document_face_features)

        # if document matched the input image then store the document index and its similarity score
        if(score != -1):
            matched_list.append((i, score))

    if len(matched_list) != 0:
        # find the document with the highest similarity score
        max_i = max(matched_list, key=itemgetter(1))[0]
        print('\n', input_img, ' matched ',  documents[max_i], ' in ', documents_folder, '\n')
    else:
        print('\n', input_img, ' not matched in ', documents_folder, '\n')


if __name__ == "__main__":
    main(sys.argv)
    