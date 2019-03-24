import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from geopy.distance import great_circle
import math

from tensorflow.keras.layers import Dense, Input, Lambda, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam, Adadelta
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2

from siamese_loader import SiameseLoader

import numpy as np

def model_fn():
    input_shape = (800, 12, 1)
    left_input = Input(input_shape)
    right_input = Input(input_shape)

    convnet = Sequential()
    convnet.add(Conv2D(64,(100, 3), strides=(1,1), activation='relu', input_shape=input_shape, kernel_regularizer=l2(2e-4)))
    convnet.add(MaxPooling2D())

    convnet.add(Conv2D(128,(50, 2),activation='relu', kernel_regularizer=l2(2e-4)))
    convnet.add(MaxPooling2D())

    convnet.add(Conv2D(128,(20,1),activation='relu', kernel_regularizer=l2(2e-4)))
    convnet.add(MaxPooling2D())

    convnet.add(Conv2D(64,(4,1),activation='relu', kernel_regularizer=l2(2e-4)))
    convnet.add(Flatten())

    convnet.add(Dense(4096, activation="sigmoid",kernel_regularizer=l2(1e-3)))

    encoded_l = convnet(left_input)
    encoded_r = convnet(right_input)

    L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([encoded_l, encoded_r])
        
    prediction = Dense(1, activation='sigmoid')(L1_distance)
    siamese_net = Model(inputs=[left_input, right_input], outputs=prediction)

    siamese_net.compile(loss="binary_crossentropy", optimizer=Adam(0.00006))

    return siamese_net

def fn_predict(model, serie, prediction_set):
    
    # Pair the serie with each instance in the prdiction set
    pairs = [[serie.reshape(seq_len, features, 1)]*len(prediction_set["X"]), prediction_set["X"]]
    
    # Calculate the probabilities
    probs = model.predict(pairs)
    
    probs_label = list(zip(probs.flatten(), prediction_set["y"]))
    for l in np.unique(prediction_set["y"]):
        current_probs = [p[0] for p in probs_label if p[1] == l]
        
    # Get the max probabily and get the corresponding label
    return prediction_set["y"][np.argmax(probs.flatten())]

model = model_fn()
model.load_weights("model/siamese_net.h5")
model._make_predict_function()
prediction_set = np.load("model/prediction_set.npz")

def distance(pos1, pos2):
    from math import sin, cos, sqrt, atan2, radians

    R = 6373.0

    lat1 = radians(pos1["lat"])
    lon1 = radians(pos1["lng"])
    lat2 = radians(pos2["lat"])
    lon2 = radians(pos2["lng"])

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (sin(dlat/2))**2 + cos(lat1) * cos(lat2) * (sin(dlon/2))**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

def load_firestore():
    try:
        firebase_admin.get_app()
    except ValueError:
        cred = credentials.ApplicationDefault()
        firebase_admin.initialize_app(cred, {
            'projectId': 'practicantes-ucode',
        })
    return firestore.client()

def angle_from_coordinate(lat1, long1, lat2, long2):
    dLon = (long2 - long1)

    y = math.sin(dLon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dLon)

    brng = math.atan2(y, x)

    brng = math.degrees(brng)
    brng = (brng + 360) % 360
    brng = 360 - brng # count degrees clockwise - remove to make counter-clockwise

    return brng

def predict(data, context):
    """ Triggered by a change to a Firestore document.
    Args:
        data (dict): The event payload.
        context (google.cloud.functions.Context): Metadata for the event.
    """
    db = load_firestore()

    trigger_resource = context.resource

    print('Function triggered by change to: %s' % trigger_resource)

    current_player_id = data["value"]["fields"]["id"]["stringValue"]

    lat_lng_values = data["value"]["fields"]["pos"]["mapValue"]["fields"]
    
    current_pos = {}
    current_pos["lat"] = lat_lng_values["lat"]["doubleValue"]
    current_pos["lng"] = lat_lng_values["lng"]["doubleValue"]

    event = fn_predict(model, data["value"]["fields"]["sensorData"], prediction_set)
    name = data["value"]["fields"]["name"]["stringValue"]

    print("Event:", event)
    print("Name:", name)
    print("Pos:", current_pos)

    ref = db.collection("uploads")\
        .where("id", "<", current_player_id)

    ref1 = db.collection("uploads")\
            .where("id", ">", current_player_id)

    uploads = list(ref.get()) + list(ref1.get())

    players = dict()
    for u in uploads:
        upload = u.to_dict()
        if upload["id"] not in players:
            players[upload["id"]] = upload
            players[upload["id"]]["distance"] = great_circle((current_pos["lat"], current_pos["lng"]), (upload["pos"]["lat"], upload["pos"]["lng"])).meters
            players[upload["id"]]["degrees"] = angle_from_coordinate(current_pos["lat"], current_pos["lng"], upload["pos"]["lat"], upload["pos"]["lng"])
        elif players[upload["id"]]["time"] < upload["time"]:
            players[upload["id"]] = upload
            players[upload["id"]]["distance"] = great_circle((current_pos["lat"], current_pos["lng"]), (upload["pos"]["lat"], upload["pos"]["lng"])).meters
            players[upload["id"]]["degrees"] = angle_from_coordinate(current_pos["lat"], current_pos["lng"], upload["pos"]["lat"], upload["pos"]["lng"])

    print("Players:", players)

    for id, val in players.items():
        if val["distance"] < 5:
            db.collection("preds").document(id).set(dict(sentence=event, name=name))

