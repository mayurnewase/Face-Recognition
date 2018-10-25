import os
import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model
from flask import Flask, render_template, request
from flask_dropzone import Dropzone
import matplotlib.pyplot as plt
import pickle
from werkzeug import secure_filename

basedir = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__)

app.config.update(
    UPLOADED_PATH=os.path.join(basedir, 'uploads'),
    IMAGE_PATH="/home/mayur/AI Explorer/Face Recognition System Project/att_faces/",
    FILE_NAME=None,
    DROPZONE_ALLOWED_FILE_TYPE='image',
    DROPZONE_MAX_FILE_SIZE=3,
    DROPZONE_MAX_FILES=30,
    DROPZONE_IN_FORM=True,
    DROPZONE_UPLOAD_ON_CLICK=True,
    DROPZONE_UPLOAD_ACTION='handle_upload',  # URL or endpoint
    DROPZONE_UPLOAD_BTN_ID='submit',
    ALLOWED=None
)
app.config['DROPZONE_ALLOWED_FILE_CUSTOM'] = True
app.config['DROPZONE_ALLOWED_FILE_TYPE'] = 'image/*, .pdf, .txt , .pgm'

dropzone = Dropzone(app)

@app.route('/')
def index():
    return render_template('first.html')

@app.route('/upload', methods=['POST'])
def handle_upload():
    for key, f in request.files.items():
        if key.startswith('file'):
            f.save(os.path.join(app.config['UPLOADED_PATH'], f.filename))
            print("here name is ",f.filename)
            app.config["FILE_NAME"] = f.filename
            print("here name is ",app.config["FILE_NAME"])

    return '', 204

def findPerson(pred , embed_orig):

    dist = np.sum(np.square(np.subtract(pred , embed_orig)))
    if dist < 0.5:
        print("Same Person")
        return "same person" , dist
    else:
        print("Not Same")
        return "Not same person" , dist

@app.route('/test', methods=['POST'])
def test_model():
    print("TEST FUNCTION CALLED-------------------------")
	
    embed = pickle.load(open("embeddings.pkl" , mode = "rb"))
    #print(embed)
	
    title = request.form.get('title1')
    f = request.files["file1"]
    f.save(secure_filename(f.filename))
    print("file is ",f)
    recieved_img = cv2.imread(f.filename , cv2.IMREAD_GRAYSCALE) / 255
    print("saved img is",recieved_img.shape)
	
    img = np.expand_dims(recieved_img , axis = 2)
    img = np.expand_dims(img , axis = 0)
	
    with tf.Session() as sess:
        model = load_model("Base_siamese_network_v3.h5")
        pred = model.predict(img)
	
    embed_orig = embed.get(title)
    result , dist = findPerson(pred , embed_orig)
    return 'result: %s <br> dist: %s' %(result,dist)
	
    """
    upath = str(app.config["IMAGE_PATH"])+title+"/"+str(app.config["FILE_NAME"])
    print("final path is " , upath)
    recieved_img = cv2.imread(upath , cv2.IMREAD_GRAYSCALE)
    print("shape is ",recieved_img.shape)
    u_img = np.expand_dims(recieved_img , 0)
    u_img = np.expand_dims(u_img , 3)
    print(u_img.shape)
    
    bpath = str(app.config["IMAGE_PATH"])+title+"/"+"1.pgm"
    print("final path is " , bpath)
    base_img = cv2.imread(bpath , cv2.IMREAD_GRAYSCALE)
    print("shape is ",base_img.shape)
    b_img = np.expand_dims(base_img , 0)
    b_img = np.expand_dims(b_img , 3)
    print(b_img.shape)

    with tf.Session() as sess:
        model = load_model("siamese_network_v2.h5")
        preds_user = np.squeeze(model.predict(u_img))
        preds_base = np.squeeze(model.predict(b_img))

    dist = np.sum(np.square(np.subtract(preds_base , preds_user)))
    if dist < 0.5:
        result = "Same Person" + str(dist)
    else:
        result = "Not Same" + str(dist)
    print(dist)

    return 'result: %s <br> dist: %s' %(result,dist)
"""
	
@app.route('/train', methods=['POST'])
def train_model():
    print("TRAIN FUNCTION CALLED-------------------------")

    embed = pickle.load(open("embeddings.pkl" , mode = "rb"))
    print("embediing have " , len(embed))
    title = request.form.get('title2')
    f = request.files["file2"]
    f.save(secure_filename(f.filename))
    print("file is ",f)
    recieved_img = cv2.imread(f.filename , cv2.IMREAD_GRAYSCALE) / 255
    print("saved img is",recieved_img.shape)
	
    img = np.expand_dims(recieved_img , axis = 2)
    img = np.expand_dims(img , axis = 0)
	
    with tf.Session() as sess:
        model = load_model("Base_siamese_network_v3.h5")
        pred = model.predict(img)
	
    embed.update({title : pred})
    print("embediing have " , len(embed))
    pickle.dump(embed , open("embeddings.pkl" , mode = "wb"))
    return "model trained for %s , and can predict correctly in future "%(title)
	
	
    """
	title = request.form.get('title')
    upath = str(app.config["IMAGE_PATH"])+title+"/"+str(app.config["FILE_NAME"])
    print("final path is " , upath)
    recieved_img = cv2.imread(upath , cv2.IMREAD_GRAYSCALE)
    print("shape is ",recieved_img.shape)
    u_img = np.expand_dims(recieved_img , 0)
    u_img = np.expand_dims(u_img , 3)
    print(u_img.shape)
    
    bpath = str(app.config["IMAGE_PATH"])+title+"/"+"1.pgm"
    print("final path is " , bpath)
    base_img = cv2.imread(bpath , cv2.IMREAD_GRAYSCALE)
    print("shape is ",base_img.shape)
    b_img = np.expand_dims(base_img , 0)
    b_img = np.expand_dims(b_img , 3)
    print(b_img.shape)

    with tf.Session() as sess:
        model = load_model("siamese_network_v2.h5")
        preds_user = np.squeeze(model.predict(u_img))
        preds_base = np.squeeze(model.predict(b_img))

    dist = np.sum(np.square(np.subtract(preds_base , preds_user)))
    if dist < 0.5:
        result = "Same Person" + str(dist)
    else:
        result = "Not Same" + str(dist)
    print(dist)
    """
    return 'result: %s <br> dist: %s' %(result,dist)	
	


if __name__ == '__main__':
    app.run(debug=True)





























