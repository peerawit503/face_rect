#!/usr/bin/python3
from flask import Flask, request, jsonify , Response
from flask_restful import Resource, Api
from flask_cors import CORS
import cv2
import math
from sklearn import neighbors
import os
import os.path
import pickle

import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
import sys
import json
import time
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
import shutil
app = Flask(__name__)
api = Api(app)
CORS(app)
def train(train_dir, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):


    X = []
    y = []
    
    # Loop through each person in the training set
    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue
        
        # Loop through each training image for the current person
        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
            image = face_recognition.load_image_file(img_path)
            face_bounding_boxes = face_recognition.face_locations(image)
            
            if len(face_bounding_boxes) != 1:
                # If there are no people (or too many people) in a training image, skip the image.
                if verbose:
                    print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(face_bounding_boxes) < 1 else "Found more than one face"))
            else:
                # Add face encoding for current image to the training set
                X.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                y.append(class_dir)

# Determine how many neighbors to use for weighting in the KNN classifier
    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))
        if verbose:
            print("Chose n_neighbors automatically:", n_neighbors)

        # Create and train the KNN classifier
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(X, y)

    # Save the trained KNN classifier
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)
    
    return knn_clf


class Training(Resource):
    def post(self, name):
        cap = cv2.VideoCapture(0)
        name = name
        face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        count = 0
        directory = "knn_examples/train/"+str(name)
        if not os.path.exists(directory):
            os.makedirs(directory)
        count2 = int(len([name for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))]))
        while(True):
            ret, img = cap.read()
            faces = face_detector.detectMultiScale(img, 1.3, 5)
            for (x,y,w,h) in faces:
                count += 1
                count2 += 1
              
                cv2.imwrite("knn_examples/train/" + name +"/" + str(name)  + str(count2) + ".jpg", img[y-30:y+h+50,x-30:x+w+50])
                cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
            k = cv2.waitKey(100) & 0xff
            if k == 27:
                break
            elif count >= 20:
                break
        cap.release()
#        classifier = train("knn_examples/train", model_save_path="trained_knn_model.clf", n_neighbors=2)
        return Response("Register success", mimetype='text/html', status=200 )


class Delete(Resource):
    def post(self, name):
        dirpath = "knn_examples/train/"+str(name)
        if os.path.exists(dirpath) and os.path.isdir(dirpath):
            shutil.rmtree(dirpath)
        time.sleep(10)
        
        return Response("Training Success", mimetype='text/html', status=200 )

class Edit(Resource):
    def get(self, name, new):
#        basedir = "knn_examples/train/"+str(name)
#        newdir = "knn_examples/train/"+str(new)
#        if os.path.exists(basedir) and os.path.isdir(basedir):
#            os.rename(basedir , newdir )

       
        return Response("Training Success", mimetype='text/html', status=200 )

class TrainData(Resource):
    def post(self):
        print("Training KNN classifier...")
        classifier = train("knn_examples/train", model_save_path="trained_knn_model.clf", n_neighbors=2)
        print("Training complete!")
        return Response("Training Success", mimetype='text/html', status=200 )

class File(Resource):
    def get(self , name):
        mypath = "knn_examples/train/" + name
        f = []
        for (dirpath, dirnames, filenames) in os.walk(mypath):
            f.extend(filenames)
        return jsonify(f)

api.add_resource(Training, '/add/<name>')
api.add_resource(Delete, '/delete/<name>')
api.add_resource(Edit, '/edit/<name>/<new>')
api.add_resource(TrainData, '/train')
api.add_resource(File, '/file/<name>')
if __name__ == '__main__':
     app.run()
