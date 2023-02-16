from flask import Flask, render_template, url_for, jsonify, request
from flask_cors import CORS
from ast import literal_eval
import pandas as pd
app = Flask(__name__)
CORS(app)

class Visualization:
    def __init__(self):
        self.dataset_name = None
        self.data = {"original": None, "filtered": None, "intersection": None, "intersection_filtered": None}
        return

    def set_dataset(self, dataset_name):
        """Set the current dataset used for the visualization"""
        self.dataset_name = dataset_name
        self.data["original"] = pd.read_csv("static/data/" + self.dataset_name + ".csv")
        self.data["original"]["points"] = self.data["original"]["points"].apply(lambda x : literal_eval(x))
        self.data["original"]["shape"] = self.data["original"]["shape"].apply(lambda x : literal_eval(x))
        if "points_coords" in self.data["original"].columns:
            self.data["original"]["points_coords"] = self.data["original"]["points_coords"].apply(lambda x : literal_eval(x))

        self.data["filtered"] = self.data["original"].copy()
        self.data["intersection"] = pd.read_csv("static/data/" + self.dataset_name + "_intersections.csv")
        self.data["intersection_filtered"] = self.data["intersection"].copy()
        print("Set dataset")

    def get_original_data(self):
        """Return the dataset with the original objects"""
        return self.data["original"], self.data["intersection"]

    def set_filtered_data(self, objects):
        """Filter the dataset based on selected objects"""
        bool_object = self.data["original"].object.isin(objects)
        self.data["filtered"] = self.data["original"][bool_object]
        bool_object = (self.data["intersection"].object1.isin(objects) & self.data["intersection"].object2.isin(objects))
        self.data["intersection_filtered"] = self.data["intersection"][bool_object]
        print("Set filtered dataset")
        
    def get_intersection_data(self, timestep):
        bool_timestep = self.data["intersection_filtered"].timestep == timestep
        return self.data["intersection_filtered"][bool_timestep]

    def get_scatter_data(self):
        print("Sent scatter dataset")
        return self.data["filtered"], self.data["intersection_filtered"]

vis = Visualization()
    
# SERVER CALLS
@app.route('/index')
def index():
    global vis
    vis = Visualization()
    url_for('static', filename='script.js')
    return render_template('index.html')

@app.route('/get_original_data')
def get_original_data():
    global vis
    df, df_intersection = vis.get_original_data()
    return jsonify({"objects": df.to_dict(orient = 'records'),
                "intersections": df_intersection.to_dict(orient = "records")})

@app.route("/set_dataset", methods=["POST"])
def set_dataset():
    if request.method == "POST":
        global vis
        vis.set_dataset(request.get_json()['dataset'])
    return '', 200

@app.route('/set_filtered_data', methods = ['POST'])
def set_filtered_data():
    if request.method == 'POST':
        global vis
        vis.set_filtered_data(request.get_json()['selected_objects'])
    return '', 200

@app.route('/get_intersection_data/<timestep>')
def get_intersection_data(timestep):
    timestep = int(timestep)
    global vis
    df = vis.get_intersection_data(timestep)
    return df.to_json(orient = "records")

@app.route('/get_scatter_data')
def get_scatter_data():
    '''Function that compute the method for objects'''
    global vis
    df, df_intersection = vis.get_scatter_data()
    return {"objects": df.to_dict(orient = "records"),
            "intersections": df_intersection.to_dict(orient = "records")}