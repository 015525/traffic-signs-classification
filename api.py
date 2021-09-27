from re import I
from flask import Flask, jsonify, request, render_template, redirect, flash
from signs_classification_vsc import signs_classification
from werkzeug.utils import secure_filename
import os
import matplotlib.pyplot as plt
from PIL import Image
from skimage.transform import rescale

app = Flask(__name__)
sc=  signs_classification()

UPLOAD_FOLDER = 'D:/images_from_websites'
 
app.secret_key = "signs_classify"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

'''
for file in (os.listdir('C:/Users/es-abdoahmed022/gui_for_signs/static')) :
     if file.endswith('.png') or file.endswith('.jpg') :
        os.remove(file) 

'''

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/data_samples")
def data_samples():
    sc.get_data()
    return render_template("data_samples.html", text = 'success', color = 'success')

@app.route("/data_preprocess")
def data_preprocesse():
    sc.data_preprocess()
    return render_template("data_preprocess.html", text = 'success', color = 'success')


@app.route("/train")
def train():
    acc, val_acc  = sc.train()
    return render_template("result.html", text1 = round(acc*100, 3), text2 = round(val_acc*100, 3), color = 'success')


@app.route("/train_plots")
def train_plots():
    sc.accs_plots()
    return render_template("train_plot.html", color = 'success')

@app.route("/conv_mat")
def conv_mat():
    sc.accs_plots()
    return render_template("conv_mat.html", color = 'success')

@app.route("/predict1")
def predict1():
    return render_template("predict.html")

@app.route("/predict2", methods = ['GET', 'post'])
def predict2():
    res = "no file"
    if 'file' not in request.files:
        flash('No file part')
        return render_template("result.html", text = res, color = 'success')
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)

        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        img = Image.open(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        img = img.resize((500,500))
        img.save('C:/Users/es-abdoahmed022/gui_for_signs/static/new_image.png')
        #print('upload_image filename: ' + filename)
        flash('Image successfully uploaded and displayed below')
        x = plt.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        #y = x.shape
        res = sc.predict(x)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')

    return render_template("predict_res.html", text = res, img_path = filename, color = 'success')

    '''
    if request.method == "POST" :
        f = request.files['file']
        f.save(secure_filename(f.filename))
        res = sc.predict(f)
        return render_template("result.html", text = res, color = 'success')
    '''

app.run(debug = True)
'''
r = requests.get("http://127.0.0.1:5000/predict2")
    img  = r.content
    res = sc.predict(img)
    return render_template("result.html", text = res, color = 'success')
'''
