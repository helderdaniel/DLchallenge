import os
#from distutils.log import debug 
from fileinput import filename 
from flask import *  
from werkzeug.utils import secure_filename

app = Flask(__name__)   
  
@app.route('/')   
def main():   
    return render_template("index.html")   
  
@app.route('/success', methods = ['POST'])   
def success():   
    if request.method == 'POST':   
        file = request.files['file'] 
        filename = secure_filename(file.filename)
        file.save(os.path.join('test', file.filename))  
        return render_template("ack.html", name = filename)   
  
if __name__ == '__main__':   
    app.run(debug=True)