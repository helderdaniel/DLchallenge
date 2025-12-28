#Routes and views for ML/DL run challenge
#
#v0.1 jul 2022, v0.2 aug 2024, v0.3 nov 2024
#hdaniel@ualg.pt
#

from datetime import datetime, timedelta
from flask import Flask, Response, render_template, request, send_from_directory, send_file, jsonify
from werkzeug.utils import secure_filename
import os, time
from modules.evalhist import EvalHist
#from modules.modelsel import ModelSelect
from modelsel   import ModelSelect
from modules.runqueue import RunQueue
from modules.evalprog import EvaluationProgess  
from modules.score import ScoreTable

class Routes:

    accPrecision = 5
    
    @classmethod
    def setup(cls, app:Flask, modelSel:ModelSelect, runQueue:RunQueue, 
              evalProg:EvaluationProgess, scoreTable:ScoreTable, evalHist:EvalHist, 
              homePageFN:str, uploadFolder:str, maxContentLen:int, 
              evalDatasetName:str, trainDatasetFN:str, 
              challengeEnd:datetime)->None:
        
        @app.route('/')  # by default method is GET
        def home():
            #Clear data on load or reset
            return render_template(homePageFN, run='', rank='', msg='', date='')


        #Needed to trap max length error (set: maxContentLen)
        @app.errorhandler(413)
        def too_large(e):
            color = "darkred"
            msg   ='model size cannot exceed {:0.1f} MBytes'.format(maxContentLen/1024/1024)
            now = datetime.now().strftime('%d/%B/%Y, %H:%M:%S')
            return render_template(homePageFN, msg=msg, color=color, date=now)


        @app.route('/', methods=['POST'])
        def upload():
            
            if request.method == 'POST':   
                
                # Inner function helper to handle errors 
                def render(msg:str, color:str): 
                    now = datetime.now().strftime('%d/%B/%Y, %H:%M:%S')
                    return render_template(homePageFN, msg=msg, color=color, date=now)
                
                #check time out
                if challengeEnd < datetime.now():
                    return render('Session ended', 'red')

                #todo model is too large takes too much time
                #try with model: T2-KNearestNeighbors
                #tail -n100 /varlog/apache2/error.log
                #Check if file was uploaded
                if 'file' not in request.files:
                    return render('Request has only header, file missing', 'darkred')
                
                #Get file
                file = request.files['file']
                
                # no file selected
                if file.filename == '':
                    return render('No file selected', 'darkred')
                
                # Save file
                filename = secure_filename(file.filename)
                modelFN = os.path.join(uploadFolder, filename)
                file.save(modelFN)
                
                # Check if saved file is a valid model
                #if not model.valid(modelFN):
                model = modelSel.fromFile(modelFN)
                if model is None:
                    os.remove(modelFN)
                    #todo: cannot return model specific mesage if model is None
                    return render(modelSel.invalidMsg(), 'darkred')
                    #return render('Invalid model file', 'darkred')
                else:
                # Add model to run queue
                    runQueue.add(filename)
                    return render('Model uploaded', 'green')
                
                                
        @app.route('/howto')
        def howto():
            return render_template('howto.html')
        #or: app.add_url_rule('/howto', view_func=howto)

        @app.route('/help')
        def help():
            return render_template('help.html')

        @app.route('/traindataset', methods=['GET', 'POST'])
        def dntrainset():
            #or: return send_from_directory(<path_to_folder>, <filename>)
            return send_file(trainDatasetFN)

        @app.route('/_time')  # long pooling test
        def time2():
            time.sleep(5)
            t=datetime.now()
            t = t.strftime("%H:%M:%S")
            return jsonify(time=t, set=evalDatasetName)
        

        @app.route('/_timeleft')  # by default method is GET
        def timeleft():
            if challengeEnd < datetime.now():
                timeleft='00:00:00'
            else:
                td:timedelta = (challengeEnd-datetime.now())
                hours, remainder = divmod(int(td.total_seconds()), 3600)
                minutes, seconds = divmod(remainder, 60)
                timeleft='{:d}:{:02d}:{:02d}'.format(hours, minutes, seconds)
            return jsonify(time=timeleft, set=evalDatasetName)


        @app.route('/_waiters')
        def waiters():
            waiters = runQueue.waiting()
            top3waiters = waiters[:3]
            return jsonify(waiters=top3waiters)


        @app.route('/_ranking')
        def ranking():
            #Shared EvaluationProgress instance
            if evalProg.complete() and \
               evalProg.position() != '' and \
               evalProg.position() > 0:
                hl =  evalProg.position()  #highlight table position
            else:
                hl = -1  #out of table: do not highlight

            #set acc precision for rank table
            l = scoreTable.get()
            score = [ [e[0], "{0:.{1:}f}".format(e[1], Routes.accPrecision), e[2]] 
                    for e in l]
                    
            return jsonify(rank=score, highlight=hl)


        @app.route('/_running')
        def running():
            #Shared EvaluationProgress instance
            [topName, topHist, data] = scoreTable.top()

            #Format batch counter
            curBatch, batches = evalProg.batch()
            batchCounter = "{0:0{1:d}d}/{2:d}". \
            format(curBatch+1, len(str(batches)), batches)

            return jsonify(tag=evalProg.tag(), acc=evalProg.evalAcc(),
                           position=evalProg.position(), batches=batchCounter, 
                           topHist=topHist, topName=topName)

        #download evaluation history raw text file
        @app.route('/hist.csv')
        def history():
            try:
                return Response(evalHist.read(), mimetype='text/plain')
            except:
                return Response('No model was evaluated yet', mimetype='text/plain')
            