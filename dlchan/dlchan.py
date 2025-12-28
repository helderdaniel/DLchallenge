#Flask app config for ML/DL run challenge web app
#
#v0.1 jul 2022, v0.2 aug 2024, v0.3 nov 2024
#hdaniel@ualg.pt
#

#https://code.visualstudio.com/docs/python/tutorial-flask

from flask import Flask
import os, secrets, sys
from filelock import FileLock
from threading import Thread
import tomllib
from modules.evalhist import EvalHist
from modules.flasklog import FlaskLog
from modules.evaluator import Evaluator
from modules.mdlRkEvKeras import ModelRkEvKeras
from modules.mdlRkEvSKL import ModelRkEvSKL
from modules.runqueue import RunQueue
from modules.evalprog import EvaluationProgess  
from modules.score import ScoreTable
#from modules.modelsel import ModelSelect
from modelsel   import ModelSelect
    

#App base folders are computed using current_app.root_path
#this allows changing root folder without changing code
#and supports absolute path when deployed in apache2 wsgi
##
#Config file however needs to be in root folder because current_app is only avaialble
#after app.run()
#
#HTML templates must be in a folder <root>/templates

#hard coded paths
#__ROOT_FOLDER        = os.getcwd()
#Needed to specify folder, since running in apache os.getcwd() returns: '/'
__ROOT_FOLDER        = '/home/hdaniel/public_html/dlchan/'
__TMP_FOLDER         = '/tmp'
__UPLOAD_FOLDER      = os.path.join(__ROOT_FOLDER, 'uploads')
__BEST_MODELS_FOLDER = os.path.join(__ROOT_FOLDER, 'uploads/best')
__DATA_FOLDER        = os.path.join(__ROOT_FOLDER, 'data')
__CONFIG_FN          = os.path.join(__DATA_FOLDER, 'dlchan.cfg') 
__LOG_FN             = os.path.join(__DATA_FOLDER, 'dlchan.log') 
__HOME_PAGE_FN       = 'home.html'
__SCORE_TABLE_FN     = os.path.join(__DATA_FOLDER, 'scoretable.pickle') 
__RUN_QUEUE_FN       = os.path.join(__DATA_FOLDER, 'runqueue.pickle')
__EVAL_HIST_FN       = os.path.join(__DATA_FOLDER, 'evalhist.csv')
__SCORE_LOCK_FN      = os.path.join(__TMP_FOLDER, 'dlscore.lock')
__QUEUE_LOCK_FN      = os.path.join(__TMP_FOLDER, 'dlqueue.lock')
__HIST_LOCK_FN       = os.path.join(__TMP_FOLDER, 'dlhist.lock')


#Import config from file
with open(__CONFIG_FN, 'rb') as f:
    cfgData = tomllib.load(f)

__EVAL_DATASET     = cfgData['evalDatasetName']
__TRAIN_DATASET_FN = os.path.join(__DATA_FOLDER, cfgData['trainDatasetFile'])
__EVAL_DATASET_FN  = os.path.join(__DATA_FOLDER, cfgData['evalDatasetFile'])
__CLASSES_DATASET  = cfgData['noClasses']
__CHANNELS_DATASET = cfgData['noChannels']
__MAPS_DATASET     = cfgData['noMaps']
__MAX_MODEL_SIZE   = cfgData['maxModelSize']
__CHALLENGE_END    = cfgData['endDate']
__EVAL_PERIOD      = cfgData['evalPeriod']  #period to check RUNQUEUE in seconds
__EVAL_SHUFFLE     = cfgData['shuffle']     #Shuffle dataset before evaluation
__EVAL_SEED        = cfgData['seed']        #Seed for shuffle dataset before evaluation
if __EVAL_SEED == '':
    __EVAL_SEED = None                       #None shuffle differently every time
                                             #int  shuffle the same way every time

#Setup logging
#disable message on develop server: http://127.0.0.1:5000/
#Setup level=WARN to omit GET messages in log
FlaskLog.setup(__LOG_FN, FlaskLog.WARN)  
FlaskLog.warning('DLChan web app started')

#Load Model Chooser
supportedModels = [ ModelRkEvKeras(), ModelRkEvSKL() ]
modelSel:ModelSelect = ModelSelect(supportedModels)

#Engine settings

#need to set multiThread = True => thread_local=False
#to work between threads: EvaluatorThread and This thread
#to animate evaluation chart
#https://py-filelock.readthedocs.io/en/latest/index.html
multiThread = True     
queueLock   = FileLock(__QUEUE_LOCK_FN, thread_local=not multiThread)
runQueue    = RunQueue(__RUN_QUEUE_FN, queueLock)
runQueue.clear()

scoreLock   = FileLock  (__SCORE_LOCK_FN, thread_local=not multiThread)
scoreTable  = ScoreTable(__SCORE_TABLE_FN, scoreLock)

histLock    = FileLock  (__HIST_LOCK_FN, thread_local=not multiThread)
evalHistory = EvalHist  (__EVAL_HIST_FN, histLock)

#Shared evaluation progress data to pass info from evaluator thread to routes
evalProg     = EvaluationProgess()  #tag, acc, progress(batch), score position, batches, blink


#Evaluator
eval = Evaluator(modelSel, runQueue, evalProg, scoreTable, scoreLock, evalHistory,
                            __UPLOAD_FOLDER, __BEST_MODELS_FOLDER, __EVAL_DATASET_FN, 
                            __CLASSES_DATASET, __CHANNELS_DATASET, __MAPS_DATASET,
                            __EVAL_SHUFFLE, __EVAL_SEED)

evalThread = Thread(target=eval.evaluatorThread, daemon=True, args=[__EVAL_PERIOD])
evalThread.start()


#Define app and routes
app:Flask = Flask(__name__)
app.config['SECRET_KEY'] = secrets.token_hex(20)
#app.config['APPLICATION_ROOT'] = '/dlchan'


from views import Routes
Routes.setup(app, modelSel, runQueue, evalProg, scoreTable, evalHistory,
             __HOME_PAGE_FN, __UPLOAD_FOLDER, __MAX_MODEL_SIZE,
             __EVAL_DATASET, __TRAIN_DATASET_FN, __CHALLENGE_END)

#Note: No need to app.run() because launch.json is running the flask app from comand line
#      Anyway, it is better this way to avoid conflicts when running with apache2 wsgi

