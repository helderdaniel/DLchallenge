#Entry point for for ML/DL run challenge web app in wsgi server
#
#v0.1 jul 2022
#hdaniel@ualg.pt
#

#To run from command line use wsgy.py file and:
#
#       python -m flask run
#  or
#       python -m flask --debug run
#  or
#
#       flask run -h localhost -p 5000

import sys

#This app path
sys.path.insert(0, "/home/hdaniel/public_html/dlchan")
#sys.path.insert(0, "/storage/OneDrive-ualg/00-ongoing/01-CTBU/01-WTurbines/04-FaultClassifiers/DLMLchallenge/site-v4")
sys.path.insert(0, "/home/hdaniel/public_html/dlchan/modules/deploy")

#Needed to find flask and other modules
#sys.path.insert(0, "/usr/lib/python3/dist-packages")
#sys.path.insert(0, "/home/hdaniel/.local/lib/python3.12/site-packages")
sys.path.insert(0, "/home/hdaniel/penv/lib/python3.12/site-packages")
#sys.path.insert(0, "/opt/penvml/lib/python3.12/site-packages")

from dlchan import app as application


