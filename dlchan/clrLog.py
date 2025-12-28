#Clear log file
#
#v0.1 nov 2024
#hdaniel@ualg.pt
#

from modules.flasklog import FlaskLog

FlaskLog.setup('data/dlchan.log')
FlaskLog.clearLogFile()