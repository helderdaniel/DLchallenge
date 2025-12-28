#Flask log wrapper for ML/DL run challenge web app
#
#v0.1 nov 2024
#hdaniel@ualg.pt
#

import logging
from enum import Enum


class FlaskLog:

    INFO  = logging.INFO
    WARN  = logging.WARNING
    ERROR = logging.ERROR

    _logger = logging.getLogger("werkzeug")

    @classmethod
    def clearLogFile(cls) -> None:
        ''' open for writting clear the file'''
        try: 
            with open(cls._logFN, 'w'):
                    pass
        except: 
            pass

    @classmethod
    def setup(cls, logFN:str, level='WARN') -> None:      
        cls._logFN = logFN
        logging.basicConfig(filename=logFN, encoding='utf-8', 
                            format='%(asctime)s %(levelname)s: %(message)s')
        # cannot use FlaskLog constants inside class as default parameters
        # so, do it this way:
        level_value = getattr(cls, level) if isinstance(level, str) else level
        cls.setLevel(level_value)                            

    @classmethod
    def setLevel(cls, level) -> None:
        cls._logger.setLevel(level)

    @classmethod
    def info(cls, msg:str) -> None:
        cls._logger.info(msg)

    @classmethod
    def warning(cls, msg:str) -> None:
        cls._logger.warning(msg)

    @classmethod
    def error(cls, msg:str) -> None:
        cls._logger.error(msg)

    @classmethod
    def clear(cls) -> None:
        cls._logger.error(msg)
