from abc import ABC, abstractmethod
from enum import Enum
import datetime

class EventSeverity(Enum):
    Info = "INFO"
    Warning = "WARNING"
    Error = "ERROR"

    def __str__(self):
        return self.name

class LoggingModule:
    def __init__(self, logging_path):
        self.start_time = datetime.datetime.now() 
        self.loggers = [CmdLogger()]

    def log(self, msg, dataset, subject, record, severity):
        timestamp = datetime.datetime.now()

        msg = 'Log [{sev}]: {msg}. [Dataset]={d} [Subject]={s} [Record]={r} [Time]={t}'.format(sev=severity,
                                                                                            msg=msg,
                                                                                           d = dataset,
                                                                                           s = subject,
                                                                                           r = record, t = timestamp)
        
        for l in self.loggers:
            l.log_message(msg, dataset, severity, self.start_time)


class Logger(ABC):        
    @abstractmethod
    def log_message(self, msg, dataset, severity, run_index):
        pass


class CmdLogger(Logger):
    def log_message(self, msg, dataset, severity, run_index):
        print(msg, flush=True)
    

class TxtLogger(Logger):
    def __init__(self, path):
        self.path = path
    
    def log_message(self, msg, dataset, severity, run_index):
        if severity == EventSeverity.Info:
            return
        
        log_path = f"{self.path}/{run_index}-{dataset}.txt".replace(":", ".")

        with open(log_path, "a") as f:
            f.write(msg+'\n')
