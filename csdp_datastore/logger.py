import datetime
from abc import ABC, abstractmethod
from enum import Enum

import xlsxwriter as xls


class EventSeverity(Enum):
    Info = "INFO"
    Warning = "WARNING"
    Error = "ERROR"

    def __str__(self):
        return self.name


class LoggingModule:
    def __init__(self):
        self.loggers = [CmdLogger(), ExcelLogger()]

    def log(self, msg, dataset, subject, record, severity):
        timestamp = datetime.datetime.now()

        for logger in self.loggers:
            logger.log_message(msg, dataset, severity, subject, record, timestamp)

    def final(self, dataset_name):
        for logger in self.loggers:
            logger.final(dataset_name)


class Logger(ABC):
    @abstractmethod
    def log_message(self, msg, dataset, severity, subject, record, timestamp):
        pass

    def final(self, dataset_name):
        pass


class CmdLogger(Logger):
    def log_message(self, msg, dataset, severity, subject, record, timestamp):
        msg = "Log [{sev}]: {msg}. [Dataset]={d} [Subject]={s} [Record]={r} [Time]={t}".format(
            sev=severity, msg=msg, d=dataset, s=subject, r=record, t=timestamp
        )
        print(msg, flush=True)


class ExcelLogger(Logger):
    def __init__(self):
        self.messages = []

    def log_message(self, msg, dataset, severity, subject, record, timestamp):
        self.messages.append([timestamp, dataset, subject, record, severity.value, msg])

    def final(self, dataset_name):
        workbook = xls.Workbook(f"{dataset_name}_port_log.xlsx")
        worksheet = workbook.add_worksheet()
        worksheet.write_row(0, 0, ["time", "dataset", "subject", "record", "severity", "message"])
        row_index = 1

        for msg in self.messages:
            worksheet.write_row(row_index, 0, msg)
            row_index += 1

        workbook.close()


class TxtLogger(Logger):
    def __init__(self, path):
        self.path = path

    def log_message(self, msg, dataset, severity, run_index):
        if severity == EventSeverity.Info:
            return

        log_path = f"{self.path}/{run_index}-{dataset}.txt".replace(":", ".")

        with open(log_path, "a") as f:
            f.write(msg + "\n")
