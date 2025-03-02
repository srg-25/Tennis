"""
This module contains logger utils
"""
import os
from pathlib import Path
import logging

# Look https://stackoverflow.com/questions/13490506/python-logging-retrieve-specific-handler/76941199#76941199
LOGGER_NAMES = {}


def create_logger(root_dir, log_name, level=logging.INFO):
    """
    Create a logger if it was not created yet. Otherwise return registered logger.
    :param root_dir: a directory to save logs in
    :param log_name: a log name
    :param level: logging level: logging.INFO, logging.DEBUG, ...
    :return: loggers
    """
    log_dir = os.path.join(root_dir, 'logs')
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = os.path.join(log_dir, log_name + '.log')

    if log_name in LOGGER_NAMES:            # This logger registered already
        l = logging.getLogger(log_name)
        l.setLevel(level)
        return l

    # Create and register new logger

    l = logging.getLogger(log_name)
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    fileHandler = logging.FileHandler(log_path, mode='a')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)

    l.setLevel(level)
    l.addHandler(fileHandler)
    l.addHandler(streamHandler)

    #  To avoid double printing
    logger = logging.getLogger(log_name)
    logger.propagate = False

    # Register this logger
    LOGGER_NAMES[log_name] = log_name
    return logger


def class_to_str(a_class):
    """
    Look:
    https://stackoverflow.com/questions/1535327/how-to-print-instances-of-a-class-using-print/50917475#50917475
    """
    return (str(a_class.__class__) + '\n'
            + '\n'.join((str(item) + ' = ' + str(a_class.__dict__[item]) for item in sorted(a_class.__dict__))))


"--------------------- Check -------------------------------"


def check_logger():
    test_log = create_logger(root_dir='..', log_name='test_log')
    test_log.info('test create_logger')


def check_class_to_str():
    class Element:
        def __init__(self, name, symbol, number):
            self.name = name
            self.symbol = symbol
            self.number = number

        def __str__(self):
            supper_str = super.__str__(self)
            return supper_str + '\n' + class_to_str(self)

    test_log = create_logger(root_dir='..', log_name='test_log_class')
    element = Element(name='New-Element', symbol='E1', number=100)
    test_log.info(f'Element class description: {element}')


def create_loger_and_log(run):
    task_name = 'create_loger_and_log'
    task_logger = create_logger(root_dir='.', log_name=f'log_{task_name}')
    task_logger.info(f'{task_name}: -------------- Start ---------------')

    task_logger.info(f'run = {run}')

    task_logger.info(f'{task_name}: -------------- End ---------------')


def check_double_logging():
    task_name = 'check_double_logging'
    task_logger = create_logger(root_dir='.', log_name=f'log_{task_name}')
    task_logger.info(f'{task_name}: -------------- Start ---------------')

    for r in range(1, 4):
        task_logger.info(f'Run#{r} Call create_loger_and_log() ->->->->->')
        create_loger_and_log(r)
        task_logger.info(f'Run#{r} create_loger_and_log() Ended -<-<-<-<-')

    task_logger.info(f'{task_name}: -------------- End ---------------')


if __name__ == "__main__":
    # check_logger()
    # check_class_to_str()
    check_double_logging()
