import logging
import os

class LOG():
    def __init__(self,
                 name='LOG', 
                 dir='./log',
                 file='testlog.log',
                 level='DEBUG'
                 ):
        if not os.path.isdir(dir):
            os.mkdir(dir)
        self.path = os.path.join(dir, file)
        self.logger = logging.getLogger(name=name)
        self.formatter =logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        if level == 'DEBUG':
            self.logger.setLevel(logging.DEBUG)
        elif level == 'INFO':
            self.logger.setLevel(logging.INFO)
        elif level == 'WARNING':
            self.logger.setLevel(logging.WARNING)
        elif level == 'ERROR':
            self.logger.setLevel(logging.ERROR)
        else:
            self.logger.setLevel(logging.CRITICAL)
        self.file_handler = logging.FileHandler(self.path)
        self.file_handler.setFormatter(self.formatter)
        self.logger.addHandler(self.file_handler)

    def __len__(self):
        with open(self.path, 'r') as f:
            lines = f.readlines()
        return len(lines)

    def log_debug(self, str):
        self.logger.debug(str)
    
    def output_debug(self, str):
        self.logger.debug(str)
        print(str)

    def log_info(self, str):
        self.logger.info(str)
    
    def output_info(self, str):
        self.logger.info(str)
        print(str)

    def log_warning(self, str):
        self.logger.warning(str)
    
    def output_warning(self, str):
        self.logger.warning(str)
        print(str)
    
    def log_error(self, str):
        self.logger.error(str)
    
    def output_error(self, str):
        self.logger.error(str)
        print(str)

    def log_critical(self, str):
        self.logger.critical(str)
    
    def output_critical(self, str):
        self.logger.critical(str)
        print(str)

class LOG_Limited(LOG):
    def __init__(self, 
                 name='LOG', 
                 dir='./log',
                 file='testlog.log',
                 level='DEBUG',
                 max=0):
        super().__init__(name=name, dir=dir, file=file, level=level)
        self.max = max
    
    def limit_log(self):
        try:
            f = open(self.path, 'r')
            lines = f.readlines()
            if self.max:
                while len(lines) > self.max:
                    lines.pop(0)
            f.close()
            with open(self.path, 'w+') as fw:
                fw.writelines(lines)
        except IOError:
            pass

    def log_debug(self, str):
        self.logger.debug(str)
        self.limit_log()
    
    def output_debug(self, str):
        self.logger.debug(str)
        print(str)
        self.limit_log()

    def log_info(self, str):
        self.logger.info(str)
        self.limit_log()
    
    def output_info(self, str):
        self.logger.info(str)
        print(str)
        self.limit_log()

    def log_warning(self, str):
        self.logger.warning(str)
        self.limit_log()
    
    def output_warning(self, str):
        self.logger.warning(str)
        print(str)
        self.limit_log()
    
    def log_error(self, str):
        self.logger.error(str)
        self.limit_log()
    
    def output_error(self, str):
        self.logger.error(str)
        print(str)
        self.limit_log()

    def log_critical(self, str):
        self.logger.critical(str)
        self.limit_log()
    
    def output_critical(self, str):
        self.logger.critical(str)
        print(str)
        self.limit_log()

if __name__ == '__main__':
    log = LOG_Limited(level='DEBUG', max=27)
    log.output_debug('This is a debug message')
    log.output_info('This is a info message')
    log.output_warning('This is a warning message')
    log.output_error('This is a error message')
    log.output_critical('This is a critical message')
    print(len(log))
