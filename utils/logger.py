import logging
import datetime
#####################################
# Setup Logging
# TODO: integrate to be used on each module with master config when running various CLI commands
#####################################
logging.basicConfig(level=logging.DEBUG)
logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
rootLogger = logging.getLogger()

fileHandler = logging.FileHandler("./logs/{0}/{1}.log".format("./", f"rx50-{datetime.datetime.now():%d-%b-%y-%H:%M:%S}"))
fileHandler.setFormatter(logFormatter)
rootLogger.addHandler(fileHandler)

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)