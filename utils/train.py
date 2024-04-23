import sys
import logging




def signal_handler(sig, frame):
    logging.info("Stopping early...")
    sys.exit(0)
