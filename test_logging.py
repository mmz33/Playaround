#!/usr/bin/env python3

import logging
import sys

logger = logging.getLogger('logger-example')
logger.setLevel(logging.DEBUG)

# create handler
handler = logging.StreamHandler(sys.stdout) 
handler.setLevel(logging.DEBUG)

# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

logger.addHandler(handler) # add to the handlers list

logger.debug('debug msg')





