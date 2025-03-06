from utils.do_logging import logger
import uvicorn
import config
from utils.pre_start_init import app
import routes, models

try:
    if __name__ == '__main__':

        uvicorn.run(app, host=config.HOST, port=config.PORT)
except KeyboardInterrupt:
    logger.info('\nDone')
except Exception as e:
    logger.error(f'\nDone with error {e}')

