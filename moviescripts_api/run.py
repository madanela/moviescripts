from api.app import create_app
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from prometheus_client import make_wsgi_app

from api.config import DevelopmentConfig, setup_app_logging


_config = DevelopmentConfig()

# setup logging as early as possible
setup_app_logging(config=_config)
main_app = create_app(config_object=_config).app
application = DispatcherMiddleware(main_app.wsgi_app, 
                                          {"/metrics": make_wsgi_app()})

if __name__ == "__main__":
    main_app.run(port=_config.SERVER_PORT, host=_config.SERVER_HOST)
