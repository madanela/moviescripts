import logging

import connexion
from api.config import Config

_logger = logging.getLogger(__name__)


def create_app(*, config_object: Config) -> connexion.App:
    """Create app instance."""

    connexion_app = connexion.App(
        __name__, debug=config_object.DEBUG, specification_dir="."
    )  # create the application instance
    flask_app = connexion_app.app
    flask_app.config.from_object(config_object)
    connexion_app.add_api("api.yaml")  # read the swagger.yml file to configure the endpoints

    _logger.info("Application instance created")

    return connexion_app
