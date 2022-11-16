import os
import json
import logging
from utils.utils import get_project_root
from models.texts_storage import TextsStorage

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
root = get_project_root()

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',)

logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.INFO)

with open(os.path.join(root, "data", "config.json")) as json_config_file:
    config = json.load(json_config_file)

restart_through_seconds = config["restart_through_seconds"]
score = config["score"]
service_host = config["service_host"]
service_port = config["service_port"]
VOCABULARY_SIZE = config["VOCABULARY_SIZE"]
SHARD_SIZE = config["SHARD_SIZE"]
etalons_add_url = config["etalons_add_url"]
answer_add_url = config["answer_add_url"]
etalons_delete_url = config["etalons_delete_url"]
answer_delete_url = config["answer_delete_url"]
DB_PATH = os.path.join(root, "data", "queries.db")



text_storage = TextsStorage(db_path=DB_PATH)
