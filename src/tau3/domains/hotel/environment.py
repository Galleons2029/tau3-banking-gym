import json
from tau3.data_model.tasks import Task
from tau3.environment.environment import Environment
from .data_model import HotelDB
from .tools import HotelTools
from .utils import HOTEL_DATA_DIR
ROOT=HOTEL_DATA_DIR
def load(name):return json.loads((ROOT/name).read_text(encoding="utf-8"))

def get_environment(db=None, solo_mode=False):
    if solo_mode:
        raise ValueError('hotel v0.3 supports interactive text mode only')
    db=db if db is not None else HotelDB.model_validate(load('db.json'))
    return Environment(domain_name='hotel',policy=(ROOT/'policy.md').read_text(encoding='utf-8'),tools=HotelTools(db))

def get_tasks_split():return load('split_tasks.json')

def get_tasks(task_split_name=None):
    ids=set(get_tasks_split()[task_split_name or 'base'])
    return [Task.model_validate(t) for t in load('tasks.json') if t['id'] in ids]

def register():
    from .dialogue_scoring import install
    install()
    from tau3.registry import registry
    registry.register_domain(get_environment,'hotel')
    registry.register_tasks(get_tasks,'hotel',get_task_splits=get_tasks_split)

