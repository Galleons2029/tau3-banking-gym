from pydantic import Field
from tau3.environment.db import DB
from .core import canonical_state

class HotelDB(DB):
    state: dict = Field(description='Hotel business state, including fixed simulation clock.')

    def get_hash(self):
        from tau3.utils import get_dict_hash
        return get_dict_hash({'state':canonical_state(self.state)})

