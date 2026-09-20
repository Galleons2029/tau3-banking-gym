from tau3.environment.toolkit import ToolKitBase, ToolType, is_tool
from .core import Hotel, READ_TOOLS, WRITE_TOOLS

class HotelTools(Hotel, ToolKitBase):
    def __init__(self, db):
        ToolKitBase.__init__(self,db)

    @property
    def state(self):
        return self.db.state

    def get_db_hash(self):
        return self.db.get_hash()

    def update_db(self, update_data=None):
        # Every hotel task supplies a complete state snapshot. Deep-merging would
        # retain default guest IDs when a task uses a different customer profile.
        if update_data is None:return
        if set(update_data)!={'state'}:raise ValueError('hotel initialization requires a complete state snapshot')
        self.db=type(self.db).model_validate(update_data).model_copy(deep=True)

    # Upstream collects decorated methods at CLASS CREATION, not afterwards.
    for _name in READ_TOOLS:
        locals()[_name]=is_tool(ToolType.READ)(getattr(Hotel,_name))
    for _name in WRITE_TOOLS:
        locals()[_name]=is_tool(ToolType.WRITE)(getattr(Hotel,_name))
    del _name

