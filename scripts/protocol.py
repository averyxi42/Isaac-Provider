class VelMessage():
    type = "VEL"
    def __init__(self, x=0,y=0,omega=0):
        super().__init__("VEL")
        self.x=x
        self.y=y
        self.omega = omega

class WaypointMessage():
    type = "WAYPOINT"
    def __init__(self):
        self.wps = [] #list of x,y

message_types = [VelMessage,WaypointMessage]