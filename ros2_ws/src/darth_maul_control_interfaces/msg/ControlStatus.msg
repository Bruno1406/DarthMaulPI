builtin_interfaces/Time stamp

uint8 STATE_IDLE=0
uint8 STATE_EXECUTING=1
uint8 STATE_STOPPING=2
uint8 STATE_SUCCEEDED=3
uint8 STATE_CANCELED=4
uint8 STATE_FAILED=5

uint8 state
string status

bool odom_available
bool command_enabled

uint32 active_waypoint_index
uint32 waypoint_count

float64 distance_to_active_waypoint_m
float64 heading_error_rad
