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

uint8 active_primitive_type
float64 distance_remaining_m
float64 distance_traveled_m
float64 heading_error_rad
