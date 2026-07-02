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
bool scan_available
bool command_enabled

uint8 active_primitive_type

float64 distance_remaining_m
float64 distance_traveled_m
float64 heading_error_rad
float64 front_clearance_m

# Median cardinal LiDAR sector ranges. Validity is represented by NaN
# if a sector does not have enough valid samples.
float64 front_range_m
float64 rear_range_m
float64 left_range_m
float64 right_range_m

# LiDAR grid-alignment diagnostics. NaN/false when unavailable.
bool grid_alignment_valid
bool grid_yaw_valid
float64 grid_yaw_error_rad
bool grid_lateral_valid
float64 grid_lateral_error_m
string grid_alignment_source
float64 grid_alignment_confidence

# Current straight-drive grid-yaw correction. False/zero/empty when inactive.
bool grid_yaw_correction_active
float64 grid_yaw_correction_radps
string grid_yaw_control_reason

bool left_wall_line_valid
float64 left_wall_offset_m
float64 left_wall_yaw_error_rad
float64 left_wall_rms_error_m
float64 left_wall_span_x_m
uint16 left_wall_support_count

bool right_wall_line_valid
float64 right_wall_offset_m
float64 right_wall_yaw_error_rad
float64 right_wall_rms_error_m
float64 right_wall_span_x_m
uint16 right_wall_support_count
