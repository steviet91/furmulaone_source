# furmulaone_source
Source repo for the car and track models

This is the private repo to hide the car and track models from entrants of the furmula one competition.

# Objectives of the Project
- 2D top down driving game
- Driver inputs are:
  - Throttle position (0-1)
  - Braking position (0-1)
  - Steering wheel position (-720 to +720 deg)
  - Inputs are recieved via a local UDP socket.
- Track model:
  - Left hand extremity
  - Right hand extremity
  - One basic test track
  - One more complex test track
  - One event track
- Vehicle model
  - Simple bicycle model
  - Would be nice to add tyres to it
  - If the vehicle model collides with the track edge then velocities set to zero to penalise
  - May need to provide this as a standalone model (minus the LIDAR).
- Vehicle sensors
  - Forward-facing, left-facing, right-facing LIDAR
    - Each LIDAR has X number of rays
    - User can chose the FOV, and therefore the ray density of each
    - User could move/slow scan the LIDARS? Maybe not straight away.
  - Vehicle sensors will be sent back via a UDP socket
    - LIDAR
    - WSS
    - Velocity?
    - gLong/gLat
    - Heading (global)
    - Slip angle
  - Perfect sensors will be given in dev mode. Noise will be added for race time, maybe an intermediate step needed with some noise.
