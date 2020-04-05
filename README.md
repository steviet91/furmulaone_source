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


# So what do we need
- Classes
  - Ray
    - Contains a point and direction of the ray
    - Needs to be pointing relative to the vehicle
    - Therefore direction must be updated based on heading
    - Needs a function to intersect with a line segment
  - Vehicle collision
    - Contains four rays - one pointing to each of the four corners of the car
    - Needs a has_collided method, this checks is the ray collides with a line segment that is equal to or closer than the vehicle extremitiy
  - Lidars
    - Needs a direction relative to the vehicle forward vector
    - Needs a FOV (limit, say 20-120 deg)
    - Needs a static no. rays (say 20)
    - Create array of rays pointing relative to vehicle + direction offset + FOV offset.
    - Feeds back an array of intersect points
    - Needs the ability to turn on sensor noise (and degree of sensor noise)
  - Track
    - Track is made from two sets of line segments; LH and RH extremities
    - Track needs a method to return the X number of closest line segments, (based on sLap?)
    - Needs a start finish line (and sector lines?) for timing.
    - Fixed MU value
    - Track is on a flat plane
  - Vehicle model
    - Bicycle
    - Inputs; throttle, brake, steering
    - Needs a torque curve (assume EV, RWD, no TV, open diff)
    - Steering ratio needed to turn steering wheel into front wheel steering angle
    - Braking will apply a negative torque 50/50 to front and rear
    - Traction control?
    - Vehicle parameters; wheelbase, cog position, mass, track (for collision only), tyre ellipse, aero and rolling resistance
    - States of interest; vVehicle, gLat, gLont, nWheelF, nWheelR, aHeading, nYaw, aSlipBody, rSlipTyre?
    - Needs three LIDAR objects, one on front, one of left and one on right. Limit the nom LIDAR angles to +/- X deg of each direction (i.e. Left must be pointing left to within +/- X deg).
  - Visualiser
    - Draw the track
    - Draw the vehicle
    - Draw the LIDARS if turned on
  - Socket
    - Two comms, in and out
    - In needs to read driver inputs and send to vehicle model
    - Out needs to read vehicle states and LIDARS and send to driver 
    
