#NOTE TO FUTURE SO I DON'T EFF THIS UP.
# I want this file to be dedicated to the actual mechanics (the cars, roads, signs, etc)

import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

try:
    import site
    for sp in site.getsitepackages():
        torch_lib = os.path.join(sp, "torch", "lib")
        if os.path.exists(torch_lib):
            os.environ["PATH"] = torch_lib + ";" + os.environ.get("PATH", "")
            if hasattr(os, "add_dll_directory"):
                try:
                    os.add_dll_directory(torch_lib)
                except Exception:
                    pass
            break
except Exception:
    pass

from ursina import *
from ursina import camera
import DataCollection


#NOTE: spline and spline tan functions for Catmull-Rom generation
def spline(p0, p1, p2, p3, t):
    #P(t) = 1/2 TMG
    #where T = [t^3, t^2, t, 1]
    # M =     [-1,  3, -3,  1]
    #         [ 2, -5,  4, -1]
    #         [-1,  0,  1,  0]
    #         [ 0,  2,  0,  0]
    #AND LAST BUT NOT LEAST
    # G = [p0, p1, p2, p3]^T (transpose, I have to clue if that's how it should be represented :/)


    T = np.array([t**3, t**2, t**1, 1])

    M = np.array([
        [-1,  3, -3,  1],
        [ 2, -5,  4, -1],
        [-1,  0,  1,  0],
        [ 0,  2,  0,  0]
    ])

    G = np.array([p0, p1, p2, p3])

    return .5*T @ M @ G

def splineTan(p0, p1, p2, p3, t):
    # dP(t)/dt = 1/2 T'MG
    #rest is the pretty much the same, we're justing getting the tan for P
    TPRIME = np.array([3*t**2, 2*t**1, 1, 0]) # first derivative of T from spline (obv)

    M = np.array([
        [-1,  3, -3,  1],
        [ 2, -5,  4, -1],
        [-1,  0,  1,  0],
        [ 0,  2,  0,  0]
    ])

    G = np.array([p0, p1, p2, p3])

    direction = .5*TPRIME @ M @ G

    #normalize it (direction / sqrt(direction * direction)) <- matrix multiplication
    unitDirection = direction / sqrt((direction)@(direction))

    return unitDirection


#Destroy all childern of parent as, for some stupid bs, ursina doesn't offer a nice way to do that, which is, BEYOND ME
def destroy_all(entity):
    """destroy() doesn't cascade to children in ursina — do it ourselves."""
    if not entity:
        return
    for child in list(entity.children):
        destroy_all(child)
    destroy(entity)


#Fix for stupid bug where stop sign spawns right infront of the car's spawn and causing the model to get stuck:

# Minimum gap between consecutive stop signs. A driver needs room to complete one stop
# and get back to speed before the next; without this a dense course produces pairs
# 25 units apart, which nothing in the recorded data resembles.
MIN_STOP_SIGN_SPACING = 200.0


#Find the next point to generate the road to
def nextMilestone(lastPoint, rng=None):
    rng = rng or random
    distanceGenerated = 300
    forwardStep = lastPoint.z + distanceGenerated

    xCurve = rng.uniform(-25, 25)
    yCurve = rng.uniform(-5, 5)

    newPoint = Vec3(xCurve, yCurve, forwardStep)

    return newPoint


#frenet serret reference
#basically, use splinetan to grab the normalized direction, take a cross product to get up (right hand rule, foward vector cross up vector = right vector)
#then using the road with and everything to set the middle, road edges, etc.
def generateRoad(controlPoints, roadWidth, grassWidth, resolution):
    vertices = []
    triangles = []
    colors = []

    GLOBAL_UP = Vec3(0, 1, 0)
    currentIndex = 0

    for i in range(1, len(controlPoints) - 2):
        p0 = controlPoints[i - 1]
        p1 = controlPoints[i]
        p2 = controlPoints[i + 1]
        p3 = controlPoints[i + 2]

        for j in range(0, resolution + 1):
            t = j / resolution
            center = spline(p0, p1, p2, p3, t)
            forward = splineTan(p0, p1, p2, p3, t)

            rawRightV = np.cross(forward, GLOBAL_UP)
            right = rawRightV / sqrt(rawRightV @ rawRightV)

            halfRoadWidth = roadWidth / 2.0
            lineWidth = 0.15

            # --- SHARP COLOR ISOLATION GEOMETRY ---
            # We calculate distinct, tight boundaries for the lines to stop bleeding
            vLeftGrass = center - (right * (halfRoadWidth + grassWidth))
            vLeftLineOuter = center - (right * halfRoadWidth)
            vLeftLineInner = center - (right * (halfRoadWidth - lineWidth))

            # Asphalt lane interior boundary points
            vLeftLaneEnd = center - (right * (lineWidth * 1.5))
            vCenterLeft = center - (right * (lineWidth / 2.0))
            vCenterRight = center + (right * (lineWidth / 2.0))
            vRightLaneStart = center + (right * (lineWidth * 1.5))

            vRightLineInner = center + (right * (halfRoadWidth - lineWidth))
            vRightLineOuter = center + (right * halfRoadWidth)
            vRightGrass = center + (right * (halfRoadWidth + grassWidth))

            # Append all 10 structural vertices to clear color channels
            for v in [vLeftGrass, vLeftLineOuter, vLeftLineInner, vLeftLaneEnd,
                      vCenterLeft, vCenterRight, vRightLaneStart, vRightLineInner, vRightLineOuter, vRightGrass]:
                vertices.append(Vec3(v[0], v[1], v[2]))

            # Procedural Dashed Center Line
            if (j // 3) % 2 == 0:  # Widened dash step slightly for clean visibility
                centerMarkingColor = color.yellow
            else:
                centerMarkingColor = color.gray

            # Assign colors strictly to prevent gradient leaking
            colors.extend([
                color.green,  # 0: Left Grass
                color.white,  # 1: White Line Outer
                color.gray,  # 2: White Line Inner -> Lane Gray
                color.gray,  # 3: Mid Lane Gray
                centerMarkingColor,  # 4: Center Line Left
                centerMarkingColor,  # 5: Center Line Right
                color.gray,  # 6: Mid Lane Gray
                color.gray,  # 7: Lane Gray -> White Line Inner
                color.white,  # 8: White Line Outer
                color.green  # 9: Right Grass
            ])

            # --- SEAMLESS TOPOLOGY SEWING ---
            # Condition: Skip ONLY the absolute beginning of the entire track
            if not (i == 1 and j == 0):
                prev = currentIndex - 10
                curr = currentIndex

                # Loop to automatically stitch all 9 parallel quad strips
                for strip in range(9):
                    triangles.append((prev + strip, curr + strip, prev + strip + 1))
                    triangles.append((curr + strip, curr + strip + 1, prev + strip + 1))

            currentIndex += 10

    return vertices, triangles, colors


# NOTE: ENTITY CLASSES

#weather class, allows for setting all the weather stuff, not really any complicated logic
#NoTE: in case I run into the same bug later, the color.rgb values NEED to be normalized (value/255)
#NOTE FOR FUTURE SELF, this is terrible for performance, swap to rendering it all as a single, larger entity than gets shifted and stuff
class WeatherSystem(Entity):
    def __init__(self, target, **kwargs):
        super().__init__(**kwargs)
        self.target = target
        self.mode = 'clear'
        self.max_particles = 150
        self.particles = []
        self.spawn_radius = 22
        self.fall_speed = 14
        self.spawn_height = 18
        self.floor_offset = -2
        self.overlay_color = None

        for i in range(self.max_particles):
            p = Entity(model='cube', scale=0.05, enabled=False, parent=self)
            self.particles.append(p)

    def set_weather(self, mode):
        self.mode = mode

        if mode == 'clear':
            self.spawn_height = 18
            camera.overlay.color = color.clear
            self.overlay_color = None
            for p in self.particles:
                p.enabled = False
            return

        if mode == 'rain':
            self.spawn_height = 18
            camera.overlay.color = color.rgba(30 / 255, 35 / 255, 45 / 255, 110 / 255)
            self.overlay_color = (30, 35, 45, 110)
            self._configure_particles(
                scale=(0.02, 0.25, 0.02),
                col=color.rgba(180 / 255, 200 / 255, 220 / 255, 160 / 255),
                fall_speed=28
            )
        elif mode == 'snow':
            self.spawn_height = 18
            camera.overlay.color = color.rgba(210 / 255, 215 / 255, 225 / 255, 35 / 255)
            self.overlay_color = (210, 215, 225, 35)
            self._configure_particles(
                scale=(0.06, 0.06, 0.06),
                col=color.white,
                fall_speed=6
            )
        elif mode == 'sandstorm':
            self.spawn_height = 8
            camera.overlay.color = color.rgba(190 / 255, 150 / 255, 90 / 255, 90 / 255)
            self.overlay_color = (190, 150, 90, 90)
            self._configure_particles(
                scale=(0.08, 0.08, 0.08),
                col=color.rgba(200 / 255, 160 / 255, 100 / 255, 200 / 255),
                fall_speed=2
            )

        # Distribute particles
        for p in self.particles:
            p.enabled = True
            p.position = Vec3(
                random.uniform(-self.spawn_radius, self.spawn_radius),
                random.uniform(0, self.spawn_height),
                random.uniform(-12, 25)
            )

    def _configure_particles(self, scale, col, fall_speed):
        self.fall_speed = fall_speed
        for p in self.particles:
            p.scale = scale
            p.color = col

    #some extra logic added to keep weather in pace with car
    def update(self):
        if self.mode == 'clear':
            return

        # Keep weather centered and oriented with the vehicle
        self.position = self.target.world_position
        self.rotation_y = self.target.rotation_y

        speed = getattr(self.target, 'velocity', 0)
        drift_x = 6 if self.mode == 'sandstorm' else 0  # sideways blow for sandstorm

        for p in self.particles:
            # 1. Fall vertically in local space
            p.y -= self.fall_speed * time.dt
            # 2. Sideways drift (e.g., sandstorm)
            p.x += drift_x * time.dt
            # 3. Vehicle driving forward moves particles BACKWARD relative to the car
            p.z -= speed * time.dt

            # Wrap conditions in local box relative to vehicle:
            # Hit floor -> respawn near top
            if p.y < self.floor_offset:
                p.y = random.uniform(self.spawn_height - 3.0, self.spawn_height)
                p.x = random.uniform(-self.spawn_radius, self.spawn_radius)
                p.z = random.uniform(-10.0, 25.0)

            # Passed behind car/chase camera -> wrap to front of vehicle
            elif p.z < -18.0:
                p.z = random.uniform(18.0, 25.0)
                p.y = random.uniform(2.0, self.spawn_height)
                p.x = random.uniform(-self.spawn_radius, self.spawn_radius)

            # Driven backward past front -> wrap to back of vehicle
            elif p.z > 25.0:
                p.z = random.uniform(-18.0, -12.0)
                p.y = random.uniform(2.0, self.spawn_height)
                p.x = random.uniform(-self.spawn_radius, self.spawn_radius)

            # Drifted too far sideways -> wrap back into lane width
            elif abs(p.x) > (self.spawn_radius + 5.0):
                p.x = random.uniform(-self.spawn_radius, self.spawn_radius)


#Literally a crappy looking pedestrian, literally doesn't do anything else
class Pedestrian(Entity):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Torso (The root basis)
        self.torso = Entity(model='cube', color=color.blue, scale=(0.5, 0.8, 0.3), y=0.9, parent=self)
        # Head
        self.head = Entity(model='cube', color=color.peach, scale=(0.3, 0.3, 0.3), y=1.45, parent=self)
        # Left Leg
        self.leg_l = Entity(model='cube', color=color.brown, scale=(0.2, 0.5, 0.2), y=0.25, x=-0.15, parent=self)
        # Right Leg
        self.leg_r = Entity(model='cube', color=color.brown, scale=(0.2, 0.5, 0.2), y=0.25, x=0.15, parent=self)
        # Arms (simplified as a single bar passing through the shoulder)
        self.arms = Entity(model='cube', color=color.blue, scale=(0.9, 0.2, 0.2), y=1.1, parent=self)



#STOP SIGNS yay
#basically defines the model. then the rotation by the direction of the road, the offset based on road width
#makes a ground strip (like there is irl, a line you're supposed to stop infront of)
#added a bounding box, so when the car stops infront of it (car.velocity ~= 0) it turns green in 1.5 seconds
#note, I have it turn green since I use a single frame cnn, so, basically this is a stop light xddd
#Could've added some gimmick to the model to check if it has stopped blah blah but that feels cheep imo
#also, think this order is right, had object cleaned up by ai (since it was a mess) at some point so idk fs :/
class StopSign(Entity):
    def __init__(self, position, road_direction, roadWidth=10, **kwargs):
        # We explicitly pass the position into super() so the entity anchors before calculating children
        super().__init__(position=position, **kwargs)

        # 1. The Metal Pole
        self.pole = Entity(model='cube', color=color.light_gray,
                           scale=(0.1, 4, 0.1), y=2, z=5, parent=self)

        # 2. The Red Sign Backing
        self.sign = Entity(model='cube', color=color.red,
                           scale=(1.2, 1.2, 0.1), y=4, z=-0.03 +5, parent=self)

        # 3. Simple visible white strip placeholder for the word "STOP"
        self.text_stripe = Entity(model='cube', color=color.white,
                                  scale=(0.8, 0.2, 0.12), y=4, z=-0.05 + 5, parent=self)

        # 4. Turn the entire master assembly to face the road direction
        self.look_at(self.position + road_direction)

        # 5. Lock the master pole so it stays perfectly vertical, ignoring steep hills
        self.rotation_x = 0
        self.rotation_z = 0

        # --- ENCAPSULATED GROUND STRIP ---
        # Because the pole is perfectly vertical and facing forward, we can jump perfectly
        # sideways (negative local X) to hit the exact center of the lane.
        x_offset = -(roadWidth / 4 + 1.5)

        self.groundStrip = Entity(
            model='cube',
            color=color.white,
            scale=(roadWidth / 2, 0.2, 1.5),
            parent=self,
            position=Vec3(x_offset, 0.15,-2)  # y=0.15 floats it just above the asphalt
        )

        # Tell the child strip to tilt up/down to match the hill slope perfectly!
        lane_center_world = self.groundStrip.world_position
        self.groundStrip.look_at(lane_center_world + road_direction)

        # --- NEW: Trigger Box across the lane ---
        # Position it across the road lane (x offset towards road center)
        x_offset = -(roadWidth / 4 + 1.5)
        self.trigger = Entity(
            model='cube',
            scale=(roadWidth, 10, 22),  # Expanded depth (Z=22) so stopping short still triggers the sign
            position=Vec3(x_offset, 2, -8),  # Positioned further back along approaching lane
            parent=self,
            collider='box',
            visible=False,  # Keep it invisible
            enabled=True
        )
        self.has_triggered = False  # Prevent firing multiple frames in a row


    def swapToGreen(self):
        self.sign.color=color.lime


#Speed limit sign
# I think I literally copied like 90 percent of this from stop sign, just changed the bounding box to update speed in the main update?
class speedLimitSign(Entity):
    def __init__(self, position, road_direction, roadWidth=10, rng=None, **kwargs):
        # We explicitly pass the position into super() so the entity anchors before calculating children
        super().__init__(position=position, **kwargs)

        # 1. The Metal Pole
        self.pole = Entity(model='cube', color=color.light_gray,
                           scale=(0.1, 4, 0.1), y=2, parent=self)

        # 2. The Red Sign Backing
        self.sign = Entity(model='cube', color=color.white,
                           scale=(1.2, 1.2, 0.1), y=4, z=-0.03, parent=self)

        self.textWritten = str((rng or random).randrange(25, 60, 5))

        # 3. Simple visible white strip placeholder for the word "STOP"
        self.text_stripe = Text(text = self.textWritten, parent=self.sign, color=color.black, position=(0,0, -0.9), origin=(0,0))
        self.text_stripe.world_scale = 30
        # 4. Turn the entire master assembly to face the road direction
        self.look_at(self.position + road_direction)

        # 5. Lock the master pole so it stays perfectly vertical, ignoring steep hills
        self.rotation_x = 0
        self.rotation_z = 0

        # --- NEW: Trigger Box across the lane ---
        # Position it across the road lane (x offset towards road center)
        x_offset = -(roadWidth / 4 + 1.5)
        self.trigger = Entity(
            model='cube',
            scale=(roadWidth, 6, 2),  # Widen to cover the whole road & height
            position=Vec3(x_offset, 2, 0),
            parent=self,
            collider='box',
            visible=False,  # Keep it invisible
            enabled=True
        )
        self.has_triggered = False  # Prevent firing multiple frames in a row

    def getSpeedLimit(self):
        return int(self.textWritten)


#car, handles the model, all the driving physics (thank you ai)
#also handles some of the ui relating the car, like current speed, speed limit of the car's current zone, etc.
#NOTE: CLEAN THIS FUNCTION UP LATER, there is way to much leftover junk from the different versions
class Car(Entity):
    def __init__(self, carColor,**kwargs):
        super().__init__(model = 'cube', color=carColor, scale=(2,3/2,2), collider='box', **kwargs)

        self.bodyBorder = Entity(model='cube', color=color.black, scale=1.001, parent=self, wireframe=True)

        self.front = Entity(model='cube', color=carColor, scale=(1,1.25/3,1/3), collider='box', parent=self, position=Vec3(0,-.5 + 1.25/6,.5 + 1/6))
        self.frontBorder = Entity(model='cube', color=color.black, scale=1.001, parent=self.front, wireframe=True)


        wheel_scale = (0.3, 0.3, 0.3)
        wheel_rotation = Vec3(0, 0, 90)

        # FIX: Changed 'sides' to 'resolution', removed 'start=0'
        procedural_wheel1 = Cylinder(resolution=16, height=1, radius=0.5, direction=(0, 1, 0))



        # Front Right
        self.wheel1 = Entity(model=copy(procedural_wheel1), color=color.black, parent=self,
                             scale=wheel_scale, rotation=wheel_rotation,
                             position=Vec3(0.55-.3, -0.5, 0.4 + .15))

        # Front Left
        self.wheel2 = Entity(model=copy(procedural_wheel1), color=color.black, parent=self,
                             scale=wheel_scale, rotation=wheel_rotation,
                             position=Vec3(-0.55, -0.5, 0.4 + .15))

        # Back Right
        self.wheel3 = Entity(model=copy(procedural_wheel1), color=color.black, parent=self,
                             scale=wheel_scale, rotation=wheel_rotation,
                             position=Vec3(0.55-.3, -0.5, -0.4))

        # Back Left
        self.wheel4 = Entity(model=copy(procedural_wheel1), color=color.black, parent=self,
                             scale=wheel_scale, rotation=wheel_rotation,
                             position=Vec3(-0.55, -0.5, -0.4))

        #CAMERAS
        # self.windshieldCentered = Entity(model='cube', color=color.black, parent=self, scale=(0.1, 0.1, 0.1), position=Vec3(0, .45, .52), rotation_x=0)
        # self.windshieldLeft = Entity(model='cube', color=color.red, parent=self, scale=(0.1, 0.1, 0.1), position=Vec3(-.45, .45, .52), rotation_x=0)
        # self.windshieldRight = Entity(model='cube', color=color.black, parent=self, scale=(0.1, 0.1, 0.1), position=Vec3(.45, .45, .52), rotation_x=0)
        # self.thirdPerson = Entity(model='cube', color=color.black, parent=self, scale=(0.1, 0.1, 0.1), position=Vec3(0, 1.5, -8), rotation_x=15)
        self.windshieldCentered = Entity(parent=self, position=Vec3(0, .45, .52), rotation_x=12)
        self.windshieldLeft = Entity(parent=self, position=Vec3(-.45, .45, .52), rotation_x=12)
        self.windshieldRight = Entity(parent=self, position=Vec3(.45, .45, .52), rotation_x=12)
        self.thirdPerson = Entity(parent=self, position=Vec3(0, 5, -15), rotation_x=10)



        #variables
        self.velocity = 0
        self.wheelAngle = 0 #[-540,540]
        self.acceleration = 0
        self.breakPedal = 0 #[0,1]
        self.accelerationPedal = 0 #[0,1]
        # --- Tweakable Tuning Values ---
        self.MAX_SPEED = 90.0  # Top speed capability
        self.MAX_ACCEL = 38.0  # Punchiness off the line (0-30 mph feel)
        self.MAX_BRAKE = 35.0  # Brakes are usually 2-3x stronger than engine!
        self.FRICTION = 0.02  # Natural slowing down (air resistance/rolling)

        self.Speedometer = Text(
            text="Speed: 0 miles/hr",
            position=(-0.8, 0.45),
            scale=2,
            color=color.white
        )

        self.SpeedLimitText = Text(
            text="Speed Limit: 45 MPH",
            position=(-0.8, 0.38),
            scale=1.5,
            color=color.light_gray
        )

        self.statusCircle = Entity(
            parent=camera.ui,
            model='circle',
            color=color.red,
            scale=(0.04, 0.04),
            position=(-0.82, 0.31)
        )

        # Sits beside the recording light: lights up light blue while a model is driving,
        # so it is obvious at a glance whether you or the net is in control.
        self.autopilotCircle = Entity(
            parent=camera.ui,
            model='circle',
            color=color.rgba32(40, 40, 55, 255),
            scale=(0.04, 0.04),
            position=(-0.75, 0.31)
        )
        self.autopilotLabel = Text(
            text="AUTO",
            parent=camera.ui,
            position=(-0.72, 0.325),
            scale=0.75,
            color=color.rgba32(90, 90, 110, 255)
        )

    def setAutopilotIndicator(self, driving):
        on = color.rgba32(90, 200, 255, 255)
        off = color.rgba32(40, 40, 55, 255)
        self.autopilotCircle.color = on if driving else off
        self.autopilotLabel.color = on if driving else color.rgba32(90, 90, 110, 255)

    def set_status_light(self, state):
        """Sets status light circle to green (True/'green') or red (False/'red')."""
        if state is True or state == 'green' or state == color.green:
            self.statusCircle.color = color.green
        elif state is False or state == 'red' or state == color.red:
            self.statusCircle.color = color.red
        elif isinstance(state, Color):
            self.statusCircle.color = state

    def set_status_green(self):
        self.set_status_light('green')

    def set_status_red(self):
        self.set_status_light('red')

    def update(self):
        # Clamp physics step delta-time to max 30 FPS step (0.033s) to prevent physics tunneling during lag spikes
        dt = min(time.dt, 0.033)

        # --- 1. SMOOTH PEDAL INPUTS ---
        pedal_response_speed = 4.0
        if held_keys['w']:
            self.accelerationPedal = min(1.0, self.accelerationPedal + dt * pedal_response_speed)
        else:
            self.accelerationPedal = max(0.0, self.accelerationPedal - dt * pedal_response_speed)

        if held_keys['s']:
            self.breakPedal = min(1.0, self.breakPedal + dt * pedal_response_speed)
        else:
            self.breakPedal = max(0.0, self.breakPedal - dt * pedal_response_speed)

        # --- 2. STEERING WHEEL INPUT ---
        steer_speed = 750.0
        return_speed = 900.0
        if held_keys['a']:
            self.wheelAngle = max(-540.0, self.wheelAngle - steer_speed * dt)
        elif held_keys['d']:
            self.wheelAngle = min(540.0, self.wheelAngle + steer_speed * dt)
        else:
            if self.wheelAngle > 10:
                self.wheelAngle -= return_speed * dt
            elif self.wheelAngle < -10:
                self.wheelAngle += return_speed * dt
            else:
                self.wheelAngle = 0.0


# --- 3. PHYSICALLY REALISTIC FORCES ---
        steer_radians = math.radians((self.wheelAngle / 540.0) * 35.0)

        # Brake overrides throttle entirely — no more pedal-fighting at low speed
        effective_pedal = 0.0 if self.breakPedal > 0.05 else self.accelerationPedal

        DRAG_COEFF = 0.0012      # quadratic drag — mostly matters at high speed
        # self.FRICTION (0.02) is your linear rolling resistance, used as-is

        # Smooth ease-in over the first ~14 units of speed, but with a floor
        # so there's still real force at a dead stop (this is what actually
        # lets the car launch at all)
        LAUNCH_FLOOR = 0.35
        launch_t = min(1.0, abs(self.velocity) / 14.0)
        smooth_t = launch_t * launch_t * (3 - 2 * launch_t)
        launch_ramp = LAUNCH_FLOOR + (1.0 - LAUNCH_FLOOR) * smooth_t
        # Torque floor tuned so full-throttle equilibrium lands right at MAX_SPEED
        # instead of plateauing early. Nudge TORQUE_FLOOR between 0.28-0.32 to
        # fine-tune exactly where top speed settles.
        TORQUE_FLOOR = 0.30
        speed_ratio = min(1.0, abs(self.velocity) / self.MAX_SPEED)
        torque_profile = max(TORQUE_FLOOR, 1.0 - speed_ratio * speed_ratio)

        engine_force = effective_pedal * self.MAX_ACCEL * torque_profile * launch_ramp
        total_resistance = (self.velocity * self.FRICTION) + (self.velocity * abs(self.velocity) * DRAG_COEFF)

        self.acceleration = engine_force - total_resistance
        self.velocity += self.acceleration * time.dt

        # Kill residual creep so it settles exactly at 0
        if effective_pedal == 0 and abs(self.velocity) < 0.05:
            self.velocity = 0.0

        # --- Braking (flat deceleration, not proportional to speed, so it
        # works just as well at 15 mph as at 80) ---
        if self.breakPedal > 0 and abs(self.velocity) > 0.01:
            brake_drop = self.breakPedal * self.MAX_BRAKE * time.dt
            if self.velocity > 0:
                self.velocity = max(0.0, self.velocity - brake_drop)
            else:
                self.velocity = min(0.0, self.velocity + brake_drop)

        self.velocity = max(-self.MAX_SPEED, min(self.MAX_SPEED, self.velocity))

#         # 2. SMOOTH GRADUAL COASTING OVERRIDE (GENTLE ROLL TO A STOP)
#         if self.accelerationPedal == 0 and self.velocity != 0:
#             direction = 1 if self.velocity > 0 else -1
#
#             # Base ratio up to 45 mph
#             speed_ratio = min(1.0, abs(self.velocity) / 45.0)
#
#             # A smooth, standard linear blend
#             smooth_curve = speed_ratio
#
#             # Lowered low-end deceleration from 5.8 to 2.2 for a soft, gradual stop.
#             # High-speed coasting remains a loose 1.0.
#             decel_floor = lerp(2.2, 1.0, smooth_curve)
#
#             # Apply the deceleration smoothly against the movement direction
#             self.velocity -= decel_floor * direction * time.dt
#
#             # Hard check to ensure it finishes neatly at zero instead of floating
#             if (direction == 1 and self.velocity < 0) or (direction == -1 and self.velocity > 0):
#                 self.velocity = 0.0
#
# # NOTE FoR ME: CHANGE SCRAWLING NOT GOING TO 0 (rn infly->0), also, lose speed to quick at top, and accelerate too quick
#
#         # 3. Precise Braking (keeps 'S' pedal sharp)
#         if self.breakPedal > 0 and abs(self.velocity) > 0.01:
#             brake_drop = self.breakPedal * self.MAX_BRAKE * time.dt
#             if self.velocity > 0:
#                 self.velocity = max(0.0, self.velocity - brake_drop)
#             else:
#                 self.velocity = min(0.0, self.velocity + brake_drop)
        # # --- 1. SMOOTH PEDAL INPUTS --- (Keep your existing pedal code here)
        # pedal_response_speed = 4.0
        # if held_keys['w']:
        #     self.accelerationPedal = min(1.0, self.accelerationPedal + time.dt * pedal_response_speed)
        # else:
        #     self.accelerationPedal = max(0.0, self.accelerationPedal - time.dt * pedal_response_speed)
        #
        # if held_keys['s']:
        #     self.breakPedal = min(1.0, self.breakPedal + time.dt * pedal_response_speed)
        # else:
        #     self.breakPedal = max(0.0, self.breakPedal - time.dt * pedal_response_speed)
        #
        # # --- 2. STEERING WHEEL INPUT ---
        # steer_speed = 750.0
        # return_speed = 900.0
        # if held_keys['a']:
        #     self.wheelAngle = max(-540.0, self.wheelAngle - steer_speed * time.dt)
        # elif held_keys['d']:
        #     self.wheelAngle = min(540.0, self.wheelAngle + steer_speed * time.dt)
        # else:
        #     if self.wheelAngle > 10:
        #         self.wheelAngle -= return_speed * time.dt
        #     elif self.wheelAngle < -10:
        #         self.wheelAngle += return_speed * time.dt
        #     else:
        #         self.wheelAngle = 0.0
        #
        # # --- 3. PHYSICALLY REALISTIC FORCES (NEW ENGINE & DRAG MODEL) ---
        # # Normalize wheel angle to standard front-tire turn degrees [-35, 35]
        # steer_radians = math.radians((self.wheelAngle / 540.0) * 35.0)
        #
        # # Split friction into dynamic categories
        # DRAG_COEFF = 0.003  # Air resistance scales with velocity squared
        # ROLLING_RESIST = 0.04  # Constant tire resistance (stops rocket take-offs)
        #
        # # 1. Smooth Launch Limiter: Prevents the 0->30 rocket launch
        # # Ramps power cleanly from 35% up to 100% as you transition from 0 to 25 mph
        # launch_ramp = lerp(0.35, 1.0, min(1.0, abs(self.velocity) / 25.0))
        #
        # # 2. High-speed torque profile drops off near top speed
        # torque_profile = max(0.1, 1.0 - (abs(self.velocity) / self.MAX_SPEED))
        #
        # # Calculate final engine force with the launch buffer applied
        # engine_force = self.accelerationPedal * self.MAX_ACCEL * torque_profile * launch_ramp
        #
        # # Calculate resistance forces (noticeably weaker at low speeds now)
        # total_resistance = (self.velocity * ROLLING_RESIST) + (self.velocity * abs(self.velocity) * DRAG_COEFF)
        #
        # self.acceleration = engine_force - total_resistance
        # self.velocity += self.acceleration * time.dt
        #
        # # Precise Braking (keeps brakes sharp when you actually hit 'S')
        # if self.breakPedal > 0 and abs(self.velocity) > 0.01:
        #     brake_drop = self.breakPedal * self.MAX_BRAKE * time.dt
        #     self.velocity = max(0.0, self.velocity - brake_drop) if self.velocity > 0 else min(0.0,
        #                                                                                        self.velocity + brake_drop)
        #
        # # Power delivery curve: Makes the start smooth, peaking in mid-range torque
        # torque_profile = max(0.1, 1.0 - (abs(self.velocity) / self.MAX_SPEED))
        # engine_force = self.accelerationPedal * self.MAX_ACCEL * torque_profile
        #
        # # Total resistive forces acting against the tires
        # # SWAP THIS LINE IN YOUR CODE:
        # total_resistance = (self.velocity * ROLLING_RESIST) + (self.velocity * abs(self.velocity) * DRAG_COEFF)
        #
        # self.acceleration = engine_force - total_resistance
        # self.velocity += self.acceleration * time.dt
        #
        # # Precise Braking
        # if self.breakPedal > 0 and abs(self.velocity) > 0.01:
        #     brake_drop = self.breakPedal * self.MAX_BRAKE * time.dt
        #     self.velocity = max(0.0, self.velocity - brake_drop) if self.velocity > 0 else min(0.0,
        #                                                                                        self.velocity + brake_drop)

        # --- 4. DYNAMIC HIGH-SPEED LATERAL GRIP TRACKING (NEW) ---
        yaw = math.radians(self.rotation_y)
        flat_forward = Vec3(math.sin(yaw), 0, math.cos(yaw))

        if abs(self.velocity) > 0.1:
            # Steering authority still tapers at speed (so full-lock doesn't
            # snap you into an instant spin), but the floor is high enough
            # that you always have real turning power to correct a mistake.
            # Old floor (0.15) meant from ~61 mph up you had almost nothing
            # left to fight a drift with — this raises that reserve.
            STEER_FLOOR = 0.40
            speed_grip_factor = max(STEER_FLOOR, 1.0 - (abs(self.velocity) / (self.MAX_SPEED * 1.15)))

            self.rotation_y += math.degrees(steer_radians) * speed_grip_factor * 4.0 * time.dt

            LATERAL_TRACTION = 8.5  # Higher = arcade rails, Lower = ice/drift simulation
            actual_movement_direction = lerp(flat_forward, flat_forward + (
                    Vec3(math.cos(yaw), 0, -math.sin(yaw)) * math.sin(steer_radians)),
                                             LATERAL_TRACTION * time.dt)

            self.position += actual_movement_direction.normalized() * self.velocity * time.dt

        # # --- 1. SMOOTH PEDAL INPUTS ---
        # # If pressing 'w', gas pedal pushes down. If let go, it springs back up.
        # pedal_response_speed = 4.0  # How fast the pedal moves down/up
        #
        # if held_keys['w']:
        #     self.accelerationPedal = min(1.0, self.accelerationPedal + time.dt * pedal_response_speed)
        # else:
        #     self.accelerationPedal = max(0.0, self.accelerationPedal - time.dt * pedal_response_speed)
        #
        # # If pressing 's', brake pedal pushes down.
        # if held_keys['s']:
        #     self.breakPedal = min(1.0, self.breakPedal + time.dt * pedal_response_speed)
        # else:
        #     self.breakPedal = max(0.0, self.breakPedal - time.dt * pedal_response_speed)
        #
        # # --- 2. UPGRADED HIGH-RESPONSIVENESS STEERING ---
        # # Doubled the input speed so the wheel snaps to your commands instantly
        # steer_speed = 750.0  # Fast, highly responsive tracking
        # return_speed = 900.0  # Snaps back to center aggressively when you let go
        #
        # if held_keys['a']:
        #     self.wheelAngle = max(-540.0, self.wheelAngle - steer_speed * time.dt)
        # elif held_keys['d']:
        #     self.wheelAngle = min(540.0, self.wheelAngle + steer_speed * time.dt)
        # else:
        #     if self.wheelAngle > 10:
        #         self.wheelAngle -= return_speed * time.dt
        #     elif self.wheelAngle < -10:
        #         self.wheelAngle += return_speed * time.dt
        #     else:
        #         self.wheelAngle = 0.0
        # # # --- 2. SMOOTH STEERING WHEEL ---
        # # # Simulating a steering wheel spinning back to center automatically
        # # steer_speed = 300.0  # Degrees per second the wheel turns
        # # return_speed = 400.0  # How fast the wheel snaps back to center
        # #
        # # if held_keys['a']:
        # #     self.wheelAngle = max(-540.0, self.wheelAngle - steer_speed * time.dt)
        # # elif held_keys['d']:
        # #     self.wheelAngle = min(540.0, self.wheelAngle + steer_speed * time.dt)
        # # else:
        # #     # No keys pressed? Automatically center the wheel
        # #     if self.wheelAngle > 5:
        # #         self.wheelAngle -= return_speed * time.dt
        # #     elif self.wheelAngle < -5:
        # #         self.wheelAngle += return_speed * time.dt
        # #     else:
        # #         self.wheelAngle = 0.0
        #
        #
        # # 1. Calculate Engine Acceleration as a function of velocity
        # # As velocity approaches MAX_SPEED, engine power efficiency drops to 0
        # speed_ratio = abs(self.velocity) / self.MAX_SPEED
        # available_horsepower = max(0.0, 1.0 - speed_ratio)
        # engine_force = self.accelerationPedal * self.MAX_ACCEL * available_horsepower
        # drag_force = self.velocity * self.FRICTION
        #
        # # Apply basic movement forces
        # self.acceleration = engine_force - drag_force
        # self.velocity += self.acceleration * time.dt
        #
        # # Apply braking cleanly as a reduction tool, preventing the 0mph jitter bug
        # if self.breakPedal > 0 and abs(self.velocity) > 0.01:
        #     brake_drop = self.breakPedal * self.MAX_BRAKE * time.dt
        #     if self.velocity > 0:
        #         self.velocity = max(0.0, self.velocity - brake_drop)
        #     else:
        #         self.velocity = min(0.0, self.velocity + brake_drop)
        #
        # # Movement Execution
        # if abs(self.velocity) > 0.1:
        #     normalized_steer = self.wheelAngle / 540.0
        #
        #     # Increased base maneuverability for micro-corrections
        #     BASE_STEER_POWER = 280.0
        #
        #     # FINE-TUNED SPEED DAMPENER:
        #     # Changes the curve so high-speed tracking drops to a reliable 0.45
        #     # instead of dropping to a useless 0.2. This gives you recovery grip!
        #     speed_ratio = abs(self.velocity) / self.MAX_SPEED
        #     speed_dampener = max(0.28, 1.0 - (speed_ratio ** 2 * 0.82))
        #
        #
        #     direction_modifier = 1 if self.velocity > 0 else -1
        #     self.rotation_y += normalized_steer * BASE_STEER_POWER * speed_dampener * direction_modifier * time.dt


        # 2. MINI GROUND-COLLISION SNAPPER
        # Cast an invisible ray downwards from slightly above the car's center
        # origin = self.world_position + Vec3(0, 1, 0)
        # hit_info = raycast(origin, Vec3(0, -1, 0), distance=10, ignore=(self,))
        #
        # if hit_info.hit:
        #     # If the ray hits your road mesh, snap the car's Y height to the road's surface!
        #     self.y = hit_info.world_point.y + 1
        # else:
        #     # If you drive off the edge of the mesh, gravity pulls you down!
        #     self.y -= 9.8 * time.dt

        # --- 2. MINI GROUND-COLLISION SNAPPER ---

        # Calculate a "flat" forward vector based ONLY on your steering (yaw)
        # This guarantees the rays stay exactly 1.7 units apart horizontally at all times. WHAT A STUPID BUG
        # yaw = math.radians(self.rotation_y)
        # flat_forward = Vec3(math.sin(yaw), 0, math.cos(yaw))
        #
        # # Apply the flat forward vector for the ray origins
        # origin1 = self.world_position + Vec3(0, 1, 0) + (flat_forward * 0.85)
        # origin2 = self.world_position + Vec3(0, 1, 0) - (flat_forward * 0.85)
        #
        # hit_info1 = raycast(origin1, Vec3(0, -1, 0), distance=10, ignore=(self,))
        # hit_info2 = raycast(origin2, Vec3(0, -1, 0), distance=10, ignore=(self,))
        #
        # if hit_info1.hit or hit_info2.hit:
        #     if not hit_info1.hit:
        #         hit_info1 = hit_info2
        #     if not hit_info2.hit:
        #         hit_info2 = hit_info1
        #
        #     # FIX THE FLOATING: Changed "+ 1" to "+ 0.65" (Tweak this to perfectly touch the wheels to the ground)
        #     self.y = (hit_info1.world_point.y + hit_info2.world_point.y) / 2 + 1
        #
        #     # The horizontal distance is permanently 1.7, making this math completely stable now
        #     self.rotation_x = -math.degrees(math.atan2((hit_info1.world_point.y - hit_info2.world_point.y), 1.7))
        #
        # else:
        #     # If you drive off the edge of the mesh, gravity pulls you down!
        #     self.y -= 9.8 * time.dt

        # Generate our flat directionals right before moving
        yaw = math.radians(self.rotation_y)
        flat_forward = Vec3(math.sin(yaw), 0, math.cos(yaw))
        flat_right = Vec3(math.cos(yaw), 0, -math.sin(yaw))

        # Move the car using flat_forward ONLY (using clamped physics dt).
        # This stops the car from physically driving itself under the terrain!
        self.position += flat_forward * self.velocity * dt

        # --- 6. ARCADE SUSPENSION SYSTEM ---
        # Raise origins to +2.0 so they survive sudden, steep drops
        center_high = self.world_position + Vec3(0, 2.0, 0)
        z_offset = 0.55
        x_offset = 0.55

        origin_FL = center_high + (flat_forward * z_offset) - (flat_right * x_offset)
        origin_FR = center_high + (flat_forward * z_offset) + (flat_right * x_offset)
        origin_BL = center_high - (flat_forward * z_offset) - (flat_right * x_offset)
        origin_BR = center_high - (flat_forward * z_offset) + (flat_right * x_offset)

        # YOU MUST IGNORE self.front! It has a box collider that ruins the raycasts.
        ignore_list = (self, self.front)
        RAY_LENGTH = 8.0

        hit_FL = raycast(origin_FL, Vec3(0, -1, 0), distance=RAY_LENGTH, ignore=ignore_list)
        hit_FR = raycast(origin_FR, Vec3(0, -1, 0), distance=RAY_LENGTH, ignore=ignore_list)
        hit_BL = raycast(origin_BL, Vec3(0, -1, 0), distance=RAY_LENGTH, ignore=ignore_list)
        hit_BR = raycast(origin_BR, Vec3(0, -1, 0), distance=RAY_LENGTH, ignore=ignore_list)

        # Expanded MAX_DROP to 5.5 to account for high speeds & steep slopes
        MAX_DROP = 5.5
        ride_height = 0.90
        hanging_y = self.y - 0.2

        valid_FL = hit_FL.hit and hit_FL.distance < MAX_DROP
        valid_FR = hit_FR.hit and hit_FR.distance < MAX_DROP
        valid_BL = hit_BL.hit and hit_BL.distance < MAX_DROP
        valid_BR = hit_BR.hit and hit_BR.distance < MAX_DROP

        y_FL = hit_FL.world_point.y if valid_FL else hanging_y
        y_FR = hit_FR.world_point.y if valid_FR else hanging_y
        y_BL = hit_BL.world_point.y if valid_BL else hanging_y
        y_BR = hit_BR.world_point.y if valid_BR else hanging_y

        grounded_wheels = sum([valid_FL, valid_FR, valid_BL, valid_BR])

        if grounded_wheels > 0:
            valid_ys = [y for y, v in [(y_FL, valid_FL), (y_FR, valid_FR),
                                       (y_BL, valid_BL), (y_BR, valid_BR)] if v]
            target_y = sum(valid_ys) / len(valid_ys) + ride_height

            # --- NEW (fixed) ---
            # Pitch: only calculate if there's real ground data on BOTH front and back
            front_hits = [y_FL if valid_FL else None, y_FR if valid_FR else None]
            back_hits = [y_BL if valid_BL else None, y_BR if valid_BR else None]
            front_hits = [y for y in front_hits if y is not None]
            back_hits = [y for y in back_hits if y is not None]

            if front_hits and back_hits:
                front_y = sum(front_hits) / len(front_hits)
                back_y = sum(back_hits) / len(back_hits)
                target_pitch = -math.degrees(math.atan2((front_y - back_y), z_offset * 2))
            else:
                target_pitch = self.rotation_x  # Hold current pitch — not enough data

            # Roll: only calculate if there's real ground data on BOTH left and right
            left_hits = [y_FL if valid_FL else None, y_BL if valid_BL else None]
            right_hits = [y_FR if valid_FR else None, y_BR if valid_BR else None]
            left_hits = [y for y in left_hits if y is not None]
            right_hits = [y for y in right_hits if y is not None]

            if left_hits and right_hits:
                left_y = sum(left_hits) / len(left_hits)
                right_y = sum(right_hits) / len(right_hits)
                target_roll = math.degrees(math.atan2((left_y - right_y), x_offset * 2))
            else:
                target_roll = self.rotation_z  # Hold current roll — not enough data
            # Stiffer springs (25) to prevent the car from sagging under gravity at high speeds
            spring_stiffness = 25 * time.dt

            self.y = lerp(self.y, target_y, spring_stiffness)
            self.rotation_x = lerp(self.rotation_x, target_pitch, spring_stiffness)
            self.rotation_z = lerp(self.rotation_z, target_roll, spring_stiffness)

        else:
            # Airborne gravity
            self.y -= 9.8 * time.dt
            self.rotation_x = lerp(self.rotation_x, 0, 2 * time.dt)
            self.rotation_z = lerp(self.rotation_z, 0, 2 * time.dt)


        self.Speedometer.text = ("Speed: " + str(int(self.velocity)) + "miles/hr")

        # Out-of-bounds safety recovery: If car falls off track edge or drops below Y=-15, reset back to track center
        if self.y < -15.0:
            self.position = Vec3(0, 5.0, self.position.z)
            self.velocity = 0.0
            self.wheelAngle = 0.0
            self.rotation_x = 0
            self.rotation_z = 0
            print("CAR OUT OF BOUNDS: Safely reset vehicle position back to track center!")

    def returnBreakPedal(self):
        return self.breakPedal
    def returnAccelerationPedal(self):
        return self.accelerationPedal
    def returnSteeringAngle(self):
        return self.wheelAngle
    def returnVelocity(self):
        return self.velocity



#oncoming car is pretty much a stripped down version of the normal car, with the added logic of auto following the lane
# based on the generated road data
class oncomingCar(Entity):
    def __init__(self, carColor, chunkINDX, start, speed, **kwargs):
        super().__init__(model = 'cube', color=carColor, scale=(2,3/2,2), collider='box', **kwargs)

        self.bodyBorder = Entity(model='cube', color=color.black, scale=1.001, parent=self, wireframe=True)

        self.front = Entity(model='cube', color=carColor, scale=(1,1.25/3,1/3), collider='box', parent=self, position=Vec3(0,-.5 + 1.25/6,.5 + 1/6))
        self.frontBorder = Entity(model='cube', color=color.black, scale=1.001, parent=self.front, wireframe=True)


        wheel_scale = (0.3, 0.3, 0.3)
        wheel_rotation = Vec3(0, 0, 90)

        # FIX: Changed 'sides' to 'resolution', removed 'start=0'
        procedural_wheel1 = Cylinder(resolution=16, height=1, radius=0.5, direction=(0, 1, 0))



        # Front Right
        self.wheel1 = Entity(model=copy(procedural_wheel1), color=color.black, parent=self,
                             scale=wheel_scale, rotation=wheel_rotation,
                             position=Vec3(0.55-.3, -0.5, 0.4 + .15))

        # Front Left
        self.wheel2 = Entity(model=copy(procedural_wheel1), color=color.black, parent=self,
                             scale=wheel_scale, rotation=wheel_rotation,
                             position=Vec3(-0.55, -0.5, 0.4 + .15))

        # Back Right
        self.wheel3 = Entity(model=copy(procedural_wheel1), color=color.black, parent=self,
                             scale=wheel_scale, rotation=wheel_rotation,
                             position=Vec3(0.55-.3, -0.5, -0.4))

        # Back Left
        self.wheel4 = Entity(model=copy(procedural_wheel1), color=color.black, parent=self,
                             scale=wheel_scale, rotation=wheel_rotation,
                             position=Vec3(-0.55, -0.5, -0.4))
        self.chunkINDX = chunkINDX
        self.t = start
        self.speed = speed

    def shift_index(self):
        self.chunkINDX -= 1

    def update(self):
        self.t -= self.speed * time.dt

        # move chunk closer to player
        if self.t < 0.0:
            if self.world_gen and self.chunkINDX in self.world_gen.worldObjects and self in self.world_gen.worldObjects[self.chunkINDX]:
                self.world_gen.worldObjects[self.chunkINDX].remove(self)

            self.chunkINDX -= 1
            self.t = 1.0  # reset

            if self.world_gen:
                if self.chunkINDX not in self.world_gen.worldObjects:
                    self.world_gen.worldObjects[self.chunkINDX] = []
                self.world_gen.worldObjects[self.chunkINDX].append(self)

        if not self.world_gen:
            return

        trackPoints = self.world_gen.trackPoints
        if self.chunkINDX - 1 < 0 or self.chunkINDX + 2 >= len(trackPoints):
            return

        p0 = trackPoints[self.chunkINDX - 1]
        p1 = trackPoints[self.chunkINDX]
        p2 = trackPoints[self.chunkINDX + 1]
        p3 = trackPoints[self.chunkINDX + 2]

        road_center = spline(p0, p1, p2, p3, self.t)
        road_forward = splineTan(p0, p1, p2, p3, self.t)

        road_right = np.cross(road_forward, np.array(Vec3(0, 1, 0)))
        norm = np.linalg.norm(road_right)
        if norm != 0:
            road_right /= norm

        laneOffset = road_right * self.world_gen.globalRoadWidth / 4
        self.position = Vec3(road_center[0], road_center[1], road_center[2]) - Vec3(-laneOffset[0], laneOffset[1], laneOffset[2]) + Vec3(0, 1, 0)

        self.look_at(self.position - Vec3(road_forward[0], road_forward[1], road_forward[2]))
        self.rotation_x = 0
        self.rotation_z = 0


# --- MAIN WORLD GENERATOR CLASS --- GEMINI REWROTE VERSION TO CLEAN UP ALL MY COMMENTED OUT BROKEN CODE AND STUFF AND BETTER ORGANIZE EVERYTHING
# AKA IF SOMETHING BREAKS, ITS SOME BS BY GEMINI
class WorldGenerator(Entity):
    def __init__(self, worldSeed=None, stopSignDensity=1, onExitToMenu=None, **kwargs):
        super().__init__(**kwargs)

        # Set by the menu so the sim can hand control back. None while evaluating, which
        # is also what marks a world as "not interactively exitable".
        self.onExitToMenu = onExitToMenu

        # Evaluation sets this to lock out inputs that would corrupt a scored run.
        self.evaluationLocked = False

        # Stop signs spawn on a 1-in-100 roll per spline sample, which yields ~4 signs in
        # a 240s evaluation -- far too few to tell a real change from noise. Evaluation
        # can raise this to widen the roll (4 => 1-in-25) and get a usable sample.
        # Training and normal play leave it at 1.
        self.stopSignDensity = max(1, int(stopSignDensity))

        # Spline samples sit ~25 units apart, so back-to-back rolls can put two stop
        # signs almost on top of each other. That is a scenario the recorded data
        # essentially never contains, and it reads to the model as one very close sign
        # (two signs = more red pixels = looks nearer), so it brakes far too early and
        # then cannot tell which sign it has satisfied. Enforced spacing keeps a dense
        # course realistic instead of pathological.
        self.lastStopSignPos = None

        # Course layout draws from its OWN generator when a seed is given.
        # Seeding the global `random` before construction is not enough: WeatherSystem
        # pulls from the global stream every frame for particle positions, so by the
        # time updateTrack() builds chunk 6 the global stream has advanced by thousands
        # of frame-rate-dependent draws. With a dedicated generator, chunk k always gets
        # the same numbers no matter when it is built or how fast the car got there --
        # which is what makes two evaluation runs comparable.
        self.worldRng = random.Random(worldSeed) if worldSeed is not None else random

        # Evaluation turns this off: weather tints and particles change what the camera
        # sees, so it has to be identical across models being compared.
        self.weatherEnabled = True

        self.sim_panel = None       # pause/settings overlay
        self.worldObjects = {}
        self.currentView = 'chase'
        self.tempFileNum = 0

        self.car_colors = [
            color.black, color.white, color.gray, color.light_gray,
            color.red, color.blue, color.orange, color.yellow, color.gold,
            color.hex("800000"), color.hex("F5F5DC"), color.hex("00008b"), color.hex("006400")
        ]

        self.speedLimit = self.worldRng.randrange(25,55, 5)
        self.globalRoadWidth = 10

        #default track start
        self.trackPoints = [
            Vec3(0, 0, -200),
            Vec3(0, 0, 0),
            Vec3(0, 0, 200),
            Vec3(0, 0, 400),
            Vec3(20, 4, 600),
            Vec3(-20, -2, 800),
            Vec3(0, 0, 1000),
            Vec3(0, 0, 1200)
        ]

        verts, tris, cols = generateRoad(self.trackPoints, roadWidth=self.globalRoadWidth, grassWidth=30, resolution=12)
        road_mesh = Mesh(vertices=verts, triangles=tris, colors=cols)
        self.road_entity = Entity(model=road_mesh, collider='mesh', double_sided=True, parent=self)

        for initial_chunk in range(1, len(self.trackPoints) - 2):
            self.spawnObstaclesForSegment(
                self.trackPoints[initial_chunk - 1], self.trackPoints[initial_chunk],
                self.trackPoints[initial_chunk + 1], self.trackPoints[initial_chunk + 2],
                initial_chunk, roadWidth=10, resolution=12
            )

        #initilize all the starting entities (car and weather system)
        self.car = Car(carColor=color.orange, position=Vec3(0, 10, 0), parent=self)

        self.weather = WeatherSystem(target=self.car, parent=self)
        self.weather.set_weather('clear')

        self.weatherChance = 1
        self.timer = 0.0
        self.weatherUpdateInterval = 10.0

        # Always go through setCameraView so fov is set explicitly. The old code set the
        # parent, position and rotation but left fov alone, so a view entered after the
        # windshield camera (fov 100) or the free-fly editor kept the wrong fov and
        # looked zoomed out.
        self.setCameraView('chase')

        #note the offscreen cameras for data collections
        self.camera_mgr = DataCollection.OffscreenCameraManager(resolution=(512, 512))
        self.camera_mgr.register_camera(self.car.windshieldLeft)
        self.camera_mgr.register_camera(self.car.windshieldCentered)
        self.camera_mgr.register_camera(self.car.windshieldRight)

        self.logger = DataCollection.DatasetLogger(worldGen=self, delay=.1) #actually make the data logger entity

        #default settings
        self.autopilot_enabled = False
        self.autopilot_model = None
        self.selected_model_path = None
        self.model_menu_panel = None
        self.model_menu_page = 0
        self.autopilot_timer = 0.0
        self.autopilot_interval = 0.05  # 50ms interval (20 Hz perception loop)
        self.last_sign_salience = 0      # red pixels in the last frame fed to the model
        self.creep_hold_timer = 0.0     # how long creep suppression has been braking
        self.creep_armed = False        # only after the car has actually driven
        self.creep_released = False     # sticky, so the hold cannot re-arm and crawl
        try:
            import torch
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(str(self.device))
        except Exception:
            self.device = None

    CAMERA_VIEWS = ("chase", "windshield")

    # Matches the world generator's own roll, worldRng.randrange(25, 60, 5), so a manual
    # choice is always a limit the model has actually seen signs for in training.
    SPEED_LIMIT_CHOICES = (25, 30, 35, 40, 45, 50, 55)

    #toggles the cameras based on the required view
    def setCameraView(self, view):
        """Only two views now -- chase (default) and windshield.

        The free-fly EditorCamera was removed: it detached the camera from the car and
        changed fov, and every other view had to remember to undo that.
        """
        if view not in self.CAMERA_VIEWS:
            view = "chase"
        self.currentView = view

        if view == "chase":
            camera.parent = self.car.thirdPerson
            camera.fov = 60
        else:
            camera.parent = self.car.windshieldCentered
            camera.fov = 100

        camera.position = Vec3(0, 0, 0)
        camera.rotation = Vec3(0, 0, 0)
        return view

    # ------------------------------------------------------------------ pause panel

    def closeSimPanel(self):
        if getattr(self, "sim_panel", None):
            destroy_all(self.sim_panel)
            self.sim_panel = None

    def toggleSimPanel(self):
        """Weather, camera and exit, moved off individual key bindings.

        Rarely used controls belong in UI rather than on the number row -- this is the
        first batch, and the panel is the natural place to add more.
        """
        if getattr(self, "sim_panel", None):
            self.closeSimPanel()
            return

        self.sim_panel = Entity(parent=camera.ui, z=-10)
        Entity(parent=self.sim_panel, model='quad', color=color.rgba32(15, 15, 25, 232),
               scale=(0.72, 0.80), z=1)
        Text(parent=self.sim_panel, text="Simulation", position=(-0.33, 0.34),
             scale=1.5, color=color.cyan)
        Text(parent=self.sim_panel, text="esc closes this", position=(-0.33, 0.305),
             scale=0.75, color=color.gray)

        Text(parent=self.sim_panel, text="Weather", position=(-0.33, 0.245),
             scale=1.0, color=color.light_gray)
        for i, mode in enumerate(("clear", "rain", "snow", "sandstorm")):
            active = getattr(self.weather, "mode", "clear") == mode
            b = Button(text=mode, parent=self.sim_panel, scale=(0.155, 0.07),
                       position=(-0.255 + i * 0.17, 0.18),
                       color=color.azure if active else color.dark_gray,
                       text_color=color.white)

            def pickWeather(m=mode):
                self.weather.set_weather(m)
                # a manual choice should stick, not get overwritten by the random timer
                self.weatherEnabled = False
                self.closeSimPanel()
                self.toggleSimPanel()
            b.on_click = pickWeather

        Text(parent=self.sim_panel, text="Random weather changes", position=(-0.33, 0.10),
             scale=0.9, color=color.light_gray)
        # Sits immediately after the label rather than off at the panel edge, where it
        # read as belonging to the sandstorm button above it.
        b = Button(text="on" if self.weatherEnabled else "off", parent=self.sim_panel,
                   scale=(0.1, 0.055), position=(0.055, 0.103),
                   color=color.azure if self.weatherEnabled else color.dark_gray,
                   text_color=color.white)

        def toggleRandomWeather():
            self.weatherEnabled = not self.weatherEnabled
            self.closeSimPanel()
            self.toggleSimPanel()
        b.on_click = toggleRandomWeather

        # Speed limit. The model takes this as a direct input alongside speed, so being
        # able to set it by hand is the only way to see how it reacts to a given limit
        # without driving around waiting for the right sign to spawn.
        Text(parent=self.sim_panel, text=f"Speed limit    {self.speedLimit} MPH",
             position=(-0.33, 0.035), scale=0.9, color=color.light_gray)
        for i, limit in enumerate(self.SPEED_LIMIT_CHOICES):
            active = self.speedLimit == limit
            b = Button(text=str(limit), parent=self.sim_panel, scale=(0.083, 0.06),
                       position=(-0.28 + i * 0.093, -0.03),
                       color=color.azure if active else color.dark_gray,
                       text_color=color.white)

            def pickLimit(v=limit):
                self.speedLimit = v
                print(f"SPEED LIMIT: manually set to {v} MPH")
                self.closeSimPanel()
                self.toggleSimPanel()
            b.on_click = pickLimit

        Text(parent=self.sim_panel, text="(the next speed sign you pass will override this)",
             position=(-0.33, -0.075), scale=0.7, color=color.gray)

        Text(parent=self.sim_panel, text="Camera", position=(-0.33, -0.135),
             scale=1.0, color=color.light_gray)
        for i, view in enumerate(self.CAMERA_VIEWS):
            active = self.currentView == view
            b = Button(text=view, parent=self.sim_panel, scale=(0.17, 0.07),
                       position=(-0.245 + i * 0.19, -0.20),
                       color=color.azure if active else color.dark_gray,
                       text_color=color.white)

            def pickView(v=view):
                self.setCameraView(v)
                self.closeSimPanel()
                self.toggleSimPanel()
            b.on_click = pickView

        b = Button(text="Return to Menu", parent=self.sim_panel, scale=(0.34, 0.08),
                   position=(-0.16, -0.32), color=color.red, text_color=color.black)

        def leave():
            self.closeSimPanel()
            if self.onExitToMenu:
                self.onExitToMenu()
            else:
                print("SIM: no menu to return to (started outside the menu)")
        b.on_click = leave

        b = Button(text="Close", parent=self.sim_panel, scale=(0.18, 0.08),
                   position=(0.22, -0.32), color=color.dark_gray, text_color=color.white)
        b.on_click = self.closeSimPanel

    def teardown(self):
        """Fully dispose of this world.

        Needed because several pieces outlive `destroy(self)`: the car's Speedometer,
        SpeedLimitText and statusCircle are parented to camera.ui rather than to the
        world, the DatasetLogger owns a background thread, and the offscreen cameras
        hold GPU framebuffers. The evaluator rebuilds the world between runs, so all of
        that has to actually go away.
        """
        self.autopilot_enabled = False

        # detach the camera before the car is destroyed
        try:
            camera.parent = scene
            camera.position = Vec3(0, 0, 0)
            camera.rotation = Vec3(0, 0, 0)
            camera.fov = 60
        except Exception:
            pass

        try:
            self.closeSimPanel()
        except Exception:
            pass

        try:
            if getattr(self, "logger", None):
                self.logger.shutdown()
                destroy_all(self.logger)
        except Exception:
            pass

        try:
            if getattr(self, "camera_mgr", None):
                self.camera_mgr.cleanup()
        except Exception:
            pass

        # camera.ui-parented HUD pieces the world does not own
        for attr in ("Speedometer", "SpeedLimitText", "statusCircle",
                     "autopilotCircle", "autopilotLabel"):
            try:
                widget = getattr(getattr(self, "car", None), attr, None)
                if widget:
                    destroy(widget)
            except Exception:
                pass

        # Same problem, but hanging off the world instead of the car: the CORRECTIONS
        # ONLY banner is built lazily the first time 8 is pressed and parented to
        # camera.ui, so destroy_all(self) below never walks to it and it was surviving
        # the trip back to the menu.
        try:
            if getattr(self, "correction_text", None):
                destroy(self.correction_text)
                self.correction_text = None
        except Exception:
            pass

        try:
            if getattr(self, "model_menu_panel", None):
                destroy(self.model_menu_panel)
                self.model_menu_panel = None
        except Exception:
            pass

        try:
            destroy_all(self)
        except Exception:
            pass

    #returns the speed limit (shocking stuff)
    def returnSpeedLimit(self):
        return self.speedLimit

    #takes the track information, and runs a bunch of random numbers to find if we should spawn a pedestrian, stop sign, or speed limit sign
    #puts all entities into a map for later deletion
    #oh, also summons cars
    def spawnObstaclesForSegment(self, p0, p1, p2, p3, chunk_idx, roadWidth, resolution):
        GLOBAL_UP = Vec3(0, 1, 0)
        if chunk_idx not in self.worldObjects:
            self.worldObjects[chunk_idx] = []

        for j in range(0, resolution + 1):
            t = j / resolution
            center = spline(p0, p1, p2, p3, t)
            forward = splineTan(p0, p1, p2, p3, t)
            rawRightV = np.cross(forward, GLOBAL_UP)
            right = rawRightV / sqrt(rawRightV @ rawRightV)

            signChance = self.worldRng.randint(1, 100)
            isStopSign = signChance <= self.stopSignDensity
            if isStopSign:
                signPosition = Vec3(center[0], center[1], center[2]) + (
                        Vec3(-right[0], right[1], right[2]) * (roadWidth / 2 + 1.5))

                if self.lastStopSignPos is not None:
                    gap = math.dist((signPosition.x, signPosition.z),
                                    (self.lastStopSignPos.x, self.lastStopSignPos.z))
                    if gap < MIN_STOP_SIGN_SPACING:
                        isStopSign = False   # falls through to the pedestrian/speed-sign rolls

            if isStopSign:
                road_direction = Vec3(forward[0], forward[1], forward[2])
                new_sign = StopSign(position=signPosition, road_direction=road_direction, roadWidth=roadWidth, parent=self)
                self.worldObjects[chunk_idx].append(new_sign)
                self.lastStopSignPos = signPosition

            pedestrianChance = self.worldRng.randint(1, 100)
            if pedestrianChance == 1 and not isStopSign:
                randPosition = self.worldRng.uniform(-1.5, .5)
                pedestrianPosition = Vec3(center[0], center[1], center[2]) + (
                        Vec3(-right[0], right[1], right[2]) * (roadWidth / 2) + Vec3(randPosition, 0, 0))
                pedestrian = Pedestrian(position=pedestrianPosition, parent=self)
                self.worldObjects[chunk_idx].append(pedestrian)

            speedSignChance = self.worldRng.randint(1, 400)
            if pedestrianChance != 1 and not isStopSign and speedSignChance == 1:
                signPosition = Vec3(center[0], center[1], center[2]) + (
                        Vec3(-right[0], right[1], right[2]) * (roadWidth / 2 + 1.5))
                road_direction = Vec3(forward[0], forward[1], forward[2])
                new_sign = speedLimitSign(position=signPosition, road_direction=road_direction,
                                          roadWidth=roadWidth, rng=self.worldRng, parent=self)
                self.worldObjects[chunk_idx].append(new_sign)

        if self.worldRng.randint(1, 5) == 1:
            new_npc = oncomingCar(carColor=self.car_colors[self.worldRng.randint(0, len(self.car_colors) - 1)],
                                 chunkINDX=chunk_idx, start=1.0, speed=self.worldRng.uniform(0.1, 0.2), parent=self)
            new_npc.world_gen = self
            self.worldObjects[chunk_idx].append(new_npc)

    #delete old chunks and generate new ones, kind of acting like the controller
    def updateTrack(self):
        if self.car.position.z > self.trackPoints[2].z:
            chunks_to_delete = [k for k in self.worldObjects.keys() if k <= 1]
            for k in chunks_to_delete:
                for obj in self.worldObjects[k]:
                    destroy_all(obj)
                del self.worldObjects[k]

            for chunk_idx in list(self.worldObjects.keys()):
                for obj in self.worldObjects[chunk_idx]:
                    if isinstance(obj, oncomingCar):
                        obj.shift_index()

            new_objects_dict = {}
            for chunk_idx, obj_list in self.worldObjects.items():
                new_objects_dict[chunk_idx - 1] = obj_list
            self.worldObjects = new_objects_dict

            self.trackPoints.pop(0)
            lastPoint = self.trackPoints[len(self.trackPoints) - 1]
            newHrznPt = nextMilestone(lastPoint, self.worldRng)
            self.trackPoints.append(newHrznPt)

            destroy(self.road_entity)
            verts, tris, cols = generateRoad(self.trackPoints, roadWidth=10, grassWidth=30, resolution=12)
            new_mesh = Mesh(vertices=verts, triangles=tris, colors=cols)
            self.road_entity = Entity(model=new_mesh, collider='mesh', double_sided=True, parent=self)

            last_idx = len(self.trackPoints) - 3
            self.spawnObstaclesForSegment(
                self.trackPoints[last_idx - 1], self.trackPoints[last_idx],
                self.trackPoints[last_idx + 1], self.trackPoints[last_idx + 2],
                last_idx, roadWidth=10, resolution=12
            )

    #set the weather conditions by input
    def weatherAssist(self):
        self.weatherChance = random.randint(1, 4)
        if self.weatherChance == 1:
            self.weather.set_weather('clear')
        elif self.weatherChance == 2:
            self.weather.set_weather('rain')
        elif self.weatherChance == 3:
            self.weather.set_weather('snow')
        elif self.weatherChance == 4:
            self.weather.set_weather('sandstorm')
        print("weatherAssist " + str(self.weatherChance) + '\n')

    #grab image from one of the offscreen cameras and return the image
    def get_camera_image(self, cam_entity):
        """
        Pass any registered camera entity (e.g. self.car.windshieldCentered)
        and get back a PIL Image object with true colors and active weather tint.
        """
        from PIL import Image
        img = self.camera_mgr.get_image(cam_entity)
        if img is None:
            return None
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        if self.weather and self.weather.mode != 'clear' and self.weather.overlay_color:
            tint_img = Image.new('RGBA', img.size, self.weather.overlay_color)
            img = Image.alpha_composite(img, tint_img)
        return img

    #extra status lights
    def set_status_light(self, state):
        """Sets status indicator light circle to green (True/'green') or red (False/'red')."""
        self.car.set_status_light(state)

    def set_status_green(self):
        self.car.set_status_green()

    def set_status_red(self):
        self.car.set_status_red()

    #trying to load a specific model selected
    def load_specified_model(self, model_path):
        """Loads a specific model path into self.autopilot_model and sets self.selected_model_path."""
        if not model_path.exists():
            print(f"MODEL ERROR: File {model_path} does not exist!")
            return False

        try:
            os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
            import torch
            torch_lib = os.path.join(os.path.dirname(torch.__file__), 'lib')
            if os.path.exists(torch_lib) and hasattr(os, 'add_dll_directory'):
                os.add_dll_directory(torch_lib)

            from Training import modelForCheckpoint
            # Builds a Net matching the lin1 width this checkpoint was saved with, so
            # models trained at different capacities all stay loadable. Without it a
            # wide checkpoint would load into a narrow Net under strict=False, keep
            # random weights for ~96% of the model, and drive like noise with no error.
            state = torch.load(model_path, map_location=self.device, weights_only=True)
            self.autopilot_model, hiddenWidth = modelForCheckpoint(state, self.device)
            # strict=False so checkpoints saved before a layer existed still load.
            missing, unexpected = self.autopilot_model.load_state_dict(state, strict=False)
            if missing:
                print(f"MODEL NOTE: {model_path.name} predates {list(missing)} "
                      f"-- left at init (unused at inference)")
            if unexpected:
                print(f"MODEL NOTE: {model_path.name} contains unused {list(unexpected)}")
            self.autopilot_model.eval()
            self.selected_model_path = model_path
            print(f"MODEL SELECTED: Active model set to {model_path.name}")
            return True
        except Exception as e:
            print(f"MODEL ERROR loading {model_path.name}: {e}")
            return False

    #show the models to pick from
    def prompt_model_selection(self, page=None):
        """Displays a scrollable/paginated in-game selection menu listing all saved models in Models/."""
        models_dir = Path(__file__).parent / "Models"
        if not models_dir.exists():
            print("MODEL SELECT ERROR: 'Models' directory does not exist!")
            return

        pth_files = sorted(list(models_dir.glob("*.pth")), key=lambda f: f.stat().st_mtime, reverse=True)
        if not pth_files:
            print("MODEL SELECT ERROR: No saved .pth models found in Models directory!")
            return

        PAGE_SIZE = 5
        total_pages = max(1, math.ceil(len(pth_files) / PAGE_SIZE))

        if page is not None:
            self.model_menu_page = page
        elif not hasattr(self, 'model_menu_page'):
            self.model_menu_page = 0

        self.model_menu_page = max(0, min(self.model_menu_page, total_pages - 1))

        # Destroy existing menu if already open
        if hasattr(self, 'model_menu_panel') and self.model_menu_panel:
            destroy(self.model_menu_panel)

        self.model_menu_panel = Entity(parent=camera.ui, z=-10)

        # Background overlay box
        bg = Entity(
            parent=self.model_menu_panel,
            model='quad',
            color=color.rgba32(20, 20, 30, 235),
            scale=(1.15, 0.88),
            z=1
        )

        title = Text(
            text=f"Select Model (Page {self.model_menu_page + 1} of {total_pages}):",
            parent=self.model_menu_panel,
            position=(-0.47, 0.36),
            scale=1.4,
            color=color.cyan
        )

        start_idx = self.model_menu_page * PAGE_SIZE
        displayed_files = pth_files[start_idx : start_idx + PAGE_SIZE]

        for i, model_path in enumerate(displayed_files):
            btn_text = f"{model_path.name}"
            is_active = (self.selected_model_path and self.selected_model_path == model_path)
            if is_active:
                btn_text += " [ACTIVE]"

            b = Button(
                text=btn_text,
                parent=self.model_menu_panel,
                scale=(0.95, 0.07),
                position=(0, 0.22 - (i * 0.09)),
                color=color.azure if is_active else color.dark_gray,
                text_color=color.white
            )

            def make_handler(path=model_path):
                def select_and_close():
                    success = self.load_specified_model(path)
                    if success and self.autopilot_enabled:
                        print(f"AUTOPILOT: Updated active driving model to {path.name}")
                    if hasattr(self, 'model_menu_panel') and self.model_menu_panel:
                        destroy(self.model_menu_panel)
                        self.model_menu_panel = None
                return select_and_close

            b.on_click = make_handler(model_path)

        # Navigation Buttons (Prev / Next)
        if self.model_menu_page > 0:
            prev_b = Button(
                text="▲ Prev Page",
                parent=self.model_menu_panel,
                scale=(0.26, 0.06),
                position=(-0.33, -0.33),
                color=color.gray,
                text_color=color.white
            )
            prev_b.on_click = lambda: self.prompt_model_selection(page=self.model_menu_page - 1)

        if self.model_menu_page < total_pages - 1:
            next_b = Button(
                text="▼ Next Page",
                parent=self.model_menu_panel,
                scale=(0.26, 0.06),
                position=(0.33, -0.33),
                color=color.gray,
                text_color=color.white
            )
            next_b.on_click = lambda: self.prompt_model_selection(page=self.model_menu_page + 1)

        # Close / Cancel Button
        close_b = Button(
            text="Close / Cancel",
            parent=self.model_menu_panel,
            scale=(0.28, 0.06),
            position=(0, -0.33),
            color=color.red,
            text_color=color.white
        )

        def close_menu():
            if hasattr(self, 'model_menu_panel') and self.model_menu_panel:
                destroy(self.model_menu_panel)
                self.model_menu_panel = None

        close_b.on_click = close_menu

    #attempt to turn on and off the models
    def toggle_autopilot(self):
        """Toggles autonomous AI driving mode using the currently selected model (or latest)."""
        if self.autopilot_enabled:
            self.autopilot_enabled = False
            self.set_status_red()
            print("AUTOPILOT: Deactivated! Returned to manual driver controls.")
            return

        # If no model is explicitly selected yet, pick the newest model
        if not self.selected_model_path or not self.selected_model_path.exists():
            models_dir = Path(__file__).parent / "Models"
            if not models_dir.exists():
                print("AUTOPILOT ERROR: 'Models' directory does not exist!")
                return

            pth_files = list(models_dir.glob("*.pth"))
            if not pth_files:
                print("AUTOPILOT ERROR: No trained .pth models found in Models directory!")
                return

            self.selected_model_path = max(pth_files, key=lambda f: f.stat().st_mtime)

        if not self.autopilot_model:
            success = self.load_specified_model(self.selected_model_path)
            if not success:
                self.autopilot_enabled = False
                return

        self.autopilot_enabled = True
        print(f"AUTOPILOT: Activated! Running model: {self.selected_model_path.name}")

    #run model with current frame and set instructions to the car
    #ai added this creep feature for one of the earlier models and it seems to work fine without it but I'm thinking just
    #leave it in idk
    def run_autopilot_step(self):
        """Performs real-time neural network inference and applies AI control predictions to vehicle physics."""
        try:
            import cv2
            import numpy as np
            import torch

            pil_img = self.get_camera_image(self.car.windshieldCentered)
            if pil_img is None:
                return

            #I can't believe I spent the last 2 days wondering why the models weren't working for this bs
            # MUST be RGB, not BGR. DrivingDataset.__getitem__ feeds preprocess() an RGB
            # array (cv2.imread gives BGR, then it converts BGR2RGB), so serving BGR here
            # swapped the red and blue channels relative to training. Steering barely
            # noticed -- grey road, green grass and white lines are unaffected -- and
            # lime is symmetric under R<->B so resuming still worked. But a red stop sign
            # arrived as blue, costing ~26% of the braking response near signs.
            img_np = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGBA2RGB)
            processed_img = DataCollection.DrivingDataset.preprocess(img_np)
            image_tensor = torch.from_numpy(processed_img.transpose(2, 0, 1)).float().unsqueeze(0).to(self.device)

            # How visible an active stop sign is in the exact frame the net just saw.
            # Purely for the evaluator -- it distinguishes "never saw the sign" from "saw
            # it and did not act", which need opposite fixes. signSalience expects BGR
            # channel order, and img_np is RGB, hence the reversal.
            self.last_sign_salience = DataCollection.signSalience(img_np[:, :, ::-1])

            speed_tensor = torch.tensor([[float((self.car.returnVelocity()))]], dtype=torch.float32).to(self.device)
            speed_limit_tensor = torch.tensor([[float((self.returnSpeedLimit()))]], dtype=torch.float32).to(self.device)

            with torch.no_grad():
                outputs = self.autopilot_model(image_tensor, speed_tensor, speed_limit_tensor)
                pred_steering = outputs[0, 0].item()
                pred_pedal = outputs[0, 1].item()

            # --- creep suppression -------------------------------------------
            # A stop sign only registers on velocity == 0 EXACTLY, and velocity only
            # snaps to zero when effective_pedal == 0 and |v| < 0.05. Any small positive
            # prediction keeps the engine producing force, so the car crawls forever and
            # the sign never turns green. Measured: 3 of 4 signs got under 2 mph, only 2
            # became full stops. Zeroing the pedal is not enough on its own -- drag alone
            # takes ~45s to bleed off the last 2 mph -- so this commits to the brake.
            #
            # The gate is safe because low-speed predictions are strongly bimodal:
            # ~+0.52 when it should pull away (no sign) versus ~-0.30 when it should hold
            # (sign visible). 0.05 sits in the empty middle, so resuming is untouched.
            # Set CREEP_BRAKE = None to disable.
            # CREEP_MAX_HOLD is a release valve, and it is not optional. Holding the
            # brake freezes the car, which freezes the camera, which freezes the
            # prediction -- so a model that stops somewhere it keeps reading as "brake"
            # (e.g. past the sign, still seeing the pole) stays pinned forever. Observed
            # in testing. After this many seconds we hand control back so it can creep
            # out and change its own view.
            # CREEP_SPEED must stay LOW. At 2.0 this fired while the car was still
            # rolling toward the line and pinned it ~21 units out, short of the trigger
            # zone. The human holds ~10 mph until roughly 25 units and only then brakes
            # hard, coming to rest 7-12 units away -- so anything that intervenes above a
            # genuine crawl is cutting the approach short.
            # CREEP_GATE must stay BELOW zero. At +0.05 it was chosen when the gap
            # between "wants to hold" and "wants to go" was empty, but a model trained on
            # sharper stop data brakes harder and resumes more weakly, and 21% of its
            # resume predictions fell under +0.05 -- so the gate braked the car exactly
            # when it was trying to pull away. Firing only on an already-negative
            # prediction means this can never fight a resume, and costs nothing on
            # stopping, since at-rest-with-a-red-sign predictions are 100% negative.
            CREEP_SPEED = 0.8
            CREEP_GATE = -0.05
            CREEP_BRAKE = -0.4
            CREEP_MAX_HOLD = 4.0

            speedNow = abs(self.car.returnVelocity())

            # Arm only once the car has actually driven. Otherwise a run that starts
            # stationary within sight of a sign is pinned before it ever moves.
            if speedNow >= CREEP_SPEED:
                self.creep_armed = True
                self.creep_hold_timer = 0.0
                self.creep_released = False

            holding = (CREEP_BRAKE is not None and self.creep_armed
                       and speedNow < CREEP_SPEED and pred_pedal < CREEP_GATE)

            if holding and not self.creep_released:
                self.creep_hold_timer += self.autopilot_interval
                if self.creep_hold_timer > CREEP_MAX_HOLD:
                    # Release is STICKY until the car is properly moving again -- a
                    # re-arming gate would just brake, release, brake, release and crawl.
                    self.creep_released = True
                else:
                    pred_pedal = CREEP_BRAKE

            if self.creep_released and speedNow < CREEP_SPEED:
                pred_pedal = max(pred_pedal, 0.12)

            # --- human override, per axis ------------------------------------
            # Car.update() writes the pedals and wheel from held_keys every frame, and
            # this writes them again at 20 Hz, so the two used to fight and the model
            # usually won -- which is why manual input only "slightly" worked. Skipping
            # the axis the human is actually holding makes the override absolute, and
            # per-axis means you can correct the brake while the model keeps steering.
            humanSteering = held_keys['a'] or held_keys['d']
            humanPedal = held_keys['w'] or held_keys['s']
            self.human_override = bool(humanSteering or humanPedal)

            if not humanSteering:
                self.car.wheelAngle = max(-540.0, min(540.0, pred_steering * 540.0))

            if not humanPedal:
                if pred_pedal >= 0:
                    self.car.accelerationPedal = max(0.0, min(1.0, pred_pedal))
                    self.car.breakPedal = 0.0
                else:
                    self.car.breakPedal = max(0.0, min(1.0, abs(pred_pedal)))
                    self.car.accelerationPedal = 0.0
        except Exception as e:
            print(f"AUTOPILOT INFERENCE ERROR: {e}")

    #update function, again, rewritten by gemini, so, if it breaks something, NOT BY FAULT
    #should just call the model when it needs to be run, run weather updates for random weather
    #handle some bounding box logic with stop signs and speed limit signs
    def update(self):
        if not self.enabled:
            return

        self.updateTrack()

        # Run autonomous AI driving control inference if active (rate-limited to 20 Hz / 50ms interval)
        if self.autopilot_enabled and self.autopilot_model is not None:
            self.autopilot_timer += time.dt
            if self.autopilot_timer >= self.autopilot_interval:
                self.autopilot_timer = 0.0
                self.run_autopilot_step()

        # Update on-screen Speed Limit text under current speed
        self.car.SpeedLimitText.text = f"Speed Limit: {self.speedLimit} MPH"
        self.car.setAutopilotIndicator(self.autopilot_enabled and self.autopilot_model is not None)

        self.timer += time.dt
        if self.timer > self.weatherUpdateInterval:
            self.timer = 0.0
            # print(str(self.speedLimit)) #TEMP FOR TESTING SPEEDLIMIT CHANGE
            if self.weatherEnabled and random.randint(1, 50) == 1:
                self.weatherAssist()

        for chunk in self.worldObjects.values():
            for obj in chunk:
                if isinstance(obj, speedLimitSign) and not obj.has_triggered:
                    if self.car.intersects(obj.trigger).hit:
                        self.speedLimit = obj.getSpeedLimit()
                        obj.has_triggered = True
                        print(f"Passed speed sign! New Speed Limit: {self.speedLimit} MPH")

        for chunk in self.worldObjects.values():
            for obj in chunk:
                if isinstance(obj, StopSign) and not obj.has_triggered:
                    if self.car.intersects(obj.trigger).hit and self.car.velocity == 0:
                        invoke(obj.swapToGreen, delay=1.5)
                        obj.has_triggered = True
                        print("fully stopped")

    #user input stuff, remade by ai, but should still support all controls, except the ones outsourced to ui
    def input(self, key):
        if not self.enabled:
            return

        if hasattr(self, 'model_menu_panel') and self.model_menu_panel:
            if key == 'scroll up':
                self.prompt_model_selection(page=self.model_menu_page - 1)
                return
            elif key == 'scroll down':
                self.prompt_model_selection(page=self.model_menu_page + 1)
                return

        # A scored run must not be altered halfway through. Camera views stay available
        # so you can still watch it; everything that would change the world, the model,
        # the recording state or the weather is locked out.
        if self.evaluationLocked and key not in ('1', '2'):
            if key in ('0', '-', '_', '8', '9', 'escape', 'k'):
                print("EVALUATION IN PROGRESS: that control is disabled until the run ends")
            return

        if key == 'escape':
            self.toggleSimPanel()

        elif key == '0':
            self.toggle_autopilot()

        elif key == '-' or key == '_':
            self.prompt_model_selection()

        elif key == '1':
            self.setCameraView('chase')
            print("CAMERA: Chase View")

        elif key == '2':
            self.setCameraView('windshield')
            print("CAMERA: Windshield View")

        elif key == '9':
            # Press 9 to toggle status light circle between red and green
            if self.car.statusCircle.color == color.red:
                self.set_status_green()
                print("STATUS LIGHT: Green")
                self.logger.toggle()
            else:
                self.set_status_red()
                print("STATUS LIGHT: Red")
                self.logger.toggle()

        elif key == '8':
            # DAgger capture: model drives, only your corrections get recorded.
            on = self.logger.toggleCorrectionsOnly()
            if not hasattr(self, 'correction_text') or not self.correction_text:
                self.correction_text = Text(text="", parent=camera.ui,
                                            position=(-0.78, 0.26), scale=1.1,
                                            color=color.orange)
            self.correction_text.text = "CORRECTIONS ONLY" if on else ""
            print(f"CORRECTION CAPTURE: {'ON' if on else 'OFF'} -- "
                  f"{'only frames where you hold W/A/S/D are logged' if on else 'logging every frame again'}")
            if on and not self.logger.recordData:
                print("   (recording is still OFF -- press 9 to start)")
            if on and not self.autopilot_enabled:
                print("   (autopilot is OFF -- press 0 so the model drives)")

    def on_destroy(self):
        """If WorldGenerator is destroyed/unloaded, shut down the logger"""
        if hasattr(self, 'logger') and self.logger:
            self.logger.shutdown()
