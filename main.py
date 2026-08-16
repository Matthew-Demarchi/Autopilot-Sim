#unused file, migrated to autopilot_sim
#Don't delete until everything works under autopilot_sim


# import timeit
# import numpy as np
# import os
#
# from panda3d.core import look_at
# from ursina import *
# from ursina import camera
# from ursina.camera import Camera
# import DataCollection
#
# # 1. Initialize the Ursina app
# app = Ursina()
#
#
#
# class WeatherSystem(Entity):
#     def __init__(self, target, **kwargs):
#         super().__init__(**kwargs)
#         self.target = target
#         self.mode = 'clear'
#         self.max_particles = 150
#         self.particles = []
#         self.spawn_radius = 22
#         self.fall_speed = 14
#         self.spawn_height = 18
#         self.floor_offset = -2
#         self.overlay_color = None
#
#         for i in range(self.max_particles):
#             p = Entity(model='cube', scale=0.05, enabled=False, parent=self)
#             self.particles.append(p)
#
#     def set_weather(self, mode):
#         self.mode = mode
#
#         if mode == 'clear':
#             self.spawn_height = 18
#             camera.overlay.color = color.clear
#             self.overlay_color = None
#             for p in self.particles:
#                 p.enabled = False
#             return
#
#         if mode == 'rain':
#             self.spawn_height = 18
#             camera.overlay.color = color.rgba(30 / 255, 35 / 255, 45 / 255, 110 / 255)
#             self.overlay_color = (30, 35, 45, 110)
#             self._configure_particles(
#                 scale=(0.02, 0.25, 0.02),
#                 col=color.rgba(180 / 255, 200 / 255, 220 / 255, 160 / 255),
#                 fall_speed=28
#             )
#         elif mode == 'snow':
#             self.spawn_height = 18
#             camera.overlay.color = color.rgba(210 / 255, 215 / 255, 225 / 255, 35 / 255)
#             self.overlay_color = (210, 215, 225, 35)
#             self._configure_particles(
#                 scale=(0.06, 0.06, 0.06),
#                 col=color.white,
#                 fall_speed=6
#             )
#         elif mode == 'sandstorm':
#             self.spawn_height = 8
#             camera.overlay.color = color.rgba(190 / 255, 150 / 255, 90 / 255, 90 / 255)
#             self.overlay_color = (190, 150, 90, 90)
#             self._configure_particles(
#                 scale=(0.08, 0.08, 0.08),
#                 col=color.rgba(200 / 255, 160 / 255, 100 / 255, 200 / 255),
#                 fall_speed=2
#             )
#
#         # Distribute particles
#         for p in self.particles:
#             p.enabled = True
#             p.position = Vec3(
#                 random.uniform(-self.spawn_radius, self.spawn_radius),
#                 random.uniform(0, self.spawn_height),
#                 random.uniform(-12, 25)
#             )
#
#     def _configure_particles(self, scale, col, fall_speed):
#         self.fall_speed = fall_speed
#         for p in self.particles:
#             p.scale = scale
#             p.color = col
#
#     def update(self):
#         if self.mode == 'clear':
#             return
#
#         # Keep WeatherSystem centered and oriented with the vehicle
#         self.position = self.target.world_position
#         self.rotation_y = self.target.rotation_y
#
#         speed = getattr(self.target, 'velocity', 0)
#         drift_x = 6 if self.mode == 'sandstorm' else 0  # sideways blow for sandstorm
#
#         for p in self.particles:
#             # 1. Fall vertically in local space
#             p.y -= self.fall_speed * time.dt
#             # 2. Sideways drift (e.g., sandstorm)
#             p.x += drift_x * time.dt
#             # 3. Vehicle driving forward moves particles BACKWARD relative to the car
#             p.z -= speed * time.dt
#
#             # Wrap conditions in local box relative to vehicle:
#             # Hit floor -> respawn near top
#             if p.y < self.floor_offset:
#                 p.y = random.uniform(self.spawn_height - 3.0, self.spawn_height)
#                 p.x = random.uniform(-self.spawn_radius, self.spawn_radius)
#                 p.z = random.uniform(-10.0, 25.0)
#
#             # Passed behind car/chase camera -> wrap to front of vehicle
#             elif p.z < -18.0:
#                 p.z = random.uniform(18.0, 25.0)
#                 p.y = random.uniform(2.0, self.spawn_height)
#                 p.x = random.uniform(-self.spawn_radius, self.spawn_radius)
#
#             # Driven backward past front -> wrap to back of vehicle
#             elif p.z > 25.0:
#                 p.z = random.uniform(-18.0, -12.0)
#                 p.y = random.uniform(2.0, self.spawn_height)
#                 p.x = random.uniform(-self.spawn_radius, self.spawn_radius)
#
#             # Drifted too far sideways -> wrap back into lane width
#             elif abs(p.x) > (self.spawn_radius + 5.0):
#                 p.x = random.uniform(-self.spawn_radius, self.spawn_radius)
#
# class Pedestrian(Entity):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         # Torso (The root basis)
#         self.torso = Entity(model='cube', color=color.blue, scale=(0.5, 0.8, 0.3), y=0.9, parent=self)
#         # Head
#         self.head = Entity(model='cube', color=color.peach, scale=(0.3, 0.3, 0.3), y=1.45, parent=self)
#         # Left Leg
#         self.leg_l = Entity(model='cube', color=color.brown, scale=(0.2, 0.5, 0.2), y=0.25, x=-0.15, parent=self)
#         # Right Leg
#         self.leg_r = Entity(model='cube', color=color.brown, scale=(0.2, 0.5, 0.2), y=0.25, x=0.15, parent=self)
#         # Arms (simplified as a single bar passing through the shoulder)
#         self.arms = Entity(model='cube', color=color.blue, scale=(0.9, 0.2, 0.2), y=1.1, parent=self)
#
#
# class StopSign(Entity):
#     def __init__(self, position, road_direction, roadWidth=10, **kwargs):
#         # We explicitly pass the position into super() so the entity anchors before calculating children
#         super().__init__(position=position, **kwargs)
#
#         # 1. The Metal Pole
#         self.pole = Entity(model='cube', color=color.light_gray,
#                            scale=(0.1, 4, 0.1), y=2, z=5, parent=self)
#
#         # 2. The Red Sign Backing
#         self.sign = Entity(model='cube', color=color.red,
#                            scale=(1.2, 1.2, 0.1), y=4, z=-0.03 +5, parent=self)
#
#         # 3. Simple visible white strip placeholder for the word "STOP"
#         self.text_stripe = Entity(model='cube', color=color.white,
#                                   scale=(0.8, 0.2, 0.12), y=4, z=-0.05 + 5, parent=self)
#
#         # 4. Turn the entire master assembly to face the road direction
#         self.look_at(self.position + road_direction)
#
#         # 5. Lock the master pole so it stays perfectly vertical, ignoring steep hills
#         self.rotation_x = 0
#         self.rotation_z = 0
#
#         # --- ENCAPSULATED GROUND STRIP ---
#         # Because the pole is perfectly vertical and facing forward, we can jump perfectly
#         # sideways (negative local X) to hit the exact center of the lane.
#         x_offset = -(roadWidth / 4 + 1.5)
#
#         self.groundStrip = Entity(
#             model='cube',
#             color=color.white,
#             scale=(roadWidth / 2, 0.2, 1.5),
#             parent=self,
#             position=Vec3(x_offset, 0.15,-2)  # y=0.15 floats it just above the asphalt
#         )
#
#         # Tell the child strip to tilt up/down to match the hill slope perfectly!
#         lane_center_world = self.groundStrip.world_position
#         self.groundStrip.look_at(lane_center_world + road_direction)
#
#         # --- NEW: Trigger Box across the lane ---
#         # Position it across the road lane (x offset towards road center)
#         x_offset = -(roadWidth / 4 + 1.5)
#         self.trigger = Entity(
#             model='cube',
#             scale=(roadWidth, 6, 2),  # Widen to cover the whole road & height
#             position=Vec3(x_offset, 2, -4),
#             parent=self,
#             collider='box',
#             visible=False,  # Keep it invisible
#             enabled=True
#         )
#         self.has_triggered = False  # Prevent firing multiple frames in a row
#
#
#     def swapToGreen(self):
#         self.sign.color=color.lime
#
#
# class speedLimitSign(Entity):
#     def __init__(self, position, road_direction, roadWidth=10, **kwargs):
#         # We explicitly pass the position into super() so the entity anchors before calculating children
#         super().__init__(position=position, **kwargs)
#
#         # 1. The Metal Pole
#         self.pole = Entity(model='cube', color=color.light_gray,
#                            scale=(0.1, 4, 0.1), y=2, parent=self)
#
#         # 2. The Red Sign Backing
#         self.sign = Entity(model='cube', color=color.white,
#                            scale=(1.2, 1.2, 0.1), y=4, z=-0.03, parent=self)
#
#         self.textWritten =str(random.randrange(25,60, 5))
#
#         # 3. Simple visible white strip placeholder for the word "STOP"
#         self.text_stripe = Text(text = self.textWritten, parent=self.sign, color=color.black, position=(0,0, -0.9), origin=(0,0))
#         self.text_stripe.world_scale = 30
#         # 4. Turn the entire master assembly to face the road direction
#         self.look_at(self.position + road_direction)
#
#         # 5. Lock the master pole so it stays perfectly vertical, ignoring steep hills
#         self.rotation_x = 0
#         self.rotation_z = 0
#
#         # --- NEW: Trigger Box across the lane ---
#         # Position it across the road lane (x offset towards road center)
#         x_offset = -(roadWidth / 4 + 1.5)
#         self.trigger = Entity(
#             model='cube',
#             scale=(roadWidth, 6, 2),  # Widen to cover the whole road & height
#             position=Vec3(x_offset, 2, 0),
#             parent=self,
#             collider='box',
#             visible=False,  # Keep it invisible
#             enabled=True
#         )
#         self.has_triggered = False  # Prevent firing multiple frames in a row
#
#     def getSpeedLimit(self):
#         return int(self.textWritten)
#
#
# class Car(Entity):
#     def __init__(self, carColor,**kwargs):
#         super().__init__(model = 'cube', color=carColor, scale=(2,3/2,2), collider='box', **kwargs)
#
#         self.bodyBorder = Entity(model='cube', color=color.black, scale=1.001, parent=self, wireframe=True)
#
#         self.front = Entity(model='cube', color=carColor, scale=(1,1.25/3,1/3), collider='box', parent=self, position=Vec3(0,-.5 + 1.25/6,.5 + 1/6))
#         self.frontBorder = Entity(model='cube', color=color.black, scale=1.001, parent=self.front, wireframe=True)
#
#
#         wheel_scale = (0.3, 0.3, 0.3)
#         wheel_rotation = Vec3(0, 0, 90)
#
#         # FIX: Changed 'sides' to 'resolution', removed 'start=0'
#         procedural_wheel1 = Cylinder(resolution=16, height=1, radius=0.5, direction=(0, 1, 0))
#
#
#
#         # Front Right
#         self.wheel1 = Entity(model=copy(procedural_wheel1), color=color.black, parent=self,
#                              scale=wheel_scale, rotation=wheel_rotation,
#                              position=Vec3(0.55-.3, -0.5, 0.4 + .15))
#
#         # Front Left
#         self.wheel2 = Entity(model=copy(procedural_wheel1), color=color.black, parent=self,
#                              scale=wheel_scale, rotation=wheel_rotation,
#                              position=Vec3(-0.55, -0.5, 0.4 + .15))
#
#         # Back Right
#         self.wheel3 = Entity(model=copy(procedural_wheel1), color=color.black, parent=self,
#                              scale=wheel_scale, rotation=wheel_rotation,
#                              position=Vec3(0.55-.3, -0.5, -0.4))
#
#         # Back Left
#         self.wheel4 = Entity(model=copy(procedural_wheel1), color=color.black, parent=self,
#                              scale=wheel_scale, rotation=wheel_rotation,
#                              position=Vec3(-0.55, -0.5, -0.4))
#
#         #CAMERAS
#         # self.windshieldCentered = Entity(model='cube', color=color.black, parent=self, scale=(0.1, 0.1, 0.1), position=Vec3(0, .45, .52), rotation_x=0)
#         # self.windshieldLeft = Entity(model='cube', color=color.red, parent=self, scale=(0.1, 0.1, 0.1), position=Vec3(-.45, .45, .52), rotation_x=0)
#         # self.windshieldRight = Entity(model='cube', color=color.black, parent=self, scale=(0.1, 0.1, 0.1), position=Vec3(.45, .45, .52), rotation_x=0)
#         # self.thirdPerson = Entity(model='cube', color=color.black, parent=self, scale=(0.1, 0.1, 0.1), position=Vec3(0, 1.5, -8), rotation_x=15)
#         self.windshieldCentered = Entity(parent=self, position=Vec3(0, .45, .52), rotation_x=12)
#         self.windshieldLeft = Entity(parent=self, position=Vec3(-.45, .45, .52), rotation_x=12)
#         self.windshieldRight = Entity(parent=self, position=Vec3(.45, .45, .52), rotation_x=12)
#         self.thirdPerson = Entity(parent=self, position=Vec3(0, 5, -15), rotation_x=10)
#
#
#
#         #variables
#         self.velocity = 0
#         self.wheelAngle = 0 #[-540,540]
#         self.acceleration = 0
#         self.breakPedal = 0 #[0,1]
#         self.accelerationPedal = 0 #[0,1]
#         # --- Tweakable Tuning Values ---
#         self.MAX_SPEED = 90.0  # Top speed capability
#         self.MAX_ACCEL = 38.0  # Punchiness off the line (0-30 mph feel)
#         self.MAX_BRAKE = 35.0  # Brakes are usually 2-3x stronger than engine!
#         self.FRICTION = 0.02  # Natural slowing down (air resistance/rolling)
#
#         self.Speedometer = Text(
#             text="Speed: 0 miles/hr",
#             position=(-0.8, 0.45),
#             scale=2,
#             color=color.white
#         )
#
#     def update(self):
#         # --- 1. SMOOTH PEDAL INPUTS ---
#         pedal_response_speed = 4.0
#         if held_keys['w']:
#             self.accelerationPedal = min(1.0, self.accelerationPedal + time.dt * pedal_response_speed)
#         else:
#             self.accelerationPedal = max(0.0, self.accelerationPedal - time.dt * pedal_response_speed)
#
#         if held_keys['s']:
#             self.breakPedal = min(1.0, self.breakPedal + time.dt * pedal_response_speed)
#         else:
#             self.breakPedal = max(0.0, self.breakPedal - time.dt * pedal_response_speed)
#
#         # --- 2. STEERING WHEEL INPUT ---
#         steer_speed = 750.0
#         return_speed = 900.0
#         if held_keys['a']:
#             self.wheelAngle = max(-540.0, self.wheelAngle - steer_speed * time.dt)
#         elif held_keys['d']:
#             self.wheelAngle = min(540.0, self.wheelAngle + steer_speed * time.dt)
#         else:
#             if self.wheelAngle > 10:
#                 self.wheelAngle -= return_speed * time.dt
#             elif self.wheelAngle < -10:
#                 self.wheelAngle += return_speed * time.dt
#             else:
#                 self.wheelAngle = 0.0
#
#
# # --- 3. PHYSICALLY REALISTIC FORCES ---
#         steer_radians = math.radians((self.wheelAngle / 540.0) * 35.0)
#
#         # Brake overrides throttle entirely — no more pedal-fighting at low speed
#         effective_pedal = 0.0 if self.breakPedal > 0.05 else self.accelerationPedal
#
#         DRAG_COEFF = 0.0012      # quadratic drag — mostly matters at high speed
#         # self.FRICTION (0.02) is your linear rolling resistance, used as-is
#
#         # Smooth ease-in over the first ~14 units of speed, but with a floor
#         # so there's still real force at a dead stop (this is what actually
#         # lets the car launch at all)
#         LAUNCH_FLOOR = 0.35
#         launch_t = min(1.0, abs(self.velocity) / 14.0)
#         smooth_t = launch_t * launch_t * (3 - 2 * launch_t)
#         launch_ramp = LAUNCH_FLOOR + (1.0 - LAUNCH_FLOOR) * smooth_t
#         # Torque floor tuned so full-throttle equilibrium lands right at MAX_SPEED
#         # instead of plateauing early. Nudge TORQUE_FLOOR between 0.28-0.32 to
#         # fine-tune exactly where top speed settles.
#         TORQUE_FLOOR = 0.30
#         speed_ratio = min(1.0, abs(self.velocity) / self.MAX_SPEED)
#         torque_profile = max(TORQUE_FLOOR, 1.0 - speed_ratio * speed_ratio)
#
#         engine_force = effective_pedal * self.MAX_ACCEL * torque_profile * launch_ramp
#         total_resistance = (self.velocity * self.FRICTION) + (self.velocity * abs(self.velocity) * DRAG_COEFF)
#
#         self.acceleration = engine_force - total_resistance
#         self.velocity += self.acceleration * time.dt
#
#         # Kill residual creep so it settles exactly at 0
#         if effective_pedal == 0 and abs(self.velocity) < 0.05:
#             self.velocity = 0.0
#
#         # --- Braking (flat deceleration, not proportional to speed, so it
#         # works just as well at 15 mph as at 80) ---
#         if self.breakPedal > 0 and abs(self.velocity) > 0.01:
#             brake_drop = self.breakPedal * self.MAX_BRAKE * time.dt
#             if self.velocity > 0:
#                 self.velocity = max(0.0, self.velocity - brake_drop)
#             else:
#                 self.velocity = min(0.0, self.velocity + brake_drop)
#
#         self.velocity = max(-self.MAX_SPEED, min(self.MAX_SPEED, self.velocity))
#
# #         # 2. SMOOTH GRADUAL COASTING OVERRIDE (GENTLE ROLL TO A STOP)
# #         if self.accelerationPedal == 0 and self.velocity != 0:
# #             direction = 1 if self.velocity > 0 else -1
# #
# #             # Base ratio up to 45 mph
# #             speed_ratio = min(1.0, abs(self.velocity) / 45.0)
# #
# #             # A smooth, standard linear blend
# #             smooth_curve = speed_ratio
# #
# #             # Lowered low-end deceleration from 5.8 to 2.2 for a soft, gradual stop.
# #             # High-speed coasting remains a loose 1.0.
# #             decel_floor = lerp(2.2, 1.0, smooth_curve)
# #
# #             # Apply the deceleration smoothly against the movement direction
# #             self.velocity -= decel_floor * direction * time.dt
# #
# #             # Hard check to ensure it finishes neatly at zero instead of floating
# #             if (direction == 1 and self.velocity < 0) or (direction == -1 and self.velocity > 0):
# #                 self.velocity = 0.0
# #
# # # NOTE FoR ME: CHANGE SCRAWLING NOT GOING TO 0 (rn infly->0), also, lose speed to quick at top, and accelerate too quick
# #
# #         # 3. Precise Braking (keeps 'S' pedal sharp)
# #         if self.breakPedal > 0 and abs(self.velocity) > 0.01:
# #             brake_drop = self.breakPedal * self.MAX_BRAKE * time.dt
# #             if self.velocity > 0:
# #                 self.velocity = max(0.0, self.velocity - brake_drop)
# #             else:
# #                 self.velocity = min(0.0, self.velocity + brake_drop)
#         # # --- 1. SMOOTH PEDAL INPUTS --- (Keep your existing pedal code here)
#         # pedal_response_speed = 4.0
#         # if held_keys['w']:
#         #     self.accelerationPedal = min(1.0, self.accelerationPedal + time.dt * pedal_response_speed)
#         # else:
#         #     self.accelerationPedal = max(0.0, self.accelerationPedal - time.dt * pedal_response_speed)
#         #
#         # if held_keys['s']:
#         #     self.breakPedal = min(1.0, self.breakPedal + time.dt * pedal_response_speed)
#         # else:
#         #     self.breakPedal = max(0.0, self.breakPedal - time.dt * pedal_response_speed)
#         #
#         # # --- 2. STEERING WHEEL INPUT ---
#         # steer_speed = 750.0
#         # return_speed = 900.0
#         # if held_keys['a']:
#         #     self.wheelAngle = max(-540.0, self.wheelAngle - steer_speed * time.dt)
#         # elif held_keys['d']:
#         #     self.wheelAngle = min(540.0, self.wheelAngle + steer_speed * time.dt)
#         # else:
#         #     if self.wheelAngle > 10:
#         #         self.wheelAngle -= return_speed * time.dt
#         #     elif self.wheelAngle < -10:
#         #         self.wheelAngle += return_speed * time.dt
#         #     else:
#         #         self.wheelAngle = 0.0
#         #
#         # # --- 3. PHYSICALLY REALISTIC FORCES (NEW ENGINE & DRAG MODEL) ---
#         # # Normalize wheel angle to standard front-tire turn degrees [-35, 35]
#         # steer_radians = math.radians((self.wheelAngle / 540.0) * 35.0)
#         #
#         # # Split friction into dynamic categories
#         # DRAG_COEFF = 0.003  # Air resistance scales with velocity squared
#         # ROLLING_RESIST = 0.04  # Constant tire resistance (stops rocket take-offs)
#         #
#         # # 1. Smooth Launch Limiter: Prevents the 0->30 rocket launch
#         # # Ramps power cleanly from 35% up to 100% as you transition from 0 to 25 mph
#         # launch_ramp = lerp(0.35, 1.0, min(1.0, abs(self.velocity) / 25.0))
#         #
#         # # 2. High-speed torque profile drops off near top speed
#         # torque_profile = max(0.1, 1.0 - (abs(self.velocity) / self.MAX_SPEED))
#         #
#         # # Calculate final engine force with the launch buffer applied
#         # engine_force = self.accelerationPedal * self.MAX_ACCEL * torque_profile * launch_ramp
#         #
#         # # Calculate resistance forces (noticeably weaker at low speeds now)
#         # total_resistance = (self.velocity * ROLLING_RESIST) + (self.velocity * abs(self.velocity) * DRAG_COEFF)
#         #
#         # self.acceleration = engine_force - total_resistance
#         # self.velocity += self.acceleration * time.dt
#         #
#         # # Precise Braking (keeps brakes sharp when you actually hit 'S')
#         # if self.breakPedal > 0 and abs(self.velocity) > 0.01:
#         #     brake_drop = self.breakPedal * self.MAX_BRAKE * time.dt
#         #     self.velocity = max(0.0, self.velocity - brake_drop) if self.velocity > 0 else min(0.0,
#         #                                                                                        self.velocity + brake_drop)
#         #
#         # # Power delivery curve: Makes the start smooth, peaking in mid-range torque
#         # torque_profile = max(0.1, 1.0 - (abs(self.velocity) / self.MAX_SPEED))
#         # engine_force = self.accelerationPedal * self.MAX_ACCEL * torque_profile
#         #
#         # # Total resistive forces acting against the tires
#         # # SWAP THIS LINE IN YOUR CODE:
#         # total_resistance = (self.velocity * ROLLING_RESIST) + (self.velocity * abs(self.velocity) * DRAG_COEFF)
#         #
#         # self.acceleration = engine_force - total_resistance
#         # self.velocity += self.acceleration * time.dt
#         #
#         # # Precise Braking
#         # if self.breakPedal > 0 and abs(self.velocity) > 0.01:
#         #     brake_drop = self.breakPedal * self.MAX_BRAKE * time.dt
#         #     self.velocity = max(0.0, self.velocity - brake_drop) if self.velocity > 0 else min(0.0,
#         #                                                                                        self.velocity + brake_drop)
#
#         # --- 4. DYNAMIC HIGH-SPEED LATERAL GRIP TRACKING (NEW) ---
#         yaw = math.radians(self.rotation_y)
#         flat_forward = Vec3(math.sin(yaw), 0, math.cos(yaw))
#
#         if abs(self.velocity) > 0.1:
#             # Steering authority still tapers at speed (so full-lock doesn't
#             # snap you into an instant spin), but the floor is high enough
#             # that you always have real turning power to correct a mistake.
#             # Old floor (0.15) meant from ~61 mph up you had almost nothing
#             # left to fight a drift with — this raises that reserve.
#             STEER_FLOOR = 0.40
#             speed_grip_factor = max(STEER_FLOOR, 1.0 - (abs(self.velocity) / (self.MAX_SPEED * 1.15)))
#
#             self.rotation_y += math.degrees(steer_radians) * speed_grip_factor * 4.0 * time.dt
#
#             LATERAL_TRACTION = 8.5  # Higher = arcade rails, Lower = ice/drift simulation
#             actual_movement_direction = lerp(flat_forward, flat_forward + (
#                     Vec3(math.cos(yaw), 0, -math.sin(yaw)) * math.sin(steer_radians)),
#                                              LATERAL_TRACTION * time.dt)
#
#             self.position += actual_movement_direction.normalized() * self.velocity * time.dt
#
#         # # --- 1. SMOOTH PEDAL INPUTS ---
#         # # If pressing 'w', gas pedal pushes down. If let go, it springs back up.
#         # pedal_response_speed = 4.0  # How fast the pedal moves down/up
#         #
#         # if held_keys['w']:
#         #     self.accelerationPedal = min(1.0, self.accelerationPedal + time.dt * pedal_response_speed)
#         # else:
#         #     self.accelerationPedal = max(0.0, self.accelerationPedal - time.dt * pedal_response_speed)
#         #
#         # # If pressing 's', brake pedal pushes down.
#         # if held_keys['s']:
#         #     self.breakPedal = min(1.0, self.breakPedal + time.dt * pedal_response_speed)
#         # else:
#         #     self.breakPedal = max(0.0, self.breakPedal - time.dt * pedal_response_speed)
#         #
#         # # --- 2. UPGRADED HIGH-RESPONSIVENESS STEERING ---
#         # # Doubled the input speed so the wheel snaps to your commands instantly
#         # steer_speed = 750.0  # Fast, highly responsive tracking
#         # return_speed = 900.0  # Snaps back to center aggressively when you let go
#         #
#         # if held_keys['a']:
#         #     self.wheelAngle = max(-540.0, self.wheelAngle - steer_speed * time.dt)
#         # elif held_keys['d']:
#         #     self.wheelAngle = min(540.0, self.wheelAngle + steer_speed * time.dt)
#         # else:
#         #     if self.wheelAngle > 10:
#         #         self.wheelAngle -= return_speed * time.dt
#         #     elif self.wheelAngle < -10:
#         #         self.wheelAngle += return_speed * time.dt
#         #     else:
#         #         self.wheelAngle = 0.0
#         # # # --- 2. SMOOTH STEERING WHEEL ---
#         # # # Simulating a steering wheel spinning back to center automatically
#         # # steer_speed = 300.0  # Degrees per second the wheel turns
#         # # return_speed = 400.0  # How fast the wheel snaps back to center
#         # #
#         # # if held_keys['a']:
#         # #     self.wheelAngle = max(-540.0, self.wheelAngle - steer_speed * time.dt)
#         # # elif held_keys['d']:
#         # #     self.wheelAngle = min(540.0, self.wheelAngle + steer_speed * time.dt)
#         # # else:
#         # #     # No keys pressed? Automatically center the wheel
#         # #     if self.wheelAngle > 5:
#         # #         self.wheelAngle -= return_speed * time.dt
#         # #     elif self.wheelAngle < -5:
#         # #         self.wheelAngle += return_speed * time.dt
#         # #     else:
#         # #         self.wheelAngle = 0.0
#         #
#         #
#         # # 1. Calculate Engine Acceleration as a function of velocity
#         # # As velocity approaches MAX_SPEED, engine power efficiency drops to 0
#         # speed_ratio = abs(self.velocity) / self.MAX_SPEED
#         # available_horsepower = max(0.0, 1.0 - speed_ratio)
#         # engine_force = self.accelerationPedal * self.MAX_ACCEL * available_horsepower
#         # drag_force = self.velocity * self.FRICTION
#         #
#         # # Apply basic movement forces
#         # self.acceleration = engine_force - drag_force
#         # self.velocity += self.acceleration * time.dt
#         #
#         # # Apply braking cleanly as a reduction tool, preventing the 0mph jitter bug
#         # if self.breakPedal > 0 and abs(self.velocity) > 0.01:
#         #     brake_drop = self.breakPedal * self.MAX_BRAKE * time.dt
#         #     if self.velocity > 0:
#         #         self.velocity = max(0.0, self.velocity - brake_drop)
#         #     else:
#         #         self.velocity = min(0.0, self.velocity + brake_drop)
#         #
#         # # Movement Execution
#         # if abs(self.velocity) > 0.1:
#         #     normalized_steer = self.wheelAngle / 540.0
#         #
#         #     # Increased base maneuverability for micro-corrections
#         #     BASE_STEER_POWER = 280.0
#         #
#         #     # FINE-TUNED SPEED DAMPENER:
#         #     # Changes the curve so high-speed tracking drops to a reliable 0.45
#         #     # instead of dropping to a useless 0.2. This gives you recovery grip!
#         #     speed_ratio = abs(self.velocity) / self.MAX_SPEED
#         #     speed_dampener = max(0.28, 1.0 - (speed_ratio ** 2 * 0.82))
#         #
#         #
#         #     direction_modifier = 1 if self.velocity > 0 else -1
#         #     self.rotation_y += normalized_steer * BASE_STEER_POWER * speed_dampener * direction_modifier * time.dt
#
#
#         # 2. MINI GROUND-COLLISION SNAPPER
#         # Cast an invisible ray downwards from slightly above the car's center
#         # origin = self.world_position + Vec3(0, 1, 0)
#         # hit_info = raycast(origin, Vec3(0, -1, 0), distance=10, ignore=(self,))
#         #
#         # if hit_info.hit:
#         #     # If the ray hits your road mesh, snap the car's Y height to the road's surface!
#         #     self.y = hit_info.world_point.y + 1
#         # else:
#         #     # If you drive off the edge of the mesh, gravity pulls you down!
#         #     self.y -= 9.8 * time.dt
#
#         # --- 2. MINI GROUND-COLLISION SNAPPER ---
#
#         # Calculate a "flat" forward vector based ONLY on your steering (yaw)
#         # This guarantees the rays stay exactly 1.7 units apart horizontally at all times. WHAT A STUPID BUG
#         # yaw = math.radians(self.rotation_y)
#         # flat_forward = Vec3(math.sin(yaw), 0, math.cos(yaw))
#         #
#         # # Apply the flat forward vector for the ray origins
#         # origin1 = self.world_position + Vec3(0, 1, 0) + (flat_forward * 0.85)
#         # origin2 = self.world_position + Vec3(0, 1, 0) - (flat_forward * 0.85)
#         #
#         # hit_info1 = raycast(origin1, Vec3(0, -1, 0), distance=10, ignore=(self,))
#         # hit_info2 = raycast(origin2, Vec3(0, -1, 0), distance=10, ignore=(self,))
#         #
#         # if hit_info1.hit or hit_info2.hit:
#         #     if not hit_info1.hit:
#         #         hit_info1 = hit_info2
#         #     if not hit_info2.hit:
#         #         hit_info2 = hit_info1
#         #
#         #     # FIX THE FLOATING: Changed "+ 1" to "+ 0.65" (Tweak this to perfectly touch the wheels to the ground)
#         #     self.y = (hit_info1.world_point.y + hit_info2.world_point.y) / 2 + 1
#         #
#         #     # The horizontal distance is permanently 1.7, making this math completely stable now
#         #     self.rotation_x = -math.degrees(math.atan2((hit_info1.world_point.y - hit_info2.world_point.y), 1.7))
#         #
#         # else:
#         #     # If you drive off the edge of the mesh, gravity pulls you down!
#         #     self.y -= 9.8 * time.dt
#
#         # Generate our flat directionals right before moving
#         yaw = math.radians(self.rotation_y)
#         flat_forward = Vec3(math.sin(yaw), 0, math.cos(yaw))
#         flat_right = Vec3(math.cos(yaw), 0, -math.sin(yaw))
#
#         # Move the car using flat_forward ONLY.
#         # This stops the car from physically driving itself under the terrain!
#         self.position += flat_forward * self.velocity * time.dt
#
#         # --- 6. ARCADE SUSPENSION SYSTEM ---
#         # Raise origins to +2.0 so they survive sudden, steep drops
#         center_high = self.world_position + Vec3(0, 2.0, 0)
#         z_offset = 0.55
#         x_offset = 0.55
#
#         origin_FL = center_high + (flat_forward * z_offset) - (flat_right * x_offset)
#         origin_FR = center_high + (flat_forward * z_offset) + (flat_right * x_offset)
#         origin_BL = center_high - (flat_forward * z_offset) - (flat_right * x_offset)
#         origin_BR = center_high - (flat_forward * z_offset) + (flat_right * x_offset)
#
#         # YOU MUST IGNORE self.front! It has a box collider that ruins the raycasts.
#         ignore_list = (self, self.front)
#         RAY_LENGTH = 5.0
#
#         hit_FL = raycast(origin_FL, Vec3(0, -1, 0), distance=RAY_LENGTH, ignore=ignore_list)
#         hit_FR = raycast(origin_FR, Vec3(0, -1, 0), distance=RAY_LENGTH, ignore=ignore_list)
#         hit_BL = raycast(origin_BL, Vec3(0, -1, 0), distance=RAY_LENGTH, ignore=ignore_list)
#         hit_BR = raycast(origin_BR, Vec3(0, -1, 0), distance=RAY_LENGTH, ignore=ignore_list)
#
#         # Expanded MAX_DROP to 3.5 to account for the new +2.0 origin height
#         MAX_DROP = 3.5
#         ride_height = 0.90
#         hanging_y = self.y - 0.2
#
#         valid_FL = hit_FL.hit and hit_FL.distance < MAX_DROP
#         valid_FR = hit_FR.hit and hit_FR.distance < MAX_DROP
#         valid_BL = hit_BL.hit and hit_BL.distance < MAX_DROP
#         valid_BR = hit_BR.hit and hit_BR.distance < MAX_DROP
#
#         y_FL = hit_FL.world_point.y if valid_FL else hanging_y
#         y_FR = hit_FR.world_point.y if valid_FR else hanging_y
#         y_BL = hit_BL.world_point.y if valid_BL else hanging_y
#         y_BR = hit_BR.world_point.y if valid_BR else hanging_y
#
#         grounded_wheels = sum([valid_FL, valid_FR, valid_BL, valid_BR])
#
#         if grounded_wheels > 0:
#             valid_ys = [y for y, v in [(y_FL, valid_FL), (y_FR, valid_FR),
#                                        (y_BL, valid_BL), (y_BR, valid_BR)] if v]
#             target_y = sum(valid_ys) / len(valid_ys) + ride_height
#
#             # --- NEW (fixed) ---
#             # Pitch: only calculate if there's real ground data on BOTH front and back
#             front_hits = [y_FL if valid_FL else None, y_FR if valid_FR else None]
#             back_hits = [y_BL if valid_BL else None, y_BR if valid_BR else None]
#             front_hits = [y for y in front_hits if y is not None]
#             back_hits = [y for y in back_hits if y is not None]
#
#             if front_hits and back_hits:
#                 front_y = sum(front_hits) / len(front_hits)
#                 back_y = sum(back_hits) / len(back_hits)
#                 target_pitch = -math.degrees(math.atan2((front_y - back_y), z_offset * 2))
#             else:
#                 target_pitch = self.rotation_x  # Hold current pitch — not enough data
#
#             # Roll: only calculate if there's real ground data on BOTH left and right
#             left_hits = [y_FL if valid_FL else None, y_BL if valid_BL else None]
#             right_hits = [y_FR if valid_FR else None, y_BR if valid_BR else None]
#             left_hits = [y for y in left_hits if y is not None]
#             right_hits = [y for y in right_hits if y is not None]
#
#             if left_hits and right_hits:
#                 left_y = sum(left_hits) / len(left_hits)
#                 right_y = sum(right_hits) / len(right_hits)
#                 target_roll = math.degrees(math.atan2((left_y - right_y), x_offset * 2))
#             else:
#                 target_roll = self.rotation_z  # Hold current roll — not enough data
#             # Stiffer springs (25) to prevent the car from sagging under gravity at high speeds
#             spring_stiffness = 25 * time.dt
#
#             self.y = lerp(self.y, target_y, spring_stiffness)
#             self.rotation_x = lerp(self.rotation_x, target_pitch, spring_stiffness)
#             self.rotation_z = lerp(self.rotation_z, target_roll, spring_stiffness)
#
#         else:
#             # Airborne gravity
#             self.y -= 9.8 * time.dt
#             self.rotation_x = lerp(self.rotation_x, 0, 2 * time.dt)
#             self.rotation_z = lerp(self.rotation_z, 0, 2 * time.dt)
#
#
#         self.Speedometer.text = ("Speed: " + str(int(self.velocity)) + "miles/hr")
#
#
#
# class oncomingCar(Entity):
#     def __init__(self, carColor, chunkINDX, start, speed, **kwargs):
#         super().__init__(model = 'cube', color=carColor, scale=(2,3/2,2), collider='box', **kwargs)
#
#         self.bodyBorder = Entity(model='cube', color=color.black, scale=1.001, parent=self, wireframe=True)
#
#         self.front = Entity(model='cube', color=carColor, scale=(1,1.25/3,1/3), collider='box', parent=self, position=Vec3(0,-.5 + 1.25/6,.5 + 1/6))
#         self.frontBorder = Entity(model='cube', color=color.black, scale=1.001, parent=self.front, wireframe=True)
#
#
#         wheel_scale = (0.3, 0.3, 0.3)
#         wheel_rotation = Vec3(0, 0, 90)
#
#         # FIX: Changed 'sides' to 'resolution', removed 'start=0'
#         procedural_wheel1 = Cylinder(resolution=16, height=1, radius=0.5, direction=(0, 1, 0))
#
#
#
#         # Front Right
#         self.wheel1 = Entity(model=copy(procedural_wheel1), color=color.black, parent=self,
#                              scale=wheel_scale, rotation=wheel_rotation,
#                              position=Vec3(0.55-.3, -0.5, 0.4 + .15))
#
#         # Front Left
#         self.wheel2 = Entity(model=copy(procedural_wheel1), color=color.black, parent=self,
#                              scale=wheel_scale, rotation=wheel_rotation,
#                              position=Vec3(-0.55, -0.5, 0.4 + .15))
#
#         # Back Right
#         self.wheel3 = Entity(model=copy(procedural_wheel1), color=color.black, parent=self,
#                              scale=wheel_scale, rotation=wheel_rotation,
#                              position=Vec3(0.55-.3, -0.5, -0.4))
#
#         # Back Left
#         self.wheel4 = Entity(model=copy(procedural_wheel1), color=color.black, parent=self,
#                              scale=wheel_scale, rotation=wheel_rotation,
#                              position=Vec3(-0.55, -0.5, -0.4))
#         self.chunkINDX = chunkINDX
#         self.t = start
#         self.speed = speed
#
#     def shift_index(self):
#         self.chunkINDX -= 1
#
#     def update(self):
#         self.t -= self.speed * time.dt
#
#         #move chunk closer to player
#         if self.t < 0.0:
#             global worldObjects
#             if self.chunkINDX in worldObjects and self in worldObjects[self.chunkINDX]:
#                 worldObjects[self.chunkINDX].remove(self)
#
#             self.chunkINDX -=1
#             self.t =1.0 #reset
#             #print("hi")
#
#             if self.chunkINDX not in worldObjects:
#                 worldObjects[self.chunkINDX] = []
#             worldObjects[self.chunkINDX].append(self)
#
#         global trackPoints
#         if self.chunkINDX - 1 < 0 or self.chunkINDX + 2 >= len(trackPoints):
#             return
#
#         p0 = trackPoints[self.chunkINDX - 1]
#         p1 = trackPoints[self.chunkINDX]
#         p2 = trackPoints[self.chunkINDX + 1]
#         p3 = trackPoints[self.chunkINDX + 2]
#
#         road_center = spline(p0, p1, p2, p3, self.t)
#         road_forward = splineTan(p0, p1, p2, p3, self.t)
#
#         road_right = np.cross(road_forward, np.array(Vec3(0, 1, 0)))
#         norm = np.linalg.norm(road_right)
#         if norm != 0:
#             road_right /= norm
#
#         global globalRoadWidth
#         laneOffset = road_right * globalRoadWidth / 4
#         self.position = Vec3(road_center[0], road_center[1], road_center[2]) - Vec3(-laneOffset[0], laneOffset[1], laneOffset[2]) + Vec3(0,1,0)
#
#         self.look_at(self.position - Vec3(road_forward[0], road_forward[1], road_forward[2]))
#         self.rotation_x = 0
#         self.rotation_z = 0
#
#
# EditorCamera()
# worldObjects = {}
# currentView = 'chase'
#
#
#
#
#
#
#
#
# def spline(p0, p1, p2, p3, t):
#     T = np.array([t**3, t**2, t**1, 1])
#
#     M = np.array([
#         [-1,  3, -3,  1],
#         [ 2, -5,  4, -1],
#         [-1,  0,  1,  0],
#         [ 0,  2,  0,  0]
#     ])
#
#     G = np.array([p0, p1, p2, p3])
#
#     return .5*T @ M @ G
#
# def splineTan(p0, p1, p2, p3, t):
#     TPRIME = np.array([3*t**2, 2*t**1, 1, 0])
#
#     M = np.array([
#         [-1,  3, -3,  1],
#         [ 2, -5,  4, -1],
#         [-1,  0,  1,  0],
#         [ 0,  2,  0,  0]
#     ])
#
#     G = np.array([p0, p1, p2, p3])
#
#     direction = .5*TPRIME @ M @ G
#
#     unitDirection = direction / sqrt((direction)@(direction))
#
#     return unitDirection
#
#
# def generateRoad(controlPoints, roadWidth, grassWidth, resolution):
#     vertices = []
#     triangles = []
#     colors = []
#
#     GLOBAL_UP = Vec3(0, 1, 0)
#     currentIndex = 0
#
#     for i in range(1, len(controlPoints) - 2):
#         p0 = controlPoints[i - 1]
#         p1 = controlPoints[i]
#         p2 = controlPoints[i + 1]
#         p3 = controlPoints[i + 2]
#
#         for j in range(0, resolution + 1):
#             t = j / resolution
#             center = spline(p0, p1, p2, p3, t)
#             forward = splineTan(p0, p1, p2, p3, t)
#
#             rawRightV = np.cross(forward, GLOBAL_UP)
#             right = rawRightV / sqrt(rawRightV @ rawRightV)
#
#             halfRoadWidth = roadWidth / 2.0
#             lineWidth = 0.15
#
#             # --- SHARP COLOR ISOLATION GEOMETRY ---
#             # We calculate distinct, tight boundaries for the lines to stop bleeding
#             vLeftGrass = center - (right * (halfRoadWidth + grassWidth))
#             vLeftLineOuter = center - (right * halfRoadWidth)
#             vLeftLineInner = center - (right * (halfRoadWidth - lineWidth))
#
#             # Asphalt lane interior boundary points
#             vLeftLaneEnd = center - (right * (lineWidth * 1.5))
#             vCenterLeft = center - (right * (lineWidth / 2.0))
#             vCenterRight = center + (right * (lineWidth / 2.0))
#             vRightLaneStart = center + (right * (lineWidth * 1.5))
#
#             vRightLineInner = center + (right * (halfRoadWidth - lineWidth))
#             vRightLineOuter = center + (right * halfRoadWidth)
#             vRightGrass = center + (right * (halfRoadWidth + grassWidth))
#
#             # Append all 10 structural vertices to clear color channels
#             for v in [vLeftGrass, vLeftLineOuter, vLeftLineInner, vLeftLaneEnd,
#                       vCenterLeft, vCenterRight, vRightLaneStart, vRightLineInner, vRightLineOuter, vRightGrass]:
#                 vertices.append(Vec3(v[0], v[1], v[2]))
#
#             # Procedural Dashed Center Line
#             if (j // 3) % 2 == 0:  # Widened dash step slightly for clean visibility
#                 centerMarkingColor = color.yellow
#             else:
#                 centerMarkingColor = color.gray
#
#             # Assign colors strictly to prevent gradient leaking
#             colors.extend([
#                 color.green,  # 0: Left Grass
#                 color.white,  # 1: White Line Outer
#                 color.gray,  # 2: White Line Inner -> Lane Gray
#                 color.gray,  # 3: Mid Lane Gray
#                 centerMarkingColor,  # 4: Center Line Left
#                 centerMarkingColor,  # 5: Center Line Right
#                 color.gray,  # 6: Mid Lane Gray
#                 color.gray,  # 7: Lane Gray -> White Line Inner
#                 color.white,  # 8: White Line Outer
#                 color.green  # 9: Right Grass
#             ])
#
#             # --- SEAMLESS TOPOLOGY SEWING ---
#             # Condition: Skip ONLY the absolute beginning of the entire track
#             if not (i == 1 and j == 0):
#                 prev = currentIndex - 10
#                 curr = currentIndex
#
#                 # Loop to automatically stitch all 9 parallel quad strips
#                 for strip in range(9):
#                     triangles.append((prev + strip, curr + strip, prev + strip + 1))
#                     triangles.append((curr + strip, curr + strip + 1, prev + strip + 1))
#
#             currentIndex += 10
#
#     return vertices, triangles, colors
#
#
# car_colors = [
#     color.black,
#     color.white,
#     color.gray,
#     color.light_gray,
#     color.red,
#     color.blue,
#     color.orange,
#     color.yellow,
#     color.gold,
#     # Custom colors defined using hex or RGB (values 0-255)
#     color.hex("800000"),       # Maroon
#     color.hex("F5F5DC"),       # Beige
#     color.hex("00008b"),       # Dark Blue
#     color.hex("006400")        # Dark Green
# ]
#
# def spawnObstaclesForSegment(p0, p1, p2, p3, chunk_idx, roadWidth, resolution):
#     print("spawnObstaclesForSegment")
#     GLOBAL_UP = Vec3(0, 1, 0)
#
#     if chunk_idx not in worldObjects:
#         worldObjects[chunk_idx] = []
#
#     for j in range(0, resolution + 1):
#         t = j / resolution
#         center = spline(p0, p1, p2, p3, t)
#         forward = splineTan(p0, p1, p2, p3, t)
#         rawRightV = np.cross(forward, GLOBAL_UP)
#         right = rawRightV / sqrt(rawRightV @ rawRightV)
#
#         signChance = random.randint(1, 100)
#         if signChance == 1:
#             signPosition = Vec3(center[0], center[1], center[2]) + (
#                     Vec3(-right[0], right[1], right[2]) * (roadWidth / 2 + 1.5))
#             road_direction = Vec3(forward[0], forward[1], forward[2])
#
#             # The class handles its own rotations, locking, and ground line generation!
#             new_sign = StopSign(position=signPosition, road_direction=road_direction, roadWidth=roadWidth)
#             worldObjects[chunk_idx].append(new_sign)
#
#         pedestrianChance = random.randint(1, 100)
#         if pedestrianChance == 1 and signChance != 1:
#             randPosition = random.uniform(-1.5, .5)
#             pedestrianPosition = Vec3(center[0], center[1], center[2]) + (
#                     Vec3(-right[0], right[1], right[2]) * (roadWidth / 2) + Vec3(randPosition, 0, 0))
#             pedestrian = Pedestrian(position=pedestrianPosition)
#             worldObjects[chunk_idx].append(pedestrian)
#
#         speedSignChance = random.randint(1, 400)
#         if pedestrianChance != 1 and signChance != 1 and speedSignChance == 1:
#             signPosition = Vec3(center[0], center[1], center[2]) + (
#                     Vec3(-right[0], right[1], right[2]) * (roadWidth / 2 + 1.5))
#             road_direction = Vec3(forward[0], forward[1], forward[2])
#
#             # The class handles its own rotations, locking, and ground line generation!
#             new_sign = speedLimitSign(position=signPosition, road_direction=road_direction, roadWidth=roadWidth)
#             worldObjects[chunk_idx].append(new_sign)
#
#
#     print ("summon car??")
#     if random.randint(1, 5) == 1:
#         print("CARRR")
#         # Spawn it at t=1.0 (the far end) with a random driving velocity speed
#         new_npc = oncomingCar(carColor=car_colors[random.randint(0, len(car_colors)-1)], chunkINDX=chunk_idx, start=1.0, speed=random.uniform(0.1, 0.2))
#         worldObjects[chunk_idx].append(new_npc)
#
# def nextMilestone(lastPoint):
#     distanceGenerated = 300
#     forwardStep = lastPoint.z + distanceGenerated
#
#     xCurve = random.uniform(-25, 25)
#     yCurve = random.uniform(-5, 5)
#
#     newPoint = Vec3(xCurve, yCurve, forwardStep)
#
#     return newPoint
#
# #THANK YOU CLAUDE
# def destroy_all(entity):
#     """destroy() doesn't cascade to children in ursina — do it ourselves."""
#     if not entity:
#         return
#     for child in list(entity.children):
#         destroy_all(child)
#     destroy(entity)
#
# speedLimit = 45;
#
# def updateTrack(carPos, controlPts, currentMeshEn):
#     global worldObjects
#     # global speedLimit
#
#     if carPos.z > controlPts[2].z:
#
#         # --- BUG 2 FIX: ESCAPED ENTITY CATCHER ---
#         # Find Chunk 1 AND any chunk less than 1 (catching cars that drove backwards off the track)
#         chunks_to_delete = [k for k in worldObjects.keys() if k <= 1]
#         for k in chunks_to_delete:
#             for obj in worldObjects[k]:
#                 # if isinstance(obj, speedLimitSign): #TEMP FOR TESTING
#                 #     speedLimit = obj.getSpeedLimit()
#                 #     print(str(speedLimit) + '\n')
#                 destroy_all(obj)
#             del worldObjects[k]
#
#         # Shift internal car indices BEFORE dictionary shift
#         for chunk_idx in list(worldObjects.keys()):
#             for obj in worldObjects[chunk_idx]:
#                 if isinstance(obj, oncomingCar):
#                     obj.shift_index()
#
#         # Downshift index markers cleanly to mirror the upcoming track chunk rotation shift
#         new_objects_dict = {}
#         for chunk_idx, obj_list in worldObjects.items():
#             new_objects_dict[chunk_idx - 1] = obj_list
#         worldObjects = new_objects_dict
#
#         controlPts.pop(0)
#
#         lastPoint = controlPts[len(controlPts) - 1]
#
#         newHrznPt = nextMilestone(lastPoint)
#         controlPts.append(newHrznPt)
#
#         destroy(currentMeshEn)
#
#         verts, tris, cols = generateRoad(controlPts, roadWidth=10, grassWidth=30, resolution=12)
#
#         new_mesh = Mesh(vertices=verts, triangles=tris, colors=cols)
#         currentMeshEn = Entity(model=new_mesh, collider='mesh', double_sided=True)
#
#         last_idx = len(controlPts) - 3
#         spawnObstaclesForSegment(
#             controlPts[last_idx - 1], controlPts[last_idx],
#             controlPts[last_idx + 1], controlPts[last_idx + 2],
#             last_idx, roadWidth=10, resolution=12
#         )
#
#     return currentMeshEn
#
#
# # # Initialize the historical boundary tracking list (6 points)
# # # We start straight so the car doesn't instantly spawn inside a hill or curve
# # track_points = [
# #     Vec3(0, 0, -100),   # P0: History padding
# #     Vec3(0, 0, 0),      # P1: Spawn line
# #     Vec3(0, 0, 100),    # P2: First checkpoint threshold
# #     Vec3(0, 0, 200),    # P3: Heading out
# #     Vec3(10, 2, 300),   # P4: First procedural hint
# #     Vec3(-10, -1, 400)  # P5: Horizon anchor
# # ]
#
# # Generate the initial environment mesh
#
# # --- RECONFIGURED HORIZON INITIALIZATION ---
# # Expanded milestone queue to look much further down the driving axis (Z)
# trackPoints = [
#     Vec3(0, 0, -200),   # P0: History padding
#     Vec3(0, 0, 0),      # P1: Spawn line
#     Vec3(0, 0, 200),    # P2: Checkpoint threshold
#     Vec3(0, 0, 400),    # P3: Middle field
#     Vec3(20, 4, 600),   # P4: Winding climb
#     Vec3(-20, -2, 800), # P5: Mid-horizon curve
#     Vec3(0, 0, 1000),   # P6: Far horizon guide
#     Vec3(0, 0, 1200)    # P7: Terminal look-ahead point
# ]
#
# globalRoadWidth = 10
# verts, tris, cols = generateRoad(trackPoints, roadWidth=globalRoadWidth, grassWidth=30, resolution=12)
# road_mesh = Mesh(vertices=verts, triangles=tris, colors=cols)
# road_entity = Entity(model=road_mesh, collider='mesh', double_sided=True)
#
# for initial_chunk in range(1, len(trackPoints) - 2):
#     spawnObstaclesForSegment(
#         trackPoints[initial_chunk - 1], trackPoints[initial_chunk],
#         trackPoints[initial_chunk + 1], trackPoints[trackPoints.index(trackPoints[initial_chunk]) + 2],
#         initial_chunk, roadWidth=10, resolution=12
#     )
#
# car = Car(carColor=color.orange, position=Vec3(0, 10, 0)) #TEMPPPP
#
# weather = WeatherSystem(target=car)
# weather.set_weather('clear')  # start clean
#
# weatherChance = 1
# weather.set_weather('clear')
#
# def weatherAssist():
#     global weatherChance
#     weatherChance = random.randint(1, 4)
#     if weatherChance == 1:
#         weather.set_weather('clear')
#     elif weatherChance == 2:
#         weather.set_weather('rain')
#     elif weatherChance == 3:
#         weather.set_weather('snow')
#     elif weatherChance == 4:
#         weather.set_weather('sandstorm')
#     print("weatherAssist " + str(weatherChance) + '\n')
#
#
#
#
# timer = 0.0
# weatherUpdateInterval = 10.0
#
#
#
# def update():
#     global car
#     global road_entity  # Tell Python to modify our master variable container
#     # Run our dynamic chunk generation check pass every single frame
#     road_entity = updateTrack(car.position, trackPoints, road_entity)
#
#     global timer
#     timer += time.dt
#     if timer > weatherUpdateInterval:
#         timer = 0.0
#         if random.randint(1, 50) == 1:
#             weatherAssist()
#
#     # Inside your main update() or car's update():
#     global speedLimit
#     for chunk in worldObjects.values():
#         for obj in chunk:
#             if isinstance(obj, speedLimitSign) and not obj.has_triggered:
#                 if car.intersects(obj.trigger).hit:
#                     speedLimit = obj.getSpeedLimit()
#                     obj.has_triggered = True  # Mark as read so it doesn't re-trigger
#                     print(f"Passed speed sign! New Speed Limit: {speedLimit} MPH")
#     for chunk in worldObjects.values():
#         for obj in chunk:
#             if isinstance(obj, StopSign) and not obj.has_triggered:
#                 if car.intersects(obj.trigger).hit and car.velocity == Vec3(0, 0, 0):
#                     invoke(obj.swapToGreen, delay=1.5)
#                     obj.has_triggered = True  # Mark as read so it doesn't re-trigger
#                     print(f"fully stopped")
#
# # Force the camera to use your custom third-person anchor on startup
# camera.parent = car.thirdPerson
# camera.position = Vec3(0, 0, 0)
# camera.rotation = Vec3(0, 0, 0)
#
#
# # --- INITIALIZE CAMERA MANAGER ONCE AT STARTUP ---
# camera_mgr = DataCollection.OffscreenCameraManager(resolution=(512, 512))
#
# # Register your 3 windshield entities
# camera_mgr.register_camera(car.windshieldLeft)
# camera_mgr.register_camera(car.windshieldCentered)
# camera_mgr.register_camera(car.windshieldRight)
#
# # --- SIMPLE WRAPPER FUNCTION ---
# def get_camera_image(cam_entity):
#     """
#     Pass any registered camera entity (e.g. car.windshieldCentered)
#     and get back a PIL Image object with true colors and active weather tint.
#     """
#     from PIL import Image
#     img = camera_mgr.get_image(cam_entity)
#     if img and weather and weather.mode != 'clear' and weather.overlay_color:
#         tint_img = Image.new('RGBA', img.size, weather.overlay_color)
#         img = Image.alpha_composite(img, tint_img)
#     return img
#
# tempFileNum = 0
# #temp for testing
# def input(key):
#     # PRESS 1: Snap to Chase Camera
#     if key == '1':
#         EditorCamera.enabled = False
#         camera.parent = car.thirdPerson  # Tell camera to follow the chase anchor
#         camera.position = Vec3(0, 0, 0)  # Snap perfectly to its position
#         camera.rotation = Vec3(0, 0, 0)  # Match its angle perfectly
#
#         camera.fov = 60  # <--- Reset back to normal perspective here!
#         print("CAMERA: Chase View Attached")
#
#     # PRESS 2: Snap to Windshield Camera
#     elif key == '2':
#         EditorCamera.enabled = False
#         camera.parent = car.windshieldCentered  # Tell camera to follow windshield anchor
#         camera.position = Vec3(0, 0, 0)  # Snap perfectly
#         camera.rotation = Vec3(0, 0, 0)  # Match angle
#
#         camera.fov = 100  # Default is usually around 40-60
#         print("CAMERA: Windshield View Attached")
#
#     # PRESS 3: Free Fly Editor Camera
#     elif key == '3':
#         camera.parent = scene  # Detach the camera from the car entirely
#         EditorCamera.enabled = True
#
#         camera.fov = 60  # <--- Reset back to normal perspective here!
#         print("CAMERA: Free Fly Editor Mode")
#     elif key == '4':
#         weather.set_weather('clear')
#     elif key == '5':
#         weather.set_weather('rain')
#     elif key == '6':
#         weather.set_weather('snow')
#     elif key == '7':
#         weather.set_weather('sandstorm')
#     elif key == '8':
#         print("getting image")
#         imagussy= get_camera_image(car.windshieldCentered)
#         print("saving image to " + os.getcwd())
#         global tempFileNum
#         imagussy.save("imagussy" + str(tempFileNum) + ".png")
#         tempFileNum += 1
#
# app.run()
#
#
#
#
