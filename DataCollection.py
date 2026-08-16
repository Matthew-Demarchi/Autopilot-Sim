import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

try:
    import torch
    from torch.utils.data import Dataset
except Exception as e:
    torch = None
    Dataset = object

from ursina import application
from panda3d.core import FrameBufferProperties, WindowProperties, GraphicsPipe, Texture, GraphicsOutput
from ursina import Entity
from ursina import time
from ursina import held_keys
from PIL import Image
from datetime import datetime
from pathlib import Path
from queue import Queue
from threading import Thread
from dataclasses import dataclass
from collections import defaultdict
import csv
import cv2
import numpy as np

class OffscreenCameraManager:
    def __init__(self, resolution=(512, 512)):
        self.resolution = resolution
        # NOTE: Map { camera_entity -> (buffer, texture) }
        self.cam_map = {}

        # 1. Set up framebuffer properties (RGBA (red, green, blue, ALPHA) channels + Depth buffer)
        self.fb_props = FrameBufferProperties() #Initilizing the blank frame buffer
        self.fb_props.set_rgba_bits(8, 8, 8, 8)
        self.fb_props.set_depth_bits(24)

        self.win_props = WindowProperties.size(*resolution)
        self.flags = GraphicsPipe.BF_refuse_window  # Invisible offscreen buffer

    def register_camera(self, entity, fov = 100):
        #Creates an offscreen render target
        if entity in self.cam_map:
            return  # Already registered in

        base = application.base

        # Create offscreen buffer
        buf = base.graphicsEngine.make_output(
            base.pipe, f"Offscreen_{entity.name}", 0,
            self.fb_props, self.win_props, self.flags,
            base.win.get_gsg(), base.win
        )

        # Attach texture target that outputs to system ram (from vram)
        # Also, since I can never remember that this can work this way, instead of reading off a texture and displaying,
        #   we're reading the display into a texture
        tex = Texture()
        buf.add_render_texture(tex, GraphicsOutput.RTM_copy_ram, GraphicsOutput.RTP_color)

        # Create camera and parent it whatever the enttiy is
        cam_node = base.make_camera(buf)
        cam_node.reparent_to(entity)

        # --- FIX 1: MATCH FOV AND ASPECT RATIO ---
        lens = cam_node.node().get_lens()
        lens.set_fov(fov)  # Match the camera.fov = 100 used in key '2'
        lens.set_aspect_ratio(self.resolution[0] / self.resolution[1])

        # Store reference
        self.cam_map[entity] = (buf, tex)

    def cleanup(self):
        """Release the offscreen GPU buffers.

        Each register_camera() call allocates a framebuffer through the graphics
        engine; without this, rebuilding the world (as the evaluator does between
        runs) leaks three buffers every time.
        """
        base = application.base
        for entity, (buf, tex) in list(self.cam_map.items()):
            try:
                buf.clear_render_textures()
                base.graphicsEngine.remove_window(buf)
            except Exception:
                pass
        self.cam_map.clear()

    def get_image(self, entity):
        if entity not in self.cam_map:
            raise ValueError(f"Entity {entity} is not registered! Call register_camera first.")

        base = application.base
        buf, tex = self.cam_map[entity]

        if not tex.has_ram_image():
            return None

        # Extract raw binary pixel bytes
        data = tex.get_ram_image()

        # Convert raw BGRA binary bytes from Panda3D to PIL RGBA Image (unflipped for fast main-thread handoff)
        img = Image.frombytes("RGBA", (tex.get_x_size(), tex.get_y_size()), data, "raw", "BGRA")
        return img

def Indentifier():
    return datetime.now().strftime("%Y%m%d-%H%M%S")


# Steering-label offset baked into each camera's rows AT RECORD TIME. The side cameras
# sit +-0.45 units off centre, and the offset is the correction a driver would apply to
# get back to the middle -- it is what teaches the model to recover, since a human
# demonstrator never actually drifts.
#
# DatasetLogger writes labels using these values, and DrivingDataset reads them back so
# it knows what is already baked into the CSVs. Keep the two in sync by keeping them
# here: changing this dict changes what NEW recordings contain, and simultaneously tells
# the loader how to reinterpret OLD ones.
RECORDED_CAMERA_OFFSETS = {"Center": 0.0, "Left": 0.12, "Right": -0.12}


# --- crop bounds -------------------------------------------------------------
# Shared by preprocess() and signSalience() so the salience score is measured over
# exactly the pixels the network is shown -- a sign cropped out of frame must score 0.
CROP_TOP, CROP_BOTTOM, CROP_LEFT, CROP_RIGHT = 80, 74, 79, 58


# --- stop-sign salience ------------------------------------------------------
# Count of saturated-red pixels inside the crop, used as a cheap proxy for "an active
# stop sign is visible, and roughly how close". A StopSign turns lime once you have
# stopped for it, so this drops to ~0 exactly when the car should pull away again --
# which is why gating on it protects the resume behaviour in a way that gating on
# "low speed" would not.
#
# Known false positive: car_colors includes red and maroon, so a red oncoming NPC
# scores too. Uncommon enough to live with, and the weighting it feeds is soft.
SIGN_SALIENCE_FILE = "sign_salience.json"

# Raw counts are heavy-tailed (78.6% zero, median-of-nonzero 102, max 8147), so the
# auxiliary target is compressed: log1p(count) / log1p(2000), clipped to [0, 1]. That
# puts p99 at 0.88 with 0.03% clipping, and -- because log is roughly linear in
# 1/distance here -- makes the target behave like "how close is the sign".
SALIENCE_LOG_SCALE = 2000.0


#salience detects the stop signs, this is was used in the aux output layer to force the model to detect stop signs
#HOWEVER, the model could detect stop signs fine, so this is just used for giving information about the collected data
def signSalience(imageBGR):
    h, w = imageBGR.shape[:2]
    crop = imageBGR[CROP_TOP:h - CROP_BOTTOM, CROP_LEFT:w - CROP_RIGHT]
    b = crop[:, :, 0].astype(np.int16)
    g = crop[:, :, 1].astype(np.int16)
    r = crop[:, :, 2].astype(np.int16)
    return int(((r > 140) & (g < 90) & (b < 90)).sum())


def buildSignSalienceCache(force=False, workers=12):
    """Score every recorded frame once and cache it beside its session CSV.

    Writes Data/<session>/sign_salience.json as {imageFilename: redPixelCount}.
    Purely additive -- no recorded data is modified.
    """
    from concurrent.futures import ThreadPoolExecutor
    import json

    dataLocation = Path(__file__).parent / "Data"
    if not dataLocation.exists():
        print("No Data directory to score.")
        return

    grandTotal = 0
    for folder in sorted(dataLocation.iterdir()):
        if not folder.is_dir():
            continue
        csvFile = folder / (folder.name + " telemtry.csv")
        if not csvFile.exists():
            continue

        cachePath = folder / SIGN_SALIENCE_FILE
        cache = {}
        if cachePath.exists() and not force:
            try:
                cache = json.loads(cachePath.read_text())
            except Exception:
                cache = {}

        names = []
        with open(csvFile, 'r') as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if not row or len(row) < 7:
                    continue
                names.append(row[2].strip())

        todo = [n for n in names if n not in cache]
        if not todo:
            grandTotal += len(names)
            continue

        def score(name):
            img = cv2.imread(str(folder / name))
            return name, (signSalience(img) if img is not None else 0)

        with ThreadPoolExecutor(max_workers=workers) as ex:
            for name, value in ex.map(score, todo):
                cache[name] = value

        cachePath.write_text(json.dumps(cache))
        grandTotal += len(names)
        visible = sum(1 for v in cache.values() if v >= 100)
        print(f"  {folder.name}: scored {len(todo)} new ({len(cache)} total, "
              f"{visible} with a sign clearly in view)")

    print(f"Sign salience cache built over {grandTotal} frames.")


#setting the information that we'll save
@dataclass
class dataToBeSaved:
    image: Image.Image
    speed: int
    speedLimit: int
    steering: float
    pedal: float
    name: str
    frameID: int
    overlay_color: tuple = None

#collect and control the info needed to make a new thread to wait for data to be passed info along for it to actually be recorded
class DatasetLogger(Entity):
    def __init__(self, worldGen, delay, **kwargs):
        # We explicitly pass the position into super() so the entity anchors before calculating children
        super().__init__(**kwargs)
        self.ready = False

        self.worldGen = worldGen
        self.recordData = False

        # DAgger mode. The model's own failures happen in states the human never drove
        # into -- it arrives at signs faster and flatter than any demonstration -- so no
        # amount of extra human driving covers them. Here the model drives and only the
        # frames where the human grabs the controls get logged, which labels exactly
        # those states with the correct action.
        #
        # This must never record while the model alone is driving: under autopilot the
        # car's control values ARE the model's output, so logging them would teach it to
        # imitate itself and cement the failure.
        self.correctionsOnly = False

        self.sessionID = Indentifier()
        self.delay = delay
        self.timer = 0.00

        self.cameras = [self.worldGen.car.windshieldCentered, self.worldGen.car.windshieldLeft, self.worldGen.car.windshieldRight]
        self.cameraNames = ["Center", "Left", "Right"]
        self.cameraWheelOffsets = [RECORDED_CAMERA_OFFSETS[n] for n in self.cameraNames]

        self.prgmDir = Path(__file__).parent
        self.dataLocation = self.prgmDir / "Data"

        if not self.dataLocation.exists():
            print("\"Data\" Directory doesn't exist, creating it")
            self.dataLocation.mkdir(parents=True)
        else:
            print("\"Data\" Directory exists")

        # Created lazily on the first captured frame, not here. Every launch of the sim
        # builds a DatasetLogger whether or not you ever press record, so creating the
        # folder up front left a trail of empty "Session ..." directories -- there were
        # 14 of them from one evening. No frames, no folder.
        self.sessionDataDir = self.dataLocation / f"Session {self.sessionID}"

        self.ready = True
        self.createdFolder = False

        self.imageNumber = 0

        self.queue = Queue()

        self.savingThread = Thread(target=self.saveData, daemon=True)
        self.savingThread.start()



    def toggle(self):
        self.recordData = not self.recordData

    def toggleCorrectionsOnly(self):
        self.correctionsOnly = not self.correctionsOnly
        return self.correctionsOnly

    @staticmethod
    def humanIsDriving():
        return bool(held_keys['w'] or held_keys['a'] or held_keys['s'] or held_keys['d'])




    #only record data every delay seconds (debating between 10hz and 20hz)
    def update(self):
        if self.recordData and self.ready:
            self.timer += time.dt
            if self.timer >= self.delay:
                self.timer = 0.0  # Reset timer cleanly to prevent catch-up lag spirals

                # In DAgger mode, capture only while the human is actually correcting.
                # Note the steering label on these frames is whatever the car is doing,
                # so if you are only braking you are implicitly endorsing the model's
                # steering for that frame -- fine while steering is good, worth
                # remembering if it ever is not.
                if self.correctionsOnly and not self.humanIsDriving():
                    return

                # Single GPU render pass so all 3 offscreen cameras capture exact current frame telemetry
                application.base.graphicsEngine.render_frame()

                overlay_col = None
                if hasattr(self.worldGen, 'weather') and self.worldGen.weather and self.worldGen.weather.mode != 'clear':
                    overlay_col = getattr(self.worldGen.weather, 'overlay_color', None)

                for i in range(3):
                    raw_img = self.worldGen.camera_mgr.get_image(self.cameras[i])
                    if raw_img is None:
                        continue
                    frameOfData = dataToBeSaved(
                        image=raw_img,
                        speed=self.worldGen.car.returnVelocity(),
                        speedLimit=self.worldGen.returnSpeedLimit(),
                        steering=self.worldGen.car.returnSteeringAngle() / 540 + self.cameraWheelOffsets[i],  # normalized
                        pedal=self.worldGen.car.returnAccelerationPedal() - self.worldGen.car.returnBreakPedal(),
                        name=self.cameraNames[i],
                        frameID=self.imageNumber,
                        overlay_color=overlay_col
                    )
                    self.queue.put(frameOfData)
                self.imageNumber += 1


                #make sure that this all works, then save to csv with image and image number then imagenumber++

    #running on different thread for lag reaasons
    def saveData(self):
        csvPath = self.sessionDataDir / f"Session {self.sessionID} telemtry.csv"

        # Wait for the first frame before touching the filesystem, so a session that
        # never records leaves nothing behind.
        first = self.queue.get()
        if first is None:
            self.queue.task_done()
            return

        self.sessionDataDir.mkdir(parents=True, exist_ok=True)
        self.createdFolder = True

        #append mode csv
        with open(csvPath, mode='a', newline='') as csvfile:
            writer = csv.writer(csvfile)

            writer.writerow(["frame", "camera", "image_path", "speed", "speed_limit", "steering", "pedal"])

            tempData = first
            while True:
                if tempData is None:
                    self.queue.task_done()
                    break

                imageFilename = f"{self.sessionID} {tempData.name} frame_{tempData.frameID:06d}.jpg"
                imageFilepath = self.sessionDataDir / imageFilename

                # Perform image processing off the main thread in background thread:
                # 1. Flip OpenGL Y-axis so image is right-side up
                tempImageForConv = tempData.image.transpose(Image.FLIP_TOP_BOTTOM)

                # 2. Apply weather overlay tint if active
                if tempData.overlay_color:
                    tint_img = Image.new('RGBA', tempImageForConv.size, tempData.overlay_color)
                    tempImageForConv = Image.alpha_composite(tempImageForConv, tint_img)

                # 3. Convert RGBA to RGB for JPEG format
                if tempImageForConv.mode in ("RGBA", "LA", "P"):
                    tempImageForConv = tempImageForConv.convert("RGB")

                # 4. Save as JPEG with 90% quality compression
                tempImageForConv.save(imageFilepath, format="JPEG", quality=90)

                writer.writerow([
                    tempData.frameID,
                    tempData.name,
                    imageFilename,
                    tempData.speed,
                    tempData.speedLimit,
                    tempData.steering,
                    tempData.pedal
                ])

                csvfile.flush() #flush to avoid data loss

                self.queue.task_done()
                tempData = self.queue.get()


    def shutdown(self):
        self.recordData = False
        self.ready = False

        #NOTE: STOP EXTRA THREAD
        self.queue.put(None)


        self.queue.join()

        #Join the background thread -- good catch from gemini
        if hasattr(self, 'savingThread') and self.savingThread.is_alive():
            self.savingThread.join()

        print("DatasetLogger successfully shut down.")



#collect all the data recording to build the dataset for training
#think I should add something that would let you select the data later, but should fix the training pipeline first
class DrivingDataset(Dataset):
    # cameras: which rigs to pull frames from. Defaults to all three.
    #          Pass ("Center",) to drop the synthetic side-camera labels entirely.
    #
    # steering_offsets: per-camera offset you WANT, e.g. {"Center":0, "Left":.02, "Right":-.02}.
    #          None keeps whatever is already in the CSVs. Any other value is applied as a
    #          delta against RECORDED_CAMERA_OFFSETS while rows are read into memory --
    #          nothing on disk is touched, so the recorded data stays the single source of
    #          truth and you can sweep this per training run. Once a value is settled on,
    #          change RECORDED_CAMERA_OFFSETS so new recordings use it directly and this
    #          remap becomes a no-op for those sessions.
    def __init__(self, cameras=("Center", "Left", "Right"), steering_offsets=None):
        self.cameras = tuple(cameras)

        if steering_offsets is None:
            self.offset_fix = {}
        else:
            self.offset_fix = {
                cam: float(steering_offsets.get(cam, 0.0)) - RECORDED_CAMERA_OFFSETS.get(cam, 0.0)
                for cam in RECORDED_CAMERA_OFFSETS
            }
            if any(abs(v) > 1e-9 for v in self.offset_fix.values()):
                shown = {c: round(v, 4) for c, v in self.offset_fix.items()}
                print(f"Steering offsets remapped at load time (files unchanged): {shown}")

        self.augment = False
        self.prgmDir = Path(__file__).parent
        self.dataLocation = self.prgmDir / "Data"
        self.image_paths = []
        self.steering_angles = []
        self.speeds = []
        self.speedLimits = []
        self.pedals = []
        self.camera_names = []   # per-row "Center"/"Left"/"Right", for center-only validation
        self.signSalience = []   # per-row red-pixel count; 0 when the cache is absent
        self.reactionLag = []    # True on frames that only exist because humans are slow
        self._missingSalience = 0

        if self.dataLocation.exists():
            for folder in self.dataLocation.iterdir():
                if folder.is_dir():
                    print(f"Found folder: {folder.name}")
                    csvFile = folder / (folder.name + " telemtry.csv")
                    imgDir = folder

                    salienceCache = {}
                    saliencePath = folder / SIGN_SALIENCE_FILE
                    if saliencePath.exists():
                        try:
                            import json
                            salienceCache = json.loads(saliencePath.read_text())
                        except Exception:
                            salienceCache = {}

                    if csvFile.exists():
                        sessionStart = len(self.image_paths)
                        with open(csvFile, 'r') as f:
                            reader = csv.reader(f)
                            next(reader, None)  # Skip CSV header row
                            for row in reader:
                                if not row or len(row) < 7:
                                    continue
                                cameraName = row[1].strip()
                                if cameraName not in self.cameras:
                                    continue
                                imgPath = str(imgDir / row[2].strip())
                                speed = float(row[3])
                                speedLimit = float(row[4])
                                angle = float(row[5]) + self.offset_fix.get(cameraName, 0.0)
                                pedal = float(row[6])

                                self.image_paths.append(imgPath)
                                self.steering_angles.append(angle)
                                self.speeds.append(speed)
                                self.speedLimits.append(speedLimit)
                                self.pedals.append(pedal)
                                self.camera_names.append(cameraName)

                                imgName = row[2].strip()
                                if imgName in salienceCache:
                                    self.signSalience.append(salienceCache[imgName])
                                else:
                                    self.signSalience.append(0)
                                    self._missingSalience += 1

                                self.reactionLag.append(False)

                        self._markReactionLag(sessionStart, len(self.image_paths))

        if self._missingSalience:
            print(f"NOTE: {self._missingSalience} of {len(self.image_paths)} frames have no "
                  f"sign-salience score (treated as 0). Run:  python train.py --build-salience")

    # Salience below this means the stop sign has turned lime; above it, still red.
    GREEN_SALIENCE = 200
    LAG_MAX_FRAMES = 60      # safety stop, ~6s at 10Hz

    def _markReactionLag(self, start, end):
        """Flag frames between a sign turning green and the driver actually moving.

        Measured over 280 stop events: the sign flips, and for a median of 0.20s the car
        is still stopped with no throttle while the human reacts. That is 588 center
        frames showing "green sign, stopped, do nothing" -- against only ~825 frames in
        the whole training split that say "go". They very nearly cancel.

        The labels are not wrong; the driver really was not pressing yet. But reaction
        time is an artifact of the demonstrator, not behaviour worth copying -- a network
        has none. Cloning it teaches hesitation in exactly the state where the car should
        pull away, which is why resuming has been unreliable.

        Rows are interleaved Center/Left/Right, so each camera is walked separately.
        """
        byCamera = defaultdict(list)
        for i in range(start, end):
            byCamera[self.camera_names[i]].append(i)

        for _, idx in byCamera.items():
            k = 1
            while k < len(idx):
                i = idx[k]
                prev = idx[k - 1]
                # moment of coming to rest with a red sign filling the view
                if (abs(self.speeds[i]) < 0.5 and abs(self.speeds[prev]) >= 0.5
                        and self.signSalience[i] > 400):
                    j = k
                    # sit while the sign stays red
                    while (j < len(idx) - 1 and abs(self.speeds[idx[j]]) < 0.5
                           and self.signSalience[idx[j]] > self.GREEN_SALIENCE):
                        j += 1
                    # from the flip until the driver actually commits
                    if (j < len(idx) - 1
                            and self.signSalience[idx[j]] <= self.GREEN_SALIENCE
                            and abs(self.speeds[idx[j]]) < 0.5):
                        m = j
                        while (m < len(idx) - 1 and m - j < self.LAG_MAX_FRAMES
                               and abs(self.speeds[idx[m]]) < 2.0
                               and self.pedals[idx[m]] <= 0.1):
                            self.reactionLag[idx[m]] = True
                            m += 1
                        k = m
                        continue
                k += 1

    def ToggleAugment(self):
        self.augment = not self.augment

    def __len__(self):
        return len(self.image_paths)

    # Augmentation knobs. Horizontal flip is deliberately NOT here: it mirrors the
    # drive-on-the-right convention, which is why it hurt when it was tried before.
    AUG_CUTOUT_PROB = 0.3    # drop to 0.0 if stop-sign behaviour degrades -- the sign
                             # is only ~25px, so a patch can erase it entirely

    # Jitter on the speed INPUT (never the label).
    #
    # At speed < 0.5 the data is 15.2:1 "holding at a red sign" versus "pulling away", and
    # speed arrives at lin1 as a clean scalar -- far cheaper to partition on than a red-
    # versus-lime patch that has to survive three strided convs. So the net carves a rule
    # at zero: with the image held fixed, moving the speed input 0 -> 2 swings its pedal
    # output by +0.875. The car will not start itself, at a green sign or at spawn.
    #
    # Blurring the input across that boundary makes "reads as stopped" unreliable, so the
    # only way to tell hold from go is the sign itself -- which the trunk already encodes
    # (linear probe R^2 = 0.91). Training only; validation stays exact.
    AUG_SPEED_JITTER = 1.5   # mph, gaussian; 0.0 disables

    def __getitem__(self, index):
        image = cv2.imread(str(self.image_paths[index]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.augment:
            image = self.augmentRaw(image)  # photometric jitter on the uint8 frame

        image = self.preprocess(image)  # your crop/resize/normalize function from earlier

        steering = self.steering_angles[index]

        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()

        speedValue = self.speeds[index]
        if self.augment and self.AUG_SPEED_JITTER:
            speedValue = max(0.0, speedValue + np.random.normal(0.0, self.AUG_SPEED_JITTER))

        speed_tensor = torch.tensor([speedValue], dtype=torch.float32)
        speed_limit_tensor = torch.tensor([(self.speedLimits[index])], dtype=torch.float32)
        # Third element is the auxiliary target. Kept inside the same tensor rather than
        # returned separately so every existing `image, speed, limit, target = ds[i]`
        # unpack keeps working, and target[0]/target[1] still mean what they did.
        salience = float(np.log1p(self.signSalience[index]) / np.log1p(SALIENCE_LOG_SCALE))
        target_tensor = torch.tensor([steering, self.pedals[index], min(1.0, salience)],
                                     dtype=torch.float32)

        return image_tensor, speed_tensor, speed_limit_tensor, target_tensor

    @classmethod
    def augmentRaw(cls, image):
        """Label-preserving photometric jitter. Applied to the uint8 RGB frame before
        cropping, so it never changes what the correct steering/pedal value is."""
        # contrast + brightness
        alpha = 1.0 + np.random.uniform(-0.25, 0.25)
        beta = np.random.uniform(-25, 25)
        image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

        # gamma
        if np.random.rand() < 0.5:
            gamma = 1.0 + np.random.uniform(-0.3, 0.3)
            lut = np.clip(((np.arange(256) / 255.0) ** (1.0 / gamma)) * 255.0, 0, 255).astype(np.uint8)
            image = cv2.LUT(image, lut)

        # cutout -- forces the net to spread its evidence instead of latching onto
        # one memorised patch of a specific training frame
        if np.random.rand() < cls.AUG_CUTOUT_PROB:
            h, w = image.shape[:2]
            for _ in range(np.random.randint(1, 4)):
                ph, pw = np.random.randint(10, 40), np.random.randint(10, 40)
                y = np.random.randint(0, max(1, h - ph))
                x = np.random.randint(0, max(1, w - pw))
                image[y:y + ph, x:x + pw] = np.random.randint(0, 256)

        return image

    #process the image before sending it for training, resizing specifically
    @staticmethod
    def preprocess(image):
        h, w = image.shape[:2]
        image = image[CROP_TOP: h - CROP_BOTTOM, CROP_LEFT: w - CROP_RIGHT, :]
        image = cv2.resize(image, (375, 358))  # match the crop's natural height -- no more squish
        image = image.astype(np.float32) / 127.5 - 1.0
        return image

    #broken version
    # @staticmethod
    # def preprocess(image):
    #     # 1. Crop using exact pixel cutoffs:
    #     # NumPy slicing syntax: image[start_y:end_y, start_x:end_x, :]
    #     # - Top cutoff: 178 px
    #     # - Bottom cutoff: 74 px off bottom -> image.shape[0] - 74
    #     # - Left cutoff: 79 px
    #     # - Right cutoff: 58 px off right -> image.shape[1] - 58
    #
    #     h, w = image.shape[:2]
    #     image = image[178: h - 74, 79: w - 58, :]
    #
    #     # 2. Resize to your target network input size (Width, Height)
    #     image = cv2.resize(image, (375, 260))
    #
    #     # 3. Normalize pixel values from [0, 255] to roughly [-1.0, 1.0]
    #     image = image.astype(np.float32) / 127.5 - 1.0
    #
    #     return image

#NOTE FOR SHUTDOWN FUNCTION, PUT "None" into the Queue to shut down the access thread