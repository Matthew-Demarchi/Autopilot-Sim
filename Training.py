#NOTE TO SELF: large parts of this file were cleaned up by ai, OriginalTraining.py was the before file, this is cleaner but I recognize my structure better
import os
import random
from collections import defaultdict
import copy

import numpy as np

from torch.optim.lr_scheduler import ReduceLROnPlateau

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Ensure PyTorch CUDA C++ DLLs are in PATH and DLL search directory on Windows
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

import torch
from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, random_split

import DataCollection


#HISTORY NOTES
"""
Tried first with just the 5 convolution layers, and 4 linear layers
    this wasn't great, and the models we're kind of large and took too long to train
    Also lead to massive overfitting
    
Tried adding a adaptiveAvgPool with 4, 8 between conv and linear
    helped but caused losses in recognition

Swapped the adaptiveAvgPool for a adaptiveMaxPool
    Way better, has good stop sign (actually recognizing them), pedestrian, and fantastic lane marking detection
    
MODEL NOTE: can't seem to stop for stop signs, it only slows down at the very very last second
    
Changed resolution of the MaxPool to be (8, 16) instead of (4, 8)
    HUGE performance increase, much more delicate control, though still isn't slowing down quick enough, but better
    Mostly an improvement to other areas, but slight stop sign improvement
    
Added droppout layers
    Can't really tell if that helped or not, seems to have gotten worse, unexpected
    
Added auxillary layers
    same here
    
Adding a sampler to up the data used for stop signs
    That worked well, not perfect yet, but a huge step in the right direction
    I'm settling on these weights:
        brake = 2.2
        Resume = 3
        Stop = 2.1
    
Removed dropout and auxillary layers

Split evaluation with validation frames (80/20)

COULDN'T REMEMBER THE CHANGE HERE SO I HAD IT GENERATED, it remembers what happened better than i do:
Steering was hedging, wouldn't actually commit to turns, just kind of mushed toward straight
    Found out the loss was something like 91% pedal
    Steering is wheelAngle/540 so it only ever uses ~7% of its range, but pedal already uses
    all of [-1,1], so pedal had ~10x the label variance and was eating basically all the gradient
    Weighted the steering term up by the variance ratio so both matter the same
    Calculated from the data instead of a magic number so it fixes itself if the data changes
    Big improvement, this is what got it committing to real turns instead of hedging
    ALSO: had to stop scoring val on the left/right cameras, inference only uses center, and the
    offsets are easy to spot so it was scoring well by guessing which camera it was, not by steering

Can't get the model to stop for stop signs

tweaked a ton of model parameters, everything made it worse so reverting

Had a constant eval made so I have physical numbers and not just general behavior
    Running a bunch of tests with that

WHY ISN'T IS STOPPING, SPECIFICALLY AdDED A BUNCH OF STOPSIGN DATA AND IT'S NOT ANY BETTER
??????????????????????????

Claude suggested trying the auxillary layer again, but as an output instead of and input, grok agrees

Added an auxiliary layer head to try and force the model to start recognizing stop signs
    (different from the previous one which acted as an input to try and give the model helpful data)
    For my remembering, it's adding an extra gradient to force the model to not zone out the stop signs
    However, after some further testing, the program's sign detection is working fine
    
Disabled the auxilary layer
    The model sees the signs fine, but just isn't stopping???   
    

KMS after two days grok found I had blue and red swapped, claude confirms it
    Evals from the aux showed it was still seeing it, the problem was in inference
    
Wow would you look at that, so much better, still not perfect but at least I can go from here


Let me add back in the dropout layer
    seemed to help a bit, played with the parameter a bit but .3 seems to work pretty good

    
The model is still having a hard time coming to a full stop
    
Lets add correcting the model
    Apparently this is making an aggregated dataset (DAgger)
    
Worked really well

    
Try a few different architectural changes, like making to seperate output heads ,but none of them really panned out so reverted

I feel like it's 99% there

Having a hard time getting everything perfect, where the steering is perfect, with the stop sign handling.
have a model that is great at steering, but isn't perfect at stop signs, and a model that's perfect at stop signs, but can't do steering as well

Problem, the steering behavior converges quickly, but behavior with this stop signs and pedestrians takes a longer time
    Ending training early causes problems with stop signs
    Letting training continue causes steering to regress
    
Played with the sampler weights, didn't help, reverted
    specifically upped them, but that flooded the data too much and caused massive regressions in steering

ran more evals with all the models I've made with the testing

Grok suggested moving the sampler's role to the loss function
    Claude also mentioned something similar
    Argument is that this would allow for similar importance to be placed on the pedals, without decreasing or corrupting
    the steering data, preventing the regressions
    
Bounced off a few ideas with claude and grok, we landed on the current criterion function
    Performance is significantly better
    
deactivated the sampler and shift fully to this new system using the updated loss function
    (NEW FUNCTIONS: criterion and pedalEmphasis) STUDY THE MATH HERE MORE, I get it but... ish


#AI drafted this one, forgot it in my notes
Resuming on green still wasn't reliable, kept having to tap W myself
    Turns out when the sign flips green there's about 0.2s where I'm stopped and haven't hit
    the gas yet, just reacting
    Those frames are green sign + stopped + no throttle, ~590 of them, against only ~825 frames
    in the whole split that actually say GO. They pretty much cancel each other out
    The labels aren't even wrong, I really wasn't pressing yet, it's just my reaction time
    But the model doesn't HAVE reaction time, so copying it just teaches it to sit there
    Dropped those frames from training only (val keeps them so old baselines still compare)
    This is the thing that actually fixed resuming, it's way quicker off the line now


Model's performance is overall fantastic

Swap to non-constant weights, calculated based off the data  

I think, at this point, any errors with the model are purely from lack of data (poorer recovery, sometimes has a bad turn
    at 55mph, etc.)
    
Going to call the training finished as I think this is about how good it will get without a major rehaul
"""


class Net(nn.Module):


    def __init__(self, hidden=512):
        super().__init__()

        #borrowing kernal size 5,5,5,3,3 and stride 2,2,2,1,1 from PilotNet paper
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(32, 48, kernel_size=5, stride=2)
        self.conv4 = nn.Conv2d(48, 64, kernel_size=3, stride=1)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.pool = nn.AdaptiveMaxPool2d((8, 16))   # <- new: shrink to a fixed 4x8 grid regardless of input res -- I feel like this is too low ima change it to 8 16. NOTE: yeah, way better

        self.drop_feat = nn.Dropout(0.3)   # on the flattened conv features
        self.drop_fc = nn.Dropout(0.3)     # between the fully-connected layers

        # Auxiliary supervision: predict how visible/close an active stop sign is, from
        # the conv features ALONE. It sits before speed/speed_limit are concatenated, so
        # it structurally cannot reach them -- the only way to lower its loss is for the
        # conv stack to genuinely encode stop signs. Ignored at inference.
        self.aux_head = nn.Linear(64 * 8 * 16, 1)

        # This one layer is where nearly all the capacity lives -- at hidden=512 it is
        # 4.19M of the model's 4.35M parameters, against 226k training frames. That ratio
        # is why steering overfits within a single epoch.
        self.lin1 = nn.Linear(64 * 8 * 16 + 2, hidden)

        self.lin2 = nn.Linear(hidden, 64)
        self.lin3 = nn.Linear(64, 2)

        # self.lin1 = nn.Linear(64002, 2048) #that 64000 + 2 for speed and speed limit
        # self.lin2 = nn.Linear(2048, 512)
        # self.lin3 = nn.Linear(512, 64)
        # self.lin4 = nn.Linear(64, 2)

    def forward(self, image, speed, speed_limit, return_aux=False):
        x = F.relu(self.conv1(image))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        # print(x.flatten(start_dim=1).shape) -- NOTE size = [1,64000] at this point
        x = self.pool(x)

        x = x.flatten(start_dim=1)

        # Read the aux head off the clean (un-dropped) features -- its job is to shape
        # the trunk, so it wants an undisturbed gradient.
        aux = self.aux_head(x).squeeze(1)

        # Dropout on the image features only -- applying it after the cat would also
        # randomly zero speed/speed_limit, which are raw magnitudes, not activations.
        x = self.drop_feat(x)
        x = torch.cat((x, speed, speed_limit), dim=1)

        x = F.relu(self.lin1(x))
        x = self.drop_fc(x)
        x = F.relu(self.lin2(x))
        x = self.lin3(x)

        if return_aux:
            return x, aux


        # x = F.relu(self.lin1(x))
        # x = F.relu(self.lin2(x))
        # x = F.relu(self.lin3(x))
        # x = self.lin4(x)
        return x #?


def modelForCheckpoint(state, device=None):
    """Build a Net matching the width the checkpoint was saved with.

    Shrinking lin1 changes its shape, so without this an older wide checkpoint would load
    into a narrow Net under strict=False, silently keeping randomly initialised weights
    for the layer holding ~96% of the model, and drive like noise. Reading the width off
    the checkpoint keeps every model you have ever trained loadable.
    """
    hidden = state["lin1.weight"].shape[0] if "lin1.weight" in state else 512
    model = Net(hidden=hidden)
    if device is not None:
        model = model.to(device)
    return model, hidden


# Module level so the UI can show the real number instead of hardcoding one that would
# quietly drift out of sync if this ever changed.
EARLY_STOP_PATIENCE = 7


def _seed_worker(worker_id):
    """Give every DataLoader worker its own RNG stream.

    Without this, workers can inherit an identical numpy seed and generate the exact
    same augmentation sequence, silently collapsing augmentation diversity.
    """
    seed = (torch.initial_seed() + worker_id) % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


def train(model, dataloader, criterion, optimizer, device, onStep=None, onBatch=None):
    model.train()
    totalBatches = len(dataloader)
    batchIndex = 0
    running_loss = 0.0
    running_steer = 0.0   # unweighted MSE, so it stays comparable to the val numbers
    running_pedal = 0.0
    n = 0

    for images, speeds, speedLimits, targets in dataloader:

        images = images.to(device, non_blocking=True)
        speeds = speeds.to(device, non_blocking=True)
        speed_limits = speedLimits.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()

        outputs, aux = model(images, speeds, speed_limits, return_aux=True)

        loss = criterion(outputs, aux, targets, speeds)

        loss.backward()

        optimizer.step()
        if onStep is not None:
            onStep()

        batchIndex += 1
        if onBatch is not None:
            onBatch(batchIndex, totalBatches)

        batch = images.size(0)
        running_loss += loss.item() * batch
        with torch.no_grad():
            running_steer += F.mse_loss(outputs[:, 0], targets[:, 0], reduction='sum').item()
            running_pedal += F.mse_loss(outputs[:, 1], targets[:, 1], reduction='sum').item()
        n += batch

    return running_loss / n, running_steer / n, running_pedal / n

def trainStart(epochs, num_workers=0, progress=None):
    """num_workers > 0 requires a __main__ module that is safe to re-import, because
    Windows spawns workers by re-running it. Launch via train.py for that -- calling
    this from Autopilot_Menu.py must stay at 0, since importing that module builds an
    Ursina app."""
    # --- EXPERIMENT SWITCH ---------------------------------------------------
    # ("Center", "Left", "Right") = original behaviour.
    # ("Center",)                 = center-only diagnostic. Drops the synthetic
    #                               +-0.12 steering offsets that account for ~67%
    #                               of total steering label variance.
    CAMERAS = ("Center", "Left", "Right")

    # Side-camera steering offset, remapped at load time (the CSVs are never modified).
    #   0.12 = exactly as recorded, i.e. no change
    #   None = disable the remap entirely
    #
    # MEASURED: 0.02 was tried and drives much worse -- wanders out of the lane -- even
    # though it beat 0.12 on every offline metric (val_steer 0.00300 vs 0.00468, straight
    # -frame deviation 14.9 vs 26.3 deg). This offset is not just a label: it is the
    # PROPORTIONAL GAIN of the recovery controller, 0.12/0.45 units of lateral error.
    # Dropping it to 0.02 cut lane-keeping authority ~6x. Do not lower it on the strength
    # of offline metrics alone -- judge it by driving.
    SIDE_CAMERA_OFFSET = 0.12
    STEERING_OFFSETS = (None if SIDE_CAMERA_OFFSET is None else
                        {"Center": 0.0, "Left": SIDE_CAMERA_OFFSET, "Right": -SIDE_CAMERA_OFFSET})

    # Relative priority of the steering head in the loss. None = auto-balance to
    # (pedal label variance / steering label variance) on the training split, which
    # makes the two heads exactly equal-priority no matter how the data changes.
    # Set a float to override. Must be revisited by hand if you ever hardcode it --
    # changing the camera offset changes the steering variance and hence this number.
    STEER_WEIGHT = None
    # -------------------------------------------------------------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(str(device))

    # lin1 holds ~96% of the weights, so this single number is the model's capacity.
    # At 512 that is 4.19M params against 226k frames, and steering overfits inside one
    # epoch (overfit gap +0.69 at epoch 1, val_steer already above its baseline).
    # 256 halves it. Older checkpoints keep their own width via modelForCheckpoint.
    HIDDEN_WIDTH = 512

    model = Net(hidden=HIDDEN_WIDTH).to(device)
    print(f"Model: lin1 width {HIDDEN_WIDTH} -> "
          f"{sum(p.numel() for p in model.parameters()):,} params")
    optimizer = optim.AdamW(model.parameters(), lr=0.0003, weight_decay=1e-4)

    # 2. Learning Rate Scheduler (Slows down LR if validation loss plateaus)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    full_dataset = DataCollection.DrivingDataset(cameras=CAMERAS, steering_offsets=STEERING_OFFSETS)
    print(f"Cameras in use: {CAMERAS}  ->  {len(full_dataset)} frames")
    if len(full_dataset) == 0:
        print("No training data found in Data directory! Record dataset before training.")
        return

    # 3. Validation Split (80% Train, 20% Validation)

    CHUNK_SIZE = 300  # consecutive frames per chunk

    session_frames = defaultdict(list)
    for i, path in enumerate(full_dataset.image_paths):
        session_frames[Path(path).parent.name].append(i)

    chunks = []
    for session, indices in session_frames.items():
        indices = sorted(indices)  # keep them in original (roughly time-ordered) sequence
        for start in range(0, len(indices), CHUNK_SIZE):
            chunks.append(indices[start:start + CHUNK_SIZE])

    # Fixed seed: without one, every run got a different train/val split, so run-to-run
    # metric changes were partly just resampling noise. Keep this constant across
    # experiments -- only change it to check a result is not a fluke of one split.
    SPLIT_SEED = 1234
    random.Random(SPLIT_SEED).shuffle(chunks)
    split_point = max(1, int(len(chunks) * 0.8))
    train_indices = [i for chunk in chunks[:split_point] for i in chunk]
    val_indices = [i for chunk in chunks[split_point:] for i in chunk]

    # Drop human reaction-lag frames from TRAINING only. These sit between a stop sign
    # turning green and the driver actually moving: stopped car, green sign, no throttle.
    # They are honest recordings of a slow human, but a network has no reaction time, so
    # copying them just teaches it to hesitate exactly when it should pull away.
    # Validation keeps them, so val baselines stay comparable with every earlier run.
    lagFlags = getattr(full_dataset, "reactionLag", None)
    if lagFlags:
        before = len(train_indices)
        train_indices = [i for i in train_indices if not lagFlags[i]]
        dropped = before - len(train_indices)
        if dropped:
            print(f"Reaction-lag frames dropped from training: {dropped} "
                  f"({100.0 * dropped / before:.2f}% of the split)")

    # --- VALIDATE ON CENTER FRAMES ONLY --------------------------------------
    # run_autopilot_step() only ever feeds windshieldCentered, so Left/Right frames
    # measure a task the car never performs -- and because the +-cameraWheelOffsets
    # are easy to read off near-field parallax, including them lets the model score
    # well by doing 3-way camera ID instead of steering. Training still uses all
    # cameras; only scoring is restricted. Keeps every run comparable.
    if "Center" in CAMERAS:
        val_indices = [i for i in val_indices if full_dataset.camera_names[i] == "Center"]
    else:
        print("WARNING: 'Center' not in CAMERAS -- validating on non-deployment cameras.")

    if len(val_indices) == 0:
        print("Validation split is empty! Lower CHUNK_SIZE or record more sessions.")
        return
    # -------------------------------------------------------------------------

    # --- PREDICT-THE-MEAN BASELINE -------------------------------------------
    # A model that learns nothing about a target scores exactly that target's
    # variance on val. If val_steer below is not well under baseline_steer, the
    # steering head has learned nothing -- regardless of what the blended loss says.
    _val_steer_labels = np.array([full_dataset.steering_angles[i] for i in val_indices])
    _val_pedal_labels = np.array([full_dataset.pedals[i] for i in val_indices])
    baseline_steer = float(_val_steer_labels.var())
    baseline_pedal = float(_val_pedal_labels.var())
    baseline_steer_mae = float(np.abs(_val_steer_labels - _val_steer_labels.mean()).mean())
    print(f"Validation: {len(val_indices)} Center frames | Training: {len(train_indices)} frames (all of {CAMERAS})")
    print(f"BASELINE (always predict the mean): "
          f"val_steer={baseline_steer:.5f} (MAE {baseline_steer_mae * 540:.1f} deg) | "
          f"val_pedal={baseline_pedal:.5f}")

    # --- LOSS BALANCE --------------------------------------------------------
    # Plain MSE over the 2-vector sums squared error in whatever units it is handed.
    # steering is wheelAngle/540 (540 deg is full lock, so normal driving uses ~7% of
    # the range) while pedal already spans [-1,1] and uses all of it -- so pedal
    # carries ~10x the label variance and therefore ~91% of the gradient. Weighting
    # by the variance ratio makes the two heads equal-priority.
    _tr_steer = np.array([full_dataset.steering_angles[i] for i in train_indices])
    _tr_pedal = np.array([full_dataset.pedals[i] for i in train_indices])
    train_steer_var = float(_tr_steer.var())
    steer_weight = float(_tr_pedal.var() / _tr_steer.var()) if STEER_WEIGHT is None else float(STEER_WEIGHT)
    print(f"Loss balance: steer x{steer_weight:.2f}, pedal x1.00 "
          f"(train label var: steer={_tr_steer.var():.5f}, pedal={_tr_pedal.var():.5f})")

    # --- auxiliary task weight -----------------------------------------------
    # Scaled off the pedal head so it is a fixed FRACTION of it regardless of how the
    # data changes. Deliberately secondary: predicting salience is a means of forcing
    # the trunk to encode stop signs, not something we care about at inference.
    # MEASURED: left OFF. A linear probe on Great Model 3's frozen conv features already
    # predicts salience at R^2 = 0.91 (corr 0.96), accurate even for distant signs, and
    # the pedal head already reproduces 94% of the human's sign/no-sign gap. The feature
    # exists and is being used -- there is nothing for aux supervision to force into
    # existence. Kept wired up (and free at 0.0) in case a future architecture needs it.
    AUX_PRIORITY = 0.0
    _tr_sal = np.array([min(1.0, np.log1p(full_dataset.signSalience[i])
                            / np.log1p(DataCollection.SALIENCE_LOG_SCALE))
                        for i in train_indices])
    aux_weight = (AUX_PRIORITY * _tr_pedal.var() / max(_tr_sal.var(), 1e-9)) if AUX_PRIORITY else 0.0
    print(f"Aux head: salience target var={_tr_sal.var():.5f}, weight x{aux_weight:.2f} "
          f"({AUX_PRIORITY:.0%} of the pedal head)")

    # --- where the pedal emphasis is applied ---------------------------------
    # MEASURED: putting stop/resume/brake emphasis in the SAMPLER fixes stopping and
    # wrecks steering. Over 5 runs of 600s on a dense course, x1.5 sampler weights took
    # stopping from 88.4% to 98.9% of signs (n=159, Fisher p=0.011) but departures from
    # 0.05/min to 1.37/min -- 27x worse.
    #
    # The cause is structural: one sampler feeds both heads, so oversampling brake and
    # at-rest frames from 27% to 36% of every batch silently reshapes what the STEERING
    # head trains on too. Steering never asked for that.
    #
    # Applying the same emphasis to the pedal loss term instead decouples them. The
    # sampler goes back to a distribution steering is happy with, while the pedal head
    # still sees those frames weighted up. Set False to restore the old sampler behaviour.
    PEDAL_WEIGHTS_IN_LOSS = True

    def pedalEmphasis(speeds, pedals):
        """Same rule the sampler used, as a per-sample multiplier."""
        s = speeds.abs().squeeze(-1) if speeds.dim() > 1 else speeds.abs()
        w = torch.ones_like(pedals)
        w = torch.where(s < 0.5, torch.full_like(w, STOP_WEIGHT), w)
        w = torch.where((s < 0.5) & (pedals > 0.1), torch.full_like(w, RESUME_WEIGHT), w)
        w = torch.where((s >= 0.5) & (pedals < -0.1), torch.full_like(w, BRAKE_WEIGHT), w)
        return w

    def criterion(outputs, aux, targets, speeds):
        steerTerm = F.mse_loss(outputs[:, 0], targets[:, 0])

        if PEDAL_WEIGHTS_IN_LOSS:
            w = pedalEmphasis(speeds, targets[:, 1])
            # normalised by mean weight so the pedal term keeps the same overall scale
            # and stays balanced against steering
            pedalTerm = (w * (outputs[:, 1] - targets[:, 1]) ** 2).mean() / w.mean()
        else:
            pedalTerm = F.mse_loss(outputs[:, 1], targets[:, 1])

        loss = steer_weight * steerTerm + pedalTerm
        if aux_weight:
            loss = loss + aux_weight * F.mse_loss(aux, targets[:, 2])
        return loss
    # -------------------------------------------------------------------------

    # Shallow copy so ToggleAugment rebinds .augment on the copy only -- val keeps
    # using full_dataset and therefore stays un-augmented.
    train_view = copy.copy(full_dataset)
    train_view.ToggleAugment()
    print(f"Train augmentation: {train_view.augment} | Val augmentation: {full_dataset.augment}")

    from torch.utils.data import Subset
    train_dataset = Subset(train_view, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    if len(train_dataset) == 0:
        print("No training data found in Data directory! Record dataset before training.")
        return

    from torch.utils.data import WeightedRandomSampler

    # 4. Map the weights ONLY to the images in the training subset
    train_weights = []

    # --- pedal emphasis, derived from the data -------------------------------
    # These used to be three hand-tuned multipliers, which only mean anything against one
    # particular class balance -- carry them to a different dataset and they silently
    # emphasise the wrong things. What actually generalises is the PRIORITY: how much of
    # the pedal head's attention each rare-but-critical situation deserves. That is a
    # statement about driving, not about a recording session.
    #
    # So the knobs are target shares of the pedal loss, and the multipliers are solved
    # for from the measured class counts:
    #
    #     weight_c = (share_c / n_c) normalised so the "other" class sits at 1.0
    #
    # A dataset with far more braking gets a lower brake multiplier automatically,
    # because the same share is spread over more frames.
    PEDAL_LOSS_SHARES = {
        "stop": 0.111,     # at rest, holding
        "resume": 0.013,   # at rest, pulling away -- tiny class, needs the leverage
        "brake": 0.183,    # braking while moving; the only one that touches an approach
    }

    _S = np.abs(np.array([full_dataset.speeds[i] for i in train_indices]))
    _P = np.array([full_dataset.pedals[i] for i in train_indices])
    _resume = (_S < 0.5) & (_P > 0.1)
    _classes = {
        "resume": _resume,
        "stop": (_S < 0.5) & ~_resume,
        "brake": (_S >= 0.5) & (_P < -0.1),
    }
    _classes["other"] = ~(_classes["resume"] | _classes["stop"] | _classes["brake"])

    _otherShare = 1.0 - sum(PEDAL_LOSS_SHARES.values())
    _otherN = max(1, int(_classes["other"].sum()))
    _perFrameOther = _otherShare / _otherN

    _derived = {}
    for _name, _share in PEDAL_LOSS_SHARES.items():
        _n = int(_classes[_name].sum())
        _derived[_name] = (_share / _n) / _perFrameOther if _n > 0 else 1.0

    STOP_WEIGHT = _derived["stop"]
    RESUME_WEIGHT = _derived["resume"]
    BRAKE_WEIGHT = _derived["brake"]

    print("Pedal emphasis derived from class counts:")
    for _name in ("stop", "resume", "brake", "other"):
        _n = int(_classes[_name].sum())
        _w = _derived.get(_name, 1.0)
        _share = PEDAL_LOSS_SHARES.get(_name, _otherShare)
        print(f"   {_name:<7} {_n:>7} frames ({100.0 * _n / len(_S):5.2f}%)"
              f"  -> weight x{_w:5.2f}   share {100.0 * _share:5.1f}%")

    # The tuned values these shares reproduce, kept as the record of how they were found:
    #   stop 2.10 - holding at a red sign
    #   resume 3.00 - rare class (0.52% of frames); at 4.0 in the SAMPLER it broke
    #     stopping, but that was checkpoint spread, not the weight itself
    #   brake 2.20 - lowered from 3.00 after rest distance went 9.5 -> 13.2 units and
    #     6 of 33 stops landed outside the zone; 2.2 brought it back to 9.66 with 0 short

    # --- stop-sign stall weighting -------------------------------------------
    # Measured failure: the car decelerates well from 120 units out, then stalls at
    # 14-16 mph and coasts through the sign. The demonstrations are unambiguous there
    # (big sign visible, 5-20 mph -> mean pedal -0.19 to -0.31) but that cell is only
    # 3.87% of sampled frames, competing against launch frames at the same speeds whose
    # labels average +0.72. Weighted by share and magnitude that predicts ~+0.12, which
    # is what the model actually outputs.
    #
    # Deliberately narrow: gated on speed so the early approach -- which already works
    # -- is untouched, and gated on the sign rather than on the pedal so launch frames
    # (no red sign in view) keep their current weight and resuming is unaffected.
    # Set to 1.0 to disable.
    # MEASURED FAILURE -- left at 1.0 (off). At 3.0 this drove the low-speed pedal
    # unconditionally negative: creep -0.09 (true +0.17), slow -0.06 (true +0.06). The
    # car then brakes whenever it is slow, sign or no sign, so it cannot resume.
    #
    # The lesson: reweighting a cell that is DEFINED by an input feature does not teach
    # the model to use that feature. It only reshapes the label distribution the model
    # sees at those speeds -- and since keying on `speed` is far easier than keying on a
    # small red patch that has to survive three stride-2 convs, more weight just made it
    # key on speed harder. Conditioning has to be forced in the representation, not
    # bribed through the sampler.
    SIGN_STALL_WEIGHT = 1.0
    SIGN_SALIENCE_MIN = 100        # red pixels; roughly "sign within ~30 units"
    SIGN_STALL_MIN_SPEED = 0.5     # already-stopped frames are excluded: the model
                                   # holds the brake correctly there (-0.35 vs -0.37),
                                   # and including them put 36% of every batch into
                                   # this one cell, which would swamp steering
    SIGN_STALL_MAX_SPEED = 20.0

    stallFrames = 0
    stallMask = []
    for idx in train_dataset.indices:
        speed = full_dataset.speeds[idx]
        pedal = full_dataset.pedals[idx]

        if PEDAL_WEIGHTS_IN_LOSS:
            # emphasis is applied in the loss instead, so the sampler leaves the frame
            # distribution alone -- applying it in both places would double-count it
            weight = 1.0
        elif abs(speed) < 0.5 and pedal > 0.1:
            weight = RESUME_WEIGHT
        elif abs(speed) < 0.5:
            weight = STOP_WEIGHT
        elif pedal < -0.1:
            weight = BRAKE_WEIGHT
        else:
            weight = 1.0

        inStallCell = (full_dataset.signSalience[idx] >= SIGN_SALIENCE_MIN
                       and SIGN_STALL_MIN_SPEED <= abs(speed) < SIGN_STALL_MAX_SPEED)
        if inStallCell:
            weight *= SIGN_STALL_WEIGHT
            stallFrames += 1

        train_weights.append(weight)
        stallMask.append(inStallCell)

    _w = np.array(train_weights)
    _m = np.array(stallMask)
    _share = 100.0 * _w[_m].sum() / _w.sum() if stallFrames else 0.0
    print(f"Stop-sign stall cell: {stallFrames} frames x{SIGN_STALL_WEIGHT} "
          f"-> {_share:.1f}% of sampled batches")
    # -------------------------------------------------------------------------

    # --- per-session recency weighting ---------------------------------------
    # Demonstrations are not equally good. A dataset accumulates over time while the
    # driver gets more consistent and starts deliberately recording the gaps they have
    # noticed -- so later sessions are usually the better teachers, and older ones dilute
    # them. Here 66 sharp stop approaches were competing with 252 older mushy ones and
    # ended up as 21% of the stop demonstrations.
    #
    # Deliberately derived from the data rather than a hand-listed set of "good"
    # sessions, so it carries over to any dataset: sessions are ordered chronologically
    # (folder names are timestamps), each gets its midpoint position by CUMULATIVE FRAME
    # COUNT -- not session index, so a 20-frame session cannot count for as much as a
    # 24,000-frame one -- and weight rises exponentially with that position.
    #
    # Weights are renormalised to average 1.0, so this redistributes emphasis without
    # changing the overall loss scale. Set the ratio to 1.0 to disable.
    SESSION_RECENCY_RATIO = 3.0   # newest session counts this much more than the oldest

    sessionOf = [Path(p).parent.name for p in full_dataset.image_paths]
    if SESSION_RECENCY_RATIO != 1.0:
        counts = defaultdict(int)
        for s in sessionOf:
            counts[s] += 1
        ordered = sorted(counts)                       # timestamped names sort chronologically
        totalFrames = sum(counts.values())
        sessionWeight, cum = {}, 0
        for s in ordered:
            midpoint = (cum + counts[s] / 2.0) / totalFrames
            sessionWeight[s] = SESSION_RECENCY_RATIO ** midpoint
            cum += counts[s]

        perFrame = np.array([sessionWeight[sessionOf[i]] for i in train_dataset.indices])
        perFrame /= perFrame.mean()                    # keep the average weight at 1.0
        train_weights = list(np.array(train_weights) * perFrame)

        print(f"Session recency weighting (x{SESSION_RECENCY_RATIO} newest:oldest) "
              f"over {len(ordered)} sessions:")
        for s in ordered[-3:]:
            print(f"   newest  {s}  {counts[s]:>6} frames  raw weight {sessionWeight[s]:.2f}")
        for s in ordered[:2]:
            print(f"   oldest  {s}  {counts[s]:>6} frames  raw weight {sessionWeight[s]:.2f}")
    # -------------------------------------------------------------------------

    sampler = WeightedRandomSampler(train_weights, num_samples=len(train_weights), replacement=True)

    # JPEG decode + augmentation is CPU-bound and was running on the main thread,
    # serialised against the GPU. Workers overlap it with the forward/backward pass.
    loader_kwargs = dict(pin_memory=(device.type == "cuda"))
    if num_workers > 0:
        loader_kwargs.update(
            num_workers=num_workers,
            persistent_workers=True,   # pay the ursina/panda3d import cost once, not per epoch
            prefetch_factor=4,
            worker_init_fn=_seed_worker,
        )
    print(f"DataLoader workers: {num_workers}"
          + ("  (single-process loading)" if num_workers == 0 else ""))

    trainLoader = DataLoader(train_dataset, batch_size=64, sampler=sampler, **loader_kwargs)

    valLoader = DataLoader(val_dataset, batch_size=64, shuffle=False, **loader_kwargs)


    prgmDir = Path(__file__).parent
    modelLocation = prgmDir / "Models"

    if not modelLocation.exists():
        print("\"Models\" Directory doesn't exist, creating it")
        modelLocation.mkdir(parents=True)
    else:
        print("\"Models\" Directory exists")




    best_val_loss = float('inf')
    best_val_steer = float('inf')
    best_epoch = 0
    patience_counter = 0
    MAX_PATIENCE = EARLY_STOP_PATIENCE  # Stop if no improvement for this many epochs

    # One run ID for the whole run, so every checkpoint from this run shares a name
    # (previously each save called Indentifier() again and minted a new timestamp).
    runID = DataCollection.Indentifier()

    # --- weight averaging ----------------------------------------------------
    # Two checkpoints from a single run drove completely differently: one steered at
    # 96.4% on-road and never stopped, the other stopped 2/2 inside the zone and left
    # the road 17 times. Individual checkpoints land in whichever corner of the loss
    # basin the last few batches pushed them into, and our val metrics cannot tell which
    # -- val_steer has been anti-correlated with driving three times now.
    #
    # An exponential moving average of the weights sits nearer the centre of that basin
    # instead of on its rim, which is both better and far more repeatable. Safe here
    # because the net has no BatchNorm, so there are no running statistics to rebuild.
    EMA_DECAY = 0.999
    ema_state = {k: v.detach().clone().float() for k, v in model.state_dict().items()}


    def emaUpdate():
        with torch.no_grad():
            for k, v in model.state_dict().items():
                ema_state[k].mul_(EMA_DECAY).add_(v.detach().float(), alpha=1.0 - EMA_DECAY)

    # Optional progress reporting, so a UI can run this on a background thread and still
    # show what is happening instead of freezing until the whole run finishes.
    def report(**fields):
        if progress is not None:
            try:
                progress(fields)
            except Exception:
                pass

    report(stage="start", epoch=0, epochs=epochs, batch=0, batches=0)

    for epoch in range(epochs):
        print(f"Epoch [{epoch + 1}/{epochs}] - START")

        # --- TRAINING PHASE ---
        def onBatch(i, total, _e=epoch):
            report(stage="training", epoch=_e + 1, epochs=epochs, batch=i, batches=total)

        train_loss, train_steer, train_pedal = train(model, trainLoader, criterion, optimizer,
                                                     device, onStep=emaUpdate, onBatch=onBatch)
        report(stage="validating", epoch=epoch + 1, epochs=epochs, batch=0, batches=0)


        #NOTE TO FUTURE SELF, CLEAN THIS STUFF UP, a bunch of junk is leftover from different tests and at this point,
        # between the tests I ran myself, and the one's claude and grok made, it's disgusting bad
        # --- VALIDATION PHASE ---
        # Scored per-target. The blended number alone is ~91% pedal (pedal label
        # variance is ~10x steering's), so it cannot detect a steering regression.
        model.eval()
        pred_chunks, targ_chunks, speed_chunks, aux_chunks = [], [], [], []
        with torch.no_grad():
            for images, speeds, speedLimits, targets in valLoader:
                images = images.to(device, non_blocking=True)
                speeds = speeds.to(device, non_blocking=True)
                speedLimits = speedLimits.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                outputs, aux = model(images, speeds, speedLimits, return_aux=True)

                pred_chunks.append(outputs.cpu())
                targ_chunks.append(targets.cpu())
                speed_chunks.append(speeds.cpu())
                aux_chunks.append(aux.cpu())

        preds = torch.cat(pred_chunks).numpy()
        targs = torch.cat(targ_chunks).numpy()
        val_speeds = torch.cat(speed_chunks).numpy().ravel()
        aux_pred = torch.cat(aux_chunks).numpy()
        aux_true = targs[:, 2]

        steer_err = preds[:, 0] - targs[:, 0]
        val_steer = float((steer_err ** 2).mean())
        val_pedal = float(((preds[:, 1] - targs[:, 1]) ** 2).mean())
        val_steer_mae = float(np.abs(steer_err).mean())

        # Same weighting as the training objective, so the scheduler, early stopping
        # and checkpoint selection all optimise the thing we actually asked for.
        # (The old unweighted (val_steer + val_pedal)/2 was ~91% pedal and could not
        # see a steering regression at all.)
        val_weighted = steer_weight * val_steer + val_pedal

        # --- BUCKETED DIAGNOSTICS ------------------------------------------------
        # Plain MAE is useless here: 68.6% of center steering labels are EXACTLY 0.000
        # (the keyboard snaps the wheel to zero when A/D is released), so "predict the
        # mean" is really "predict the mode" and scores a MAE no continuous-output
        # model can match. Split straight-line holding from actual turning instead.
        straight = np.abs(targs[:, 0]) < 0.02
        turning = ~straight

        # NOT "jitter". You steer with a keyboard, so 68.6% of labels are exactly 0.000
        # while the model outputs continuous corrections -- good closed-loop control
        # registers here as error. Read it together with the autocorrelation below:
        #   high autocorr (~0.45) = slow, sustained lane-centering -> this is the model
        #                           driving, and the models that do it drive WELL
        #   low  autocorr (~0.13) = frame-to-frame noise -> genuinely bad
        straight_dev_deg = float(np.sqrt((steer_err[straight] ** 2).mean())) * 540 if straight.any() else float('nan')
        # valLoader is shuffle=False over chunk-ordered indices, so consecutive entries
        # are consecutive frames; the ~1% that straddle chunk boundaries are ignorable.
        steer_autocorr = float(np.corrcoef(steer_err[:-1], steer_err[1:])[0, 1]) if len(steer_err) > 2 else float('nan')
        turn_rmse_deg = float(np.sqrt((steer_err[turning] ** 2).mean())) * 540 if turning.any() else float('nan')
        turn_base_deg = float(np.sqrt((targs[turning, 0] ** 2).mean())) * 540 if turning.any() else float('nan')
        # <1 means the head is hedging toward the mean instead of committing to turns
        commit = float(preds[:, 0].std() / targs[:, 0].std())

        report(stage="epoch done", epoch=epoch + 1, epochs=epochs, batch=0, batches=0,
               trainLoss=train_loss, valLoss=val_weighted, valSteer=val_steer,
               valPedal=val_pedal, commit=commit)

        print(f"Epoch [{epoch + 1}/{epochs}] - Train {train_loss:.4f} | Val {val_weighted:.4f}")
        # Both expressed as fraction of their OWN label variance -- train is all-camera
        # (var ~0.0143), val is Center-only (var ~0.0046), so raw MSE is not comparable.
        train_unexp = train_steer / train_steer_var
        val_unexp = val_steer / baseline_steer
        print(f"   steer  unexplained variance: train {train_unexp * 100:5.1f}%  val {val_unexp * 100:5.1f}%  "
              f"(overfit gap {val_unexp - train_unexp:+.2f})")
        print(f"          straight-frame dev {straight_dev_deg:5.1f} deg (autocorr {steer_autocorr:+.2f})  |  "
              f"turning RMSE {turn_rmse_deg:5.1f} vs {turn_base_deg:5.1f} deg if it never turned")
        # commit is the ONLY offline number that has tracked real driving quality so far:
        # 0.86 and 0.97 drove well, 0.65 drove badly, while val_steer said the opposite.
        print(f"          commit {commit:.2f}  <- 0.85-1.0 has meant good driving; low means hedging")
        print(f"   pedal  val {val_pedal:.5f} ({val_pedal / baseline_pedal * 100:5.1f}% of baseline) "
              f"| train {train_pedal:.5f}")

        # --- CAN THE TRUNK SEE THE SIGN? -----------------------------------------
        # The point of the aux head is as much diagnostic as fix. If it predicts distant
        # signs well, the sign IS resolvable at this resolution and any remaining stop
        # failure is policy. If it only works up close, no amount of policy tuning will
        # help and the answer is resolution (a zoomed second stream / less early stride).
        if aux_weight:
            unexp = ((aux_pred - aux_true) ** 2).mean() / max(aux_true.var(), 1e-9)
            corr = float(np.corrcoef(aux_pred, aux_true)[0, 1]) if aux_true.std() > 1e-9 else float('nan')
            print(f"   sign   aux unexplained {unexp * 100:5.1f}%  corr {corr:+.2f}")
            seg = []
            for lo, hi, name in ((-1e-9, 1e-9, "none"), (1e-9, 0.35, "far"),
                                 (0.35, 0.62, "mid"), (0.62, 2.0, "near")):
                m = (aux_true > lo) & (aux_true <= hi) if lo >= 0 else (aux_true <= hi)
                if m.sum() < 5:
                    continue
                seg.append(f"{name} n={int(m.sum())} pred={aux_pred[m].mean():.2f} true={aux_true[m].mean():.2f}")
            print("          " + "  |  ".join(seg))
        # -------------------------------------------------------------------------

        band_report = []
        for lo, hi, name in ((0.0, 0.5, "stopped"), (0.5, 5.0, "creep"), (5.0, 20.0, "slow"), (20.0, 1e9, "cruise")):
            m = (np.abs(val_speeds) >= lo) & (np.abs(val_speeds) < hi)
            if m.sum() == 0:
                continue
            band_report.append(f"{name} n={int(m.sum())} pred={preds[m, 1].mean():+.2f} true={targs[m, 1].mean():+.2f}")
        print(f"          pedal by speed: " + "  |  ".join(band_report))
        # -------------------------------------------------------------------------

        # Step the scheduler to potentially slow down the learning rate
        scheduler.step(val_weighted)

        # --- STEERING-ONLY CHECKPOINT ---
        # Kept as a separate checkpoint so a run that trades steering for pedal is
        # still visible, and recoverable, after the fact.
        if val_steer < best_val_steer:
            best_val_steer = val_steer
            torch.save(model.state_dict(), modelLocation / f"Model_{runID}_BESTSTEER.pth")
            print(f"   -> New best STEERING model saved!")

        # --- EARLY STOPPING & SAVING ---
        if val_weighted < best_val_loss:
            best_val_loss = val_weighted
            best_epoch = epoch + 1
            patience_counter = 0

            # Save the new "best" model
            savePath = modelLocation / f"Model_{runID}_BEST.pth"
            torch.save(model.state_dict(), savePath)
            print(f"   -> New best model saved!")
        else:
            patience_counter += 1
            if patience_counter >= MAX_PATIENCE:
                print(f"Early stopping triggered! Validation loss hasn't improved in {MAX_PATIENCE} epochs.")
                break

    # Saved every run regardless of val metrics -- it is not selected by them, which is
    # the point. Evaluate it alongside _BEST rather than trusting either blind.
    emaPath = modelLocation / f"Model_{runID}_EMA.pth"
    torch.save({k: v.to(dtype=p.dtype) for (k, v), p in
                zip(ema_state.items(), model.state_dict().values())}, emaPath)

    print(f"\nRun {runID} finished.  (cameras={CAMERAS}, steer weight x{steer_weight:.2f})")
    print(f"  averaged weights  : EMA decay {EMA_DECAY}   -> Model_{runID}_EMA.pth")
    print(f"  best weighted val : {best_val_loss:.5f}   -> Model_{runID}_BEST.pth")
    print(f"  best val_steer    : {best_val_steer:.5f}   -> Model_{runID}_BESTSTEER.pth")
    print(f"  best epoch        : {best_epoch} of {epoch + 1} run")

    report(stage="finished", epoch=epoch + 1, epochs=epochs, batch=0, batches=0,
           runID=runID, bestEpoch=best_epoch)

#going to do rbg, then height of 260 pixels and a width of 375 (cutting 79 off left side, 58 from right side, 178 from the top, and 74 from the bottomn)

# temp =Net()
#
# dummy_image = torch.randn(1, 3, 260, 375)
#
# output = temp(dummy_image)