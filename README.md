# Autopilot Sim

This sim lets you record yourself driving, and trains a cnn to mimic the behavior.

The road is fully procedurally generated. You drive it to collect demonstrations, train a
network on those frames, and then can run an eval on the trained model by letting it drive a fixed course (not one you train on to avoid memorization in the model).
From there, you can go back to the sim and correct the model as it drives to enhance the dataset to perfect the model.
Note that, in training, be sure to collect an ample starting dataset as understanding stop signs is an emergent behavior
from the model.

Ursina (Panda3D) for the sim, PyTorch for the model.

## Results

My final model
`Models/Model 5.3.pth` on the standard evaluation course (`seed 20260808`, dense signs,
600 seconds):

| | |
|---|---|
| distance driven | 31,367 units |
| time on road | 100% |
| time in correct lane | 100% |
| lane departures | 0 |
| mean lane deviation | 0.43 units |
| stop signs stopped at | 35 / 36 |
| resumed on its own | 34 / 34 |
| human intervention | 0.00% |

The missed sign was on the sharpest curve of the course (3.76 against a 1.88 median),
where it comes into view late. The sign was visible and an earlier checkpoint stopped
there, so this is a gap in the training data rather than a perception limit.

## Setup

Python 3.13. Install PyTorch from the CUDA index first — a plain `pip install torch`
gets the CPU-only build, which like makes training impractically slow, especially for larger datasets:

```bash
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
```

Then:

```bash
pip install -r requirements.txt
```

Driving the sim and running a trained model work likely without a GPU. Training likely does not, at least not practically.

## Running it

```bash
python Autopilot_Menu.py
```

The menu has five options:

- **Start Simulation** — drive the world yourself, with recording and autopilot
  available. Asks first how densely stop signs should spawn.
- **Train Model** — trains on everything in `Data/`. Runs on a background thread, so the
  window stays responsive and shows progress. Saves three checkpoints and lets you name
  them.
- **Evaluate Model** — drives one or more models on a fixed seeded course and scores
  them, then shows a comparison table.
- **View Current Models** — every checkpoint in `Models/` with its size and evaluation
  history.
- **Data check** — frame counts, and how much of the rare material is present: stop
  signs, braking, resuming, sharp turns. Classes under 2% are flagged.

## Controls

Driving is `W` / `A` / `S` / `D`.

| key |                                                                    |
|---|--------------------------------------------------------------------|
| `0` | autopilot on/off                                                   |
| `-` | choose which model drives                                          |
| `9` | start/stop recording (the status light turns green when recording) |
| `8` | corrections-only capture                                           |
| `1` / `2` | chase (suggested) / windshield camera                              |
| `esc` | settings panel — weather, speed limit, camera, exit to menu        |

Corrections-only capture (`8`) leaves the model driving and writes only the frames where
you intervene, so a session records the cases it gets wrong rather than more of what it
already handles.

## The workflow

1. Drive with `9` on. Use dense signs if you want stop-sign approaches specifically.
   Each session gets its own folder under `Data/`.
2. Run Data check. If braking or resuming is at a couple of percent, more general
   driving won't move it.
3. Train. Up to 100 epochs, stopping early after 7 with no improvement. Produces
   `_BEST`, `_BESTSTEER` and `_EMA` checkpoints.
4. Evaluate. Use 240s or more with dense signs, or there are too few stop signs to draw
   anything from.
5. Repeat.

## Evaluation

Models are ranked on the seeded course rather than on validation loss.

Validation steering error ranked checkpoints incorrectly three separate times — models
with better offline numbers drove worse. Two causes: most steering labels are exactly
zero, because keyboard steering snaps back to centre, and the network could distinguish
the side cameras from the centre one and use that instead of the road.

The harness measures the closed-loop outcomes instead — time on road, lane deviation,
whether it stopped at each sign, whether it stopped where the sign turns green, whether
it resumed unaided. Every model drives an identical course, so runs are comparable.

## Repo layout

| file | |
|---|---|
| `Autopilot_Menu.py` | entry point and menu wiring |
| `Autopilot_sim.py` | the simulator — world generation, car physics, autopilot loop |
| `DataCollection.py` | dataset, recorder, augmentation |
| `Training.py` | model and training loop |
| `Evaluation.py` | closed-loop scoring harness |
| `MenuScreens.py` | menu screens and popups |
| `train.py` | train from the command line instead of the menu |

`Training.py` opens with a `HISTORY NOTES` block covering what was tried and what came
of it, including the changes that didn't work.

## Notes

- Recorded data isn't included. `Data/` and most of `Models/` are gitignored; only the
  one model is here.
- Training reads three cameras (centre, left, right), with a steering offset applied to
  the side views so they teach recovery. Inference uses the centre camera only.
- Seed 20260808 is the evaluation course. Recording training data on it would invalidate
  the numbers above.
