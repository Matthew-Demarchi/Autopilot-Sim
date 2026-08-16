"""Menu screens: popups, hover help, training progress, model browser, data check.

Kept out of Autopilot_Menu.py so that file stays small and, critically, free of
import-time side effects -- Windows spawns DataLoader workers by re-importing whatever
module is __main__, so anything that builds UI at import time would open blank windows.
Nothing here runs until it is constructed.
"""

import csv
import json
import math
import threading
from collections import defaultdict
from pathlib import Path

from ursina import *

import Autopilot_sim  # destroy_all -- ursina's destroy() does not cascade to children


PROJECT_DIR = Path(__file__).parent
MODELS_DIR = PROJECT_DIR / "Models"
EVALS_DIR = PROJECT_DIR / "Evaluations"
DATA_DIR = PROJECT_DIR / "Data"

# Frames per second to run the window at while training. Training shares the process
# with the renderer, so every frame drawn is GIL time taken from the training thread.
TRAINING_FPS = 10


def close(entity):
    """destroy() leaves children on screen; always tear panels down with this."""
    if entity:
        Autopilot_sim.destroy_all(entity)


def _panel(parent, w=1.15, h=0.92, alpha=238):
    return Entity(parent=parent, model="quad",
                  color=color.rgba32(20, 20, 30, alpha), scale=(w, h), z=1)


def _inputField(parent, **kwargs):
    """An InputField that is readable and does not blow up on the scroll wheel.

    Two things ursina gets wrong for our use:

    * InputField takes a text_color argument and then never applies it -- it is a dead
      parameter. The text entity keeps Text's default (white), so a white field renders
      white-on-white. The colour has to be set on the text entity afterwards, where it
      sticks because Text stores it as its 'default' tag colour and so survives the
      re-render that happens on every keystroke.
    * TextField binds the scroll wheel to set_scroll -> render(), which rebuilds the
      Panda text nodes. Scrolling over a field inside a screen that also pages on scroll
      hits that render on a node path that is already being torn down, and Panda aborts
      with 'AssertionError: _error_type == ET_ok'. These are one-line fields; there is
      nothing to scroll, so unbind it.
    """
    ink = color.black
    kwargs.setdefault("color", color.white)
    f = InputField(parent=parent, **kwargs)
    f.text_field.text_entity.color = ink
    f.text_field.shortcuts["scroll_up"] = ()
    f.text_field.shortcuts["scroll_down"] = ()

    # Flatten every hover and press state, because InputField inherits Button's and each
    # one fights the two colours set above:
    #   highlight_color   InputField asks for black, so hovering colour-scales the quad
    #                     to black -- and you have to hover a field to type in it.
    #   highlight_text_color  Button's on_mouse_enter runs "if self.text:", and
    #                     InputField redefines text to mean the field's contents. So any
    #                     non-empty field (typed, or pre-filled like the browser's) gets
    #                     its text repainted in Button's off-white default on hover. It
    #                     sticks: on_mouse_exit restores from self.text_color, whose
    #                     getter reads the entity's current colour -- the off-white the
    #                     enter just wrote. Pinning it to the ink colour makes both the
    #                     enter and the exit no-ops.
    f.highlight_color = f.color
    f.pressed_color = f.color
    f.highlight_text_color = ink
    return f


def readEvalFile(path):
    out = {}
    try:
        for line in path.read_text().split("\n"):
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            k, v = k.strip(), v.strip()
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v
    except Exception:
        return None
    return out


# ============================================================== hover descriptions


class HoverHelp(Entity):
    """Description panel on the right of the main menu.

    Buttons say what they are, not what they do or what they will cost you -- which
    matters when one of them starts a 45 minute training run.
    """

    def __init__(self, **kwargs):
        # setdefault, not a literal: the menu parents this to menu_parent so it hides
        # along with the menu, and passing parent= twice is a TypeError.
        kwargs.setdefault("parent", camera.ui)
        super().__init__(**kwargs)
        self.bg = Entity(parent=self, model="quad", color=color.rgba32(18, 18, 28, 200),
                         scale=(0.5, 0.42), position=(0.52, 0.0), z=1)
        self.title = Text(parent=self, text="", position=(0.30, 0.17), scale=1.1,
                          color=color.cyan)
        self.body = Text(parent=self, text="", position=(0.30, 0.13), scale=0.8,
                         color=color.light_gray)
        self.setHelp(None, None)

    def setHelp(self, title, body):
        self.title.text = title or ""
        self.body.text = body or "Hover a button for details."
        self.bg.color = color.rgba32(18, 18, 28, 200 if title else 120)


def attachHover(button, help_panel, title, body):
    """Wire a button's hover state to the description panel."""
    def onEnter():
        help_panel.setHelp(title, body)

    def onExit():
        help_panel.setHelp(None, None)

    button.on_mouse_enter = onEnter
    button.on_mouse_exit = onExit
    return button


# ============================================================== start-simulation popup


class SimStartPopup(Entity):
    """Asks how the world should be built instead of needing a second menu button."""

    def __init__(self, onStart, onCancel=None, **kwargs):
        kwargs.setdefault("parent", camera.ui)
        kwargs.setdefault("z", -10)
        super().__init__(**kwargs)
        self.onStart = onStart
        self.onCancel = onCancel
        self.signDensity = 1
        self._rebuild = False
        self._build()

    def _build(self):
        _panel(self, 0.86, 0.5)
        Text(parent=self, text="Start Simulation", position=(-0.38, 0.17),
             scale=1.5, color=color.cyan)

        Text(parent=self, text="Stop sign density", position=(-0.38, 0.09),
             scale=0.95, color=color.light_gray)
        for i, (dens, lab, note) in enumerate((
                (1, "normal", "as the world normally generates"),
                (3, "dense x3", "~3x the signs, 250+ units apart"))):
            active = self.signDensity == dens
            b = Button(text=lab, parent=self, scale=(0.2, 0.075),
                       position=(-0.25 + i * 0.23, 0.03),
                       color=color.azure if active else color.dark_gray,
                       text_color=color.white)

            def pick(d=dens):
                self.signDensity = d
                self._rebuild = True
            b.on_click = pick

        note = ("More signs per minute, which is what you want when recording stop "
                "approaches or corrections." if self.signDensity == 3 else
                "Normal traffic density. Use this for general driving data.")
        Text(parent=self, text=note, position=(-0.38, -0.04), scale=0.75, color=color.gray)

        b = Button(text="Start", parent=self, scale=(0.26, 0.08), position=(-0.19, -0.15),
                   color=color.lime, text_color=color.black)

        def go():
            d = self.signDensity
            close(self)
            self.onStart(d)
        b.on_click = go

        if self.onCancel:
            b = Button(text="Cancel", parent=self, scale=(0.2, 0.08), position=(0.16, -0.15),
                       color=color.dark_gray, text_color=color.white)

            def cancel():
                cb = self.onCancel
                close(self)
                cb()
            b.on_click = cancel

    def update(self):
        if self._rebuild:
            self._rebuild = False
            for c in list(self.children):
                Autopilot_sim.destroy_all(c)
            self._build()


# ============================================================== training progress


class TrainingProgress(Entity):
    """Runs trainStart on a worker thread and shows what it is doing.

    Training used to block the Ursina main loop outright, so the window froze -- often
    for 30+ minutes -- with no indication of whether it was working. Torch releases the
    GIL during CUDA work and DataLoader waits, so the UI keeps animating.

    The worker writes into a plain dict and update() reads it. That is safe here because
    every value is a single assignment of an immutable scalar; no locking needed.
    """

    def __init__(self, epochs, workers, onFinish=None, **kwargs):
        kwargs.setdefault("parent", camera.ui)
        kwargs.setdefault("z", -10)
        super().__init__(**kwargs)
        self.onFinish = onFinish
        self.state = {"stage": "starting", "epoch": 0, "epochs": epochs,
                      "batch": 0, "batches": 0}
        self.done = False
        self.error = None
        self._renameBuilt = False
        self.nameField = None

        # Kept as a reference so the panel can grow when the rename controls appear on
        # finish, rather than sizing for them up front and leaving dead space throughout
        # the run.
        self.panelBg = _panel(self, 1.0, 0.5)
        Text(parent=self, text="Training", position=(-0.45, 0.17), scale=1.5, color=color.cyan)
        self.status = Text(parent=self, text="starting...", position=(-0.45, 0.10),
                           scale=0.95, color=color.white)
        self.detail = Text(parent=self, text="", position=(-0.45, 0.04),
                           scale=0.8, color=color.light_gray)

        Entity(parent=self, model="quad", color=color.dark_gray,
               scale=(0.9, 0.035), position=(0.0, -0.04), z=0)
        self.bar = Entity(parent=self, model="quad", color=color.cyan,
                          scale=(0.001, 0.035), position=(-0.45, -0.04), z=-1,
                          origin=(-0.5, 0))

        import Training as _T
        patience = getattr(_T, "EARLY_STOP_PATIENCE", 7)
        Text(parent=self,
             text=f"runs at most {epochs} epochs, and stops early once {patience} in a row"
                  f" go by with no improvement",
             position=(-0.45, -0.10), scale=0.75, color=color.light_gray)
        Text(parent=self, text="the window stays responsive; console has the full log",
             position=(-0.45, -0.145), scale=0.7, color=color.gray)

        # Sits in the top-right corner of the *grown* panel, so it is only in a sane spot
        # once _grow() has run. That is fine -- it stays disabled (and so unrendered)
        # until training finishes, which is the same moment the panel grows.
        self.closeButton = Button(text="Close", parent=self, scale=(0.2, 0.08),
                                  position=(0.38, 0.29), color=color.dark_gray,
                                  text_color=color.white, enabled=False)
        self.closeButton.on_click = self._finish

        # Throttle the renderer while training. The progress callback itself costs about
        # 1ms per epoch (0.33us x ~2900 batches) so it is not the bottleneck -- the cost
        # is that a 60fps render loop and the training thread fight over the GIL, and the
        # train loop's per-batch .item() calls force GPU syncs while holding it. A
        # progress bar does not need 60fps; 10 is plenty and gives the GIL back.
        self._restoreClock = None
        try:
            from panda3d.core import ClockObject
            clock = ClockObject.getGlobalClock()
            self._restoreClock = (clock, clock.getMode(), clock.getFrameRate())
            clock.setMode(ClockObject.M_limited)
            clock.setFrameRate(TRAINING_FPS)
        except Exception as e:
            print(f"(could not throttle the renderer while training: {e})")

        import Training

        def run():
            try:
                Training.trainStart(epochs, num_workers=workers, progress=self.state.update)
            except Exception as e:                      # surfaced in the panel, not swallowed
                self.error = str(e)
                print(f"TRAINING ERROR: {e}")
            finally:
                self.done = True

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()

    def restoreClock(self):
        if self._restoreClock:
            clock, mode, rate = self._restoreClock
            try:
                clock.setMode(mode)
                clock.setFrameRate(rate)
            except Exception:
                pass
            self._restoreClock = None

    def _finish(self):
        self.restoreClock()
        cb = self.onFinish
        close(self)
        if cb:
            cb()

    def _grow(self):
        """Give the panel the extra height the finished state needs.

        The quad is centre-origin, so this opens up room at the bottom for the rename
        row and at the top for the Close button in one move.
        """
        self.panelBg.scale_y = 0.70

    def _buildRename(self):
        """Offer to name the run's checkpoints rather than leaving them as timestamps.

        Only appears once training succeeds. Renames all three files (BEST, BESTSTEER,
        EMA) together, keeping their suffixes -- they are one run and only mean anything
        compared against each other.
        """
        runID = self.state.get("runID")
        if not runID:
            return
        made = sorted(MODELS_DIR.glob(f"Model_{runID}_*.pth"))
        if not made:
            return

        Text(parent=self, text=f"saved {len(made)} checkpoint(s) as Model_{runID}_*.pth",
             position=(-0.45, -0.19), scale=0.75, color=color.light_gray)
        Text(parent=self, text="rename to:", position=(-0.45, -0.265), scale=0.8,
             color=color.light_gray)

        self.nameField = _inputField(self, position=(-0.05, -0.26),
                                     scale=(0.44, 0.055), character_limit=40,
                                     default_value="")
        self.renameStatus = Text(parent=self, text="", position=(-0.45, -0.315),
                                 scale=0.72, color=color.gray)

        b = Button(text="Rename", parent=self, scale=(0.2, 0.07), position=(0.27, -0.26),
                   color=color.azure, text_color=color.white)

        def doRename():
            newName = (self.nameField.text or "").strip()
            if not newName:
                self.renameStatus.text = "type a name first"
                self.renameStatus.color = color.orange
                return
            if any(c in newName for c in '\\/:*?"<>|'):
                self.renameStatus.text = "name contains a character the filesystem will not take"
                self.renameStatus.color = color.orange
                return
            renamed, clashes = [], []
            for p in made:
                # keep the _BEST / _BESTSTEER / _EMA suffix so the three stay identifiable
                suffix = p.stem.replace(f"Model_{runID}", "")
                target = p.with_name(f"{newName}{suffix}.pth")
                if target.exists():
                    clashes.append(target.name)
                    continue
                try:
                    p.rename(target)
                    renamed.append(target.name)
                except Exception as e:
                    clashes.append(f"{p.name} ({e})")
            if renamed:
                self.renameStatus.text = f"renamed {len(renamed)}: " + ", ".join(renamed)[:70]
                self.renameStatus.color = color.lime
                print("RENAMED:", ", ".join(renamed))
            if clashes:
                self.renameStatus.text += f"   skipped (already exists): {', '.join(clashes)[:40]}"
                self.renameStatus.color = color.orange
        b.on_click = doRename

    def update(self):
        s = self.state
        epoch, epochs = s.get("epoch", 0), max(1, s.get("epochs", 1))
        batch, batches = s.get("batch", 0), s.get("batches", 0)

        withinEpoch = (batch / batches) if batches else 0.0
        overall = min(1.0, ((epoch - 1) + withinEpoch) / epochs) if epoch else 0.0
        self.bar.scale_x = max(0.001, 0.9 * overall)

        if self.done:
            self.bar.color = color.red if self.error else color.lime
            # A run that early-stops leaves the bar short of full, which looks like it
            # was interrupted -- say so rather than leaving it ambiguous.
            if self.error:
                self.status.text = f"failed: {self.error}"
            elif epoch < epochs:
                self.status.text = f"finished early at epoch {epoch} of max {epochs} (no improvement)"
            else:
                self.status.text = f"finished all {epochs} epochs"
            self.detail.text = "" if self.error else f"best epoch {s.get('bestEpoch', '?')}   run {s.get('runID', '')}"
            self.closeButton.enabled = True
            self.closeButton.text = "Close"
            if not self._renameBuilt:
                self._renameBuilt = True
                self.restoreClock()          # training is over; give the UI its fps back
                self._grow()                 # room for the Close corner (and rename below)
                if not self.error:
                    self._buildRename()
            return

        self.status.text = f"{s.get('stage', '')}   epoch {epoch} of max {epochs}"
        parts = []
        if batches:
            parts.append(f"batch {batch}/{batches}  ({100 * withinEpoch:.0f}% of epoch)")
        if "trainLoss" in s:
            parts.append(f"train {s['trainLoss']:.4f}")
        if "valLoss" in s:
            parts.append(f"val {s['valLoss']:.4f}")
        if "commit" in s:
            parts.append(f"commit {s['commit']:.2f}")
        self.detail.text = "   |   ".join(parts)


# ============================================================== model browser


class ModelBrowser(Entity):
    """Saved models, their evaluation results, and a comparison across runs."""

    PAGE = 9

    def __init__(self, onClose=None, **kwargs):
        kwargs.setdefault("parent", camera.ui)
        kwargs.setdefault("z", -10)
        super().__init__(**kwargs)
        self.onClose = onClose
        self.page = 0
        self.selected = None
        self._rebuild = False
        self.renameField = None

        self.models = sorted(MODELS_DIR.glob("*.pth"),
                             key=lambda f: f.stat().st_mtime, reverse=True) \
            if MODELS_DIR.exists() else []
        self.evals = []
        if EVALS_DIR.exists():
            for p in sorted(EVALS_DIR.glob("*.txt"), key=lambda f: f.stat().st_mtime, reverse=True):
                d = readEvalFile(p)
                if d:
                    self.evals.append((p.stem, d))
        self._build()

    def evalsForModel(self, name):
        """Match an eval to a model by the name recorded inside the eval file.

        Exact matches only. A substring test here matched "4.7A" against
        "4.7A - Upped Weights", so a model picked up evaluations belonging to a
        different model -- worse than showing nothing, because it looks like data.
        """
        stem = name.replace(".pth", "").strip().lower()
        hits = []
        for label, d in self.evals:
            recorded = str(d.get("model", "")).replace(".pth", "").strip().lower()
            # Either the name recorded at eval time, or the eval file's own name --
            # renaming a model after the fact leaves the recorded name stale, and
            # renaming the eval file to match the model is exactly how that gets fixed.
            # Both comparisons are exact; substring matching cross-linked models.
            if (recorded and recorded == stem) or label.strip().lower() == stem:
                hits.append((label, d))
        return hits

    def _build(self):
        _panel(self, 1.62, 0.94, 242)
        Text(parent=self, text="Models", position=(-0.79, 0.42), scale=1.5, color=color.cyan)
        Text(parent=self, text=f"{len(self.models)} saved   |   {len(self.evals)} evaluation files",
             position=(-0.79, 0.385), scale=0.8, color=color.gray)

        if not self.models:
            Text(parent=self, text="No .pth files in Models/", position=(-0.3, 0),
                 scale=1.2, color=color.red)
        else:
            pages = max(1, math.ceil(len(self.models) / self.PAGE))
            self.page = max(0, min(self.page, pages - 1))
            start = self.page * self.PAGE
            for i, path in enumerate(self.models[start:start + self.PAGE]):
                mb = path.stat().st_size / (1024 * 1024)
                n = len(self.evalsForModel(path.name))
                sel = self.selected == path
                label = f"{path.name[:34]:<34} {mb:5.1f}MB" + (f"  {n} eval" if n else "")
                b = Button(text=label, parent=self, scale=(0.72, 0.068),
                           position=(-0.42, 0.30 - i * 0.075),
                           color=color.azure if sel else color.dark_gray,
                           text_color=color.white)

                def pick(p=path):
                    self.selected = p
                    self._rebuild = True
                b.on_click = pick

            Text(parent=self, text=f"page {self.page + 1} of {pages}  -  scroll to page",
                 position=(-0.79, -0.40), scale=0.75, color=color.gray)

        self._buildDetail()

        b = Button(text="Close", parent=self, scale=(0.18, 0.075), position=(0.68, -0.42),
                   color=color.dark_gray, text_color=color.white)

        def shut():
            cb = self.onClose
            close(self)
            if cb:
                cb()
        b.on_click = shut

    def _buildDetail(self):
        # Panel spans x -0.81..0.81 and the model list occupies -0.78..-0.06, so the
        # detail column starts just right of it and the value columns have to land
        # inside 0.78 or they render off the panel.
        x = -0.02
        cols = (0.30, 0.50, 0.70)
        # Cleared every rebuild: the old field is destroyed, and input() must not test
        # .hovered on a dead entity.
        self.renameField = None

        if not self.selected:
            Text(parent=self, text="Select a model on the left", position=(x, 0.30),
                 scale=0.9, color=color.gray)
            return

        p = self.selected
        Text(parent=self, text=p.name[:38], position=(x, 0.32), scale=1.0, color=color.white)

        # Always show something, even with no evaluation attached -- an empty right-hand
        # side reads as a broken screen.
        import datetime
        stat = p.stat()
        when = datetime.datetime.fromtimestamp(stat.st_mtime).strftime("%b %d  %H:%M")
        width = "?"
        try:
            import torch
            sd = torch.load(p, map_location="cpu", weights_only=True)
            if "lin1.weight" in sd:
                width = str(sd["lin1.weight"].shape[0])
            params = sum(v.numel() for v in sd.values())
        except Exception:
            params = None
        info = f"{stat.st_size / (1024 * 1024):.1f} MB   saved {when}   lin1 width {width}"
        if params:
            info += f"   {params:,} params"
        Text(parent=self, text=info, position=(x, 0.28), scale=0.72, color=color.gray)

        # Rename in place. Timestamped checkpoint names are unreadable a week later, and
        # renaming outside the app is how the eval files ended up pointing at stale names.
        self.renameField = _inputField(self, position=(x + 0.19, 0.235),
                                       scale=(0.4, 0.05*.8), character_limit=40,
                                       default_value=p.stem)
        b = Button(text="Rename", parent=self, scale=(0.16*.75, 0.055*.75), position=(x + 0.48, 0.235),
                   color=color.azure, text_color=color.white)
        self.renameStatus = Text(parent=self, text="", position=(x, 0.315), scale=0.7,
                                 color=color.gray)

        def doRename():
            newName = (self.renameField.text or "").strip()
            if not newName or newName == p.stem:
                return
            if any(c in newName for c in '\\/:*?"<>|'):
                self.renameStatus.text = "invalid character for a filename"
                self.renameStatus.color = color.orange
                return
            target = p.with_name(newName + ".pth")
            if target.exists():
                self.renameStatus.text = f"{target.name} already exists"
                self.renameStatus.color = color.orange
                return
            try:
                p.rename(target)
                print(f"RENAMED: {p.name} -> {target.name}")
                # Reload the list and keep the same model selected under its new name
                self.models = sorted(MODELS_DIR.glob("*.pth"),
                                     key=lambda f: f.stat().st_mtime, reverse=True)
                self.selected = target
                self._rebuild = True
            except Exception as e:
                self.renameStatus.text = f"failed: {e}"
                self.renameStatus.color = color.red
        b.on_click = doRename

        runs = self.evalsForModel(p.name)
        if not runs:
            Text(parent=self, text="No evaluation recorded for this model.",
                 position=(x, 0.21), scale=0.85, color=color.orange)
            Text(parent=self, text="Run it from Evaluate Model to fill this in.",
                 position=(x, 0.17), scale=0.75, color=color.gray)
            return

        rows = [
            ("run length", lambda d: f"{d.get('duration', 0):.0f}s"),
            ("sign density", lambda d: f"x{int(d.get('signDensity', 1))}"),
            ("time on road", lambda d: f"{d.get('timeOnRoadPct', 0):.1f}%"),
            ("departures", lambda d: f"{int(d.get('departures', 0))}"),
            ("lane dev median", lambda d: f"{d.get('laneDevMedian', float('nan')):.2f}"),
            ("steer movement", lambda d: f"{d.get('steerSmoothness', 0):.0f}"),
            ("speed / limit", lambda d: f"{d.get('speedRatio', 0) * 100:.0f}%"),
            ("signs stopped", lambda d: f"{int(d.get('signsFullyStopped', 0))}/{int(d.get('signsEncountered', 0))}"),
            ("stopped in zone", lambda d: f"{int(d.get('stoppedInZone', 0))}"),
            ("stopped short", lambda d: f"{int(d.get('stoppedShort', 0))}"),
            ("resumed unaided", lambda d: f"{int(d.get('signsResumedUnaided', d.get('signsResumed', 0)))}"),
            ("your input", lambda d: f"{d.get('interventionPct', 0):.2f}%"),
        ]

        shown = runs[:len(cols)]
        for i, (label, _) in enumerate(shown):
            Text(parent=self, text=label[:14], position=(cols[i], 0.21),
                 scale=0.7, color=color.yellow)

        for j, (name, fn) in enumerate(rows):
            y = 0.16 - j * 0.042
            Text(parent=self, text=name, position=(x, y), scale=0.8, color=color.light_gray)
            for i, (_, d) in enumerate(shown):
                try:
                    v = fn(d)
                except Exception:
                    v = "-"
                Text(parent=self, text=v, position=(cols[i], y), scale=0.8, color=color.white)

        if len(runs) > len(cols):
            Text(parent=self, text=f"(+{len(runs) - len(cols)} older run(s) not shown)",
                 position=(x, 0.16 - len(rows) * 0.042 - 0.02), scale=0.7, color=color.gray)

    def input(self, key):
        if not self.models:
            return
        # Don't page while the cursor is in the rename box. Paging rebuilds the whole
        # panel, which destroys that field mid-frame -- and quite apart from the crash
        # that used to cause, losing a half-typed name to a stray scroll is obnoxious.
        f = self.renameField
        if f and (f.hovered or f.active):
            return
        if key == "scroll up":
            self.page -= 1
            self._rebuild = True
        elif key == "scroll down":
            self.page += 1
            self._rebuild = True

    def update(self):
        if self._rebuild:
            self._rebuild = False
            for c in list(self.children):
                Autopilot_sim.destroy_all(c)
            self._build()


# ============================================================== data check


def summariseDataset():
    """Scan Data/ without loading images. Cheap enough to run on the main thread."""
    sessions, totalRows, withFrames = 0, 0, 0
    speeds, pedals, steers, salience = [], [], [], []
    missingSalience = 0

    if not DATA_DIR.exists():
        return None

    for folder in sorted(DATA_DIR.iterdir()):
        if not folder.is_dir():
            continue
        sessions += 1
        csvFile = folder / (folder.name + " telemtry.csv")
        if not csvFile.exists():
            continue
        sal = {}
        salPath = folder / "sign_salience.json"
        if salPath.exists():
            try:
                sal = json.loads(salPath.read_text())
            except Exception:
                sal = {}
        rows = 0
        with open(csvFile) as f:
            r = csv.reader(f)
            next(r, None)
            for row in r:
                if not row or len(row) < 7 or row[1].strip() != "Center":
                    continue
                try:
                    speeds.append(float(row[3]))
                    pedals.append(float(row[6]))
                    steers.append(float(row[5]))
                except ValueError:
                    continue
                name = row[2].strip()
                if name in sal:
                    salience.append(sal[name])
                else:
                    salience.append(0)
                    missingSalience += 1
                rows += 1
        totalRows += rows
        if rows:
            withFrames += 1

    if not totalRows:
        return {"sessions": sessions, "withFrames": 0, "frames": 0}

    import numpy as np
    sp = np.array(speeds); pd = np.array(pedals); st = np.array(steers); sl = np.array(salience)
    return {
        "sessions": sessions,
        "withFrames": withFrames,
        "frames": totalRows,
        "minutes": totalRows / 10.0 / 60.0,          # center frames at 10 Hz
        "atRest": float((np.abs(sp) < 0.5).mean()),
        "resume": float(((np.abs(sp) < 0.5) & (pd > 0.1)).mean()),
        "braking": float(((np.abs(sp) >= 0.5) & (pd < -0.1)).mean()),
        "signVisible": float((sl >= 4).mean()),
        "signClose": float((sl >= 100).mean()),
        "turning": float((np.abs(st) > 0.3).mean()),
        "sharpTurn": float((np.abs(st) > 0.4).mean()),
        "steerZero": float((st == 0).mean()),
        "missingSalience": missingSalience,
        "meanSpeed": float(sp.mean()),
    }


class DataCheck(Entity):
    """Dataset summary: how much data, and how much of the rare stuff."""

    def __init__(self, onClose=None, **kwargs):
        kwargs.setdefault("parent", camera.ui)
        kwargs.setdefault("z", -10)
        super().__init__(**kwargs)
        self.onClose = onClose
        self.body = None
        self.summary = None
        self._scanned = False
        self.salienceThread = None
        self.salienceStatus = None
        self.salienceButton = None
        self._rescan = False

        self._rebuildShell()

    def _rebuildShell(self):
        """Panel chrome only. Called again after a rescan, so the Close button lives
        here rather than in __init__ -- rebuilding wipes every child."""
        _panel(self, 1.3, 0.94, 242)
        Text(parent=self, text="Dataset", position=(-0.63, 0.42), scale=1.5, color=color.cyan)
        Text(parent=self, text="scanning...", position=(-0.63, 0.36), scale=0.85,
             color=color.gray)

        b = Button(text="Close", parent=self, scale=(0.18, 0.075), position=(0.52, -0.42),
                   color=color.dark_gray, text_color=color.white)

        def shut():
            cb = self.onClose
            close(self)
            if cb:
                cb()
        b.on_click = shut

    def update(self):
        # Redraw once the background scoring pass finishes, so the numbers reflect it.
        if self._rescan:
            self._rescan = False
            self._scanned = False
            for c in list(self.children):
                Autopilot_sim.destroy_all(c)
            self._rebuildShell()
            return

        # Scan on the frame after construction so "scanning..." actually paints first.
        if self._scanned:
            return
        self._scanned = True

        s = summariseDataset()
        self.summary = s
        if not s:
            Text(parent=self, text="No Data/ directory found.", position=(-0.63, 0.25),
                 scale=1.0, color=color.red)
            return
        if not s.get("frames"):
            Text(parent=self, text=f"{s['sessions']} session folders, none containing frames.",
                 position=(-0.63, 0.25), scale=1.0, color=color.red)
            return

        Text(parent=self,
             text=f"{s['frames']:,} center frames  ({s['frames'] * 3:,} across all cameras)"
                  f"   ~{s['minutes']:.0f} min of driving",
             position=(-0.63, 0.33), scale=0.95, color=color.white)
        Text(parent=self,
             text=f"{s['withFrames']} of {s['sessions']} session folders contain data"
                  f"     mean speed {s['meanSpeed']:.1f} mph",
             position=(-0.63, 0.29), scale=0.8, color=color.gray)

        # Each row carries what the number means, since "1.3% resume" only reads as a
        # problem if you know it competes with 6% at-rest holds.
        rows = [
            ("stop sign visible", s["signVisible"], "any sign in frame"),
            ("stop sign close", s["signClose"], "close enough to act on"),
            ("braking while moving", s["braking"], "the approach to a stop"),
            ("stopped", s["atRest"], "at rest, mostly holding at a sign"),
            ("pulling away from rest", s["resume"], "rare class -- resuming on green"),
            ("turning (|steer| > 0.3)", s["turning"], "corners"),
            ("sharp turns (> 0.4)", s["sharpTurn"], "the rarest steering"),
            ("wheel exactly centred", s["steerZero"], "keyboard snaps to zero"),
        ]
        Text(parent=self, text="share of frames", position=(-0.63, 0.21), scale=0.9,
             color=color.light_gray)
        for i, (label, frac, note) in enumerate(rows):
            y = 0.15 - i * 0.055
            Text(parent=self, text=label, position=(-0.63, y), scale=0.85, color=color.white)
            Text(parent=self, text=f"{100 * frac:5.2f}%", position=(-0.18, y), scale=0.85,
                 color=color.yellow if frac < 0.02 else color.white)
            Entity(parent=self, model="quad", color=color.dark_gray,
                   scale=(0.30, 0.022), position=(0.13, y + 0.008), z=0)
            Entity(parent=self, model="quad",
                   color=color.red if frac < 0.02 else color.cyan,
                   scale=(max(0.002, 0.30 * min(1.0, frac / 0.25)), 0.022),
                   position=(-0.02, y + 0.008), z=-1, origin=(-0.5, 0))
            Text(parent=self, text=note, position=(0.31, y), scale=0.7, color=color.gray)

        Text(parent=self, text="bars are scaled to 25% -- red marks a class under 2%",
             position=(-0.63, -0.33), scale=0.7, color=color.gray)

        if s["missingSalience"]:
            self.salienceStatus = Text(
                parent=self,
                text=f"{s['missingSalience']:,} frames have no sign-salience score "
                     f"(new recordings need scoring).",
                position=(-0.63, -0.38), scale=0.75, color=color.orange)
            self.salienceButton = Button(
                text="Score them now", parent=self, scale=(0.26, 0.07),
                position=(-0.36, -0.44), color=color.orange, text_color=color.black)
            self.salienceButton.on_click = self.runSalience
        else:
            Text(parent=self, text="All frames scored for stop-sign salience.",
                 position=(-0.63, -0.38), scale=0.75, color=color.lime)

    def runSalience(self):
        """Score new frames from here instead of sending the user to a terminal.

        On a worker thread: it reads every unscored JPEG, which takes about a minute
        for a few thousand and would otherwise freeze the window.
        """
        if self.salienceThread is not None:
            return
        if self.salienceButton:
            self.salienceButton.enabled = False
        if self.salienceStatus:
            self.salienceStatus.text = "scoring... (see console for per-session progress)"
            self.salienceStatus.color = color.yellow

        import DataCollection

        def work():
            try:
                DataCollection.buildSignSalienceCache()
                self._rescan = True
            except Exception as e:
                print(f"SALIENCE ERROR: {e}")
                if self.salienceStatus:
                    self.salienceStatus.text = f"failed: {e}"
                    self.salienceStatus.color = color.red
            finally:
                self.salienceThread = None

        self.salienceThread = threading.Thread(target=work, daemon=True)
        self.salienceThread.start()
