# This will be what opens up immediately after launching program
#NOTE: This file is messy, very very messy, I should clean it up, but I don't really want to remove a lot of the junk in
    #case I use it for later bug testing. Might just leave it like this
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
from Autopilot_sim import WorldGenerator
import Training
import multiprocessing
import sys

# NOTE ON THE MISSING GUARD
# This file used to bail out with sys.exit(0) in non-MainProcess, to stop DataLoader
# workers opening blank Ursina windows. That worked, but it also killed the workers
# before they could load anything, which is why num_workers > 0 never functioned from
# this menu.
#
# Windows has no fork(): PyTorch starts workers with "spawn", and each worker re-imports
# whatever module is __main__ -- under the name __mp_main__, not __main__. So everything
# with a side effect (creating the app, building the UI, running the loop) lives inside
# the __main__ guard at the bottom of this file and simply does not execute in a worker.
# Function and class definitions up here are safe to re-import.
#
# Consequence: do not move app/UI construction back to module level, or training from
# the Train button will start spawning windows again.

app = None
menu_parent = None
simulation_parent = None
title = None
hoverHelp = None



def loadWorld(stopSignDensity=1):
    """stopSignDensity > 1 widens the stop-sign spawn roll for data collection.
    MIN_STOP_SIGN_SPACING still applies, so signs stay realistically far apart --
    at density 3 the median gap is ~650 units, enough to reach cruise between stops."""
    import random as _random

    menu_parent.enabled = False
    simulation_parent.enabled = True

    # Reseed from the OS. evaluateModel() pins the global RNG to EVAL_WORLD_SEED so every
    # model drives an identical course; without this, a normal sim started afterwards
    # inherits that seed and silently replays the evaluation track.
    _random.seed()

    global activeWorld
    activeWorld = WorldGenerator(stopSignDensity=stopSignDensity, parent=simulation_parent,
                                 onExitToMenu=returnToMenu)
    return activeWorld


activeWorld = None


def returnToMenu():
    """Tear the world down and show the menu again."""
    global activeWorld
    if activeWorld is not None:
        try:
            activeWorld.teardown()
        except Exception as e:
            print(f"MENU: world teardown issue ({e})")
        activeWorld = None
    simulation_parent.enabled = False
    menu_parent.enabled = True

TRAIN_EPOCHS = 100
TRAIN_WORKERS = 6      # matches train.py; measured ~7.7x faster than single-process


activeScreen = None


def closeScreen():
    """Dismiss whatever overlay is open and re-enable the menu."""
    global activeScreen
    import MenuScreens
    if activeScreen is not None:
        MenuScreens.close(activeScreen)
        activeScreen = None
    menu_parent.enabled = True


def runTraining():
    """Train on a worker thread so the window stays responsive.

    Safe to spawn DataLoader workers from here now that this module has no import-time
    side effects -- workers re-import it as __mp_main__ and skip the __main__ block.
    """
    global activeScreen
    import MenuScreens
    if activeScreen is not None:
        return
    menu_parent.enabled = False
    print(f"Training for up to {TRAIN_EPOCHS} epochs with {TRAIN_WORKERS} loader workers...")
    activeScreen = MenuScreens.TrainingProgress(TRAIN_EPOCHS, TRAIN_WORKERS,
                                                onFinish=closeScreen)


def startSimulation():
    """Ask how to build the world, then build it."""
    global activeScreen
    import MenuScreens
    if activeScreen is not None:
        return
    menu_parent.enabled = False

    def begin(density):
        global activeScreen
        activeScreen = None
        loadWorld(stopSignDensity=density)

    activeScreen = MenuScreens.SimStartPopup(onStart=begin, onCancel=closeScreen)


def viewCurrentModels():
    global activeScreen
    import MenuScreens
    if activeScreen is not None:
        return
    menu_parent.enabled = False
    activeScreen = MenuScreens.ModelBrowser(onClose=closeScreen)


def checkData():
    global activeScreen
    import MenuScreens
    if activeScreen is not None:
        return
    menu_parent.enabled = False
    activeScreen = MenuScreens.DataCheck(onClose=closeScreen)



def evaluateModel():
    """Drive a model on a fixed seeded course and score it quantitatively."""
    import random as _random
    import Evaluation

    menu_parent.enabled = False
    simulation_parent.enabled = True

    def makeWorld(signDensity=1):
        # worldSeed gives course layout its own generator, so chunk k is identical no
        # matter when it gets built. Seeding the global `random` is not sufficient --
        # WeatherSystem consumes the global stream every frame.
        # Called fresh for each run: a world that has already been driven is a
        # different course.
        _random.seed(Evaluation.EVAL_WORLD_SEED)
        world = WorldGenerator(worldSeed=Evaluation.EVAL_WORLD_SEED,
                               stopSignDensity=signDensity, parent=simulation_parent)
        world.weatherEnabled = False
        try:
            world.weather.set_weather('clear')
        except Exception:
            pass
        return world

    def backToMenu():
        simulation_parent.enabled = False
        menu_parent.enabled = True

    Evaluation.startEvaluationFlow(makeWorld, onExit=backToMenu)

#note, this function was simply for testing and is not used anywhere in the program, however, I left it for later debugging if needed

def tempForCheckingData():
    # import pandas as pd
    #
    # all_rows = []
    # data_dir = Path("Data")
    # for session_folder in data_dir.iterdir():
    #     csv_path = session_folder / f"{session_folder.name} telemtry.csv"
    #     if csv_path.exists():
    #         try:
    #             df = pd.read_csv(csv_path)
    #             all_rows.append(df)
    #         except pd.errors.EmptyDataError:
    #             print(f"skipping empty/corrupt CSV: {csv_path}")
    #             continue
    #
    # full_df = pd.concat(all_rows, ignore_index=True)
    # print(f"total frames: {len(full_df)}")
    #
    # near_zero = full_df[full_df['speed'].abs() < 0.5]
    # print(f"frames near-zero speed: {len(near_zero)}  ({len(near_zero) / len(full_df):.2%})")
    # print(near_zero['pedal'].describe())
    #///////////////////////////////////////////////////////////
    # import pandas as pd
    # from pathlib import Path
    #
    # for session_folder in Path("Data").iterdir():
    #     csv_path = session_folder / f"{session_folder.name} telemtry.csv"
    #     if not csv_path.exists():
    #         continue
    #     try:
    #         df = pd.read_csv(csv_path)
    #     except pd.errors.EmptyDataError:
    #         continue
    #
    #     center_df = df[df['camera'] == 'Center'].sort_values('frame').reset_index(drop=True)
    #     is_stopped = center_df['speed'].abs() < 0.5
    #
    #     # count contiguous runs of "stopped" frames -- each run is one discrete stop event
    #     episode_id = (is_stopped != is_stopped.shift()).cumsum()
    #     episodes = center_df[is_stopped].groupby(episode_id[is_stopped])
    #     lengths = episodes.size()
    #
    #     if len(lengths) > 0:
    #         print(f"{session_folder.name}: {len(lengths)} stop episodes, lengths={list(lengths)}")
    # import torch
    # from Training import Net
    # from DataCollection import DrivingDataset
    #
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = Net().to(device)
    # model.load_state_dict(torch.load("Models/Model_20260801-230609.pth", map_location=device))
    # model.eval()
    #
    # dataset = DrivingDataset()
    # target_session = "20260801-183421"
    # low_speed_indices = [
    #     i for i, (path, s) in enumerate(zip(dataset.image_paths, dataset.speeds))
    #     if target_session in str(path) and abs(s) < 0.5
    # ]
    # print(f"testing {len(low_speed_indices)} examples from the good session")
    #
    # with torch.no_grad():
    #     for i in low_speed_indices[::5][:20]:  # sample spread across the episodes, not just the first few
    #         image, speed, speed_limit, target = dataset[i]
    #         pred = model(image.unsqueeze(0).to(device), speed.unsqueeze(0).to(device),
    #                      speed_limit.unsqueeze(0).to(device))
    #         print(f"true_pedal={target[1]:.2f}  pred_pedal={pred[0, 1].item():.2f}")
    # /////////////////////////////////////////
    # import pandas as pd
    # from pathlib import Path
    #
    # for session_folder in Path("Data").iterdir():
    #     csv_path = session_folder / f"{session_folder.name} telemtry.csv"
    #     if not csv_path.exists():
    #         continue
    #     try:
    #         df = pd.read_csv(csv_path)
    #     except pd.errors.EmptyDataError:
    #         continue
    #
    #     center_df = df[df['camera'] == 'Center'].sort_values('frame').reset_index(drop=True)
    #     is_stopped = center_df['speed'].abs() < 0.5
    #     episode_id = (is_stopped != is_stopped.shift()).cumsum()
    #
    #     for ep_id, group in center_df[is_stopped].groupby(episode_id[is_stopped]):
    #         if len(group) < 3:
    #             continue
    #         pedals_in_episode = group['pedal'].values
    #         print(f"{session_folder.name} episode {ep_id}: len={len(group)}, "
    #               f"pedal trace: {[round(p, 2) for p in pedals_in_episode]}")

    # ////////////////////////////////////////////////
    import torch
    from Training import Net
    from DataCollection import DrivingDataset
    from torch.utils.data import DataLoader
    import numpy as np

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)
    model.load_state_dict(torch.load("Models/Extra specified data.pth", map_location=device))
    model.eval()

    dataset = DrivingDataset()
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    all_true, all_pred = [], []
    with torch.no_grad():
        for images, speeds, speed_limits, targets in loader:
            images, speeds, speed_limits = images.to(device), speeds.to(device), speed_limits.to(device)
            preds = model(images, speeds, speed_limits)
            all_true.extend(targets[:, 0].tolist())
            all_pred.extend(preds[:, 0].cpu().tolist())

    print(f"true steering  std: {np.std(all_true):.4f}, range: [{min(all_true):.2f}, {max(all_true):.2f}]")
    print(f"pred steering  std: {np.std(all_pred):.4f}, range: [{min(all_pred):.2f}, {max(all_pred):.2f}]")

    steering_array = np.array(dataset.steering_angles)
    speed_array = np.array(dataset.speeds)

    sharp_high_speed = np.sum((np.abs(steering_array) > 0.5) & (np.abs(speed_array) > 40))
    print(
        f"frames with sharp steering AND high speed: {sharp_high_speed} / {len(steering_array)} ({sharp_high_speed / len(steering_array):.2%})")

#note, this function was simply for testing and is not used anywhere in the program, however, I left it for later debugging if needed
def runWheelChecksForImbalance():
    # from DataCollection import DrivingDataset
    # import numpy as np
    # dataset = DrivingDataset()
    # steering_array = np.array(dataset.steering_angles)
    # speed_array = np.array(dataset.speeds)
    #
    # left_strong = np.sum(steering_array < -0.5)
    # right_strong = np.sum(steering_array > 0.5)
    # print(f"strong left-turn frames (< -0.5): {left_strong}")
    # print(f"strong right-turn frames (> 0.5): {right_strong}")
    # print(f"ratio left:right = {left_strong / max(right_strong, 1):.2f} : 1")
    #
    # sharp_fast_left = np.sum((steering_array < -0.5) & (speed_array > 40))
    # sharp_fast_right = np.sum((steering_array > 0.5) & (speed_array > 40))
    # print(f"sharp LEFT + high speed: {sharp_fast_left}")
    # print(f"sharp RIGHT + high speed: {sharp_fast_right}")
    #
    # left_values = steering_array[steering_array < 0]
    # right_values = steering_array[steering_array > 0]
    # print("\n\n\n")
    #
    # print(f"left turns  - count: {len(left_values)}, most extreme: {left_values.min():.2f}")
    # print(f"right turns - count: {len(right_values)}, most extreme: {right_values.max():.2f}")
    #
    # import re
    # from collections import Counter
    # from pathlib import Path
    #
    # # if your recovery frames were recorded in dedicated session(s), check their spread
    # recovery_sessions = [p for p in Path("Data").iterdir() if
    #                      p.is_dir()]  # narrow this to your actual recovery session folder(s)
    # for session in recovery_sessions:
    #     csv_path = session / f"{session.name} telemtry.csv"
    #     if csv_path.exists() and csv_path.stat().st_size > 0:
    #         import pandas as pd
    #         try:
    #             df = pd.read_csv(csv_path)
    #             if not df.empty and 'steering' in df.columns:
    #                 print(
    #                     f"{session.name}: {len(df)} frames, steering range [{df['steering'].min():.2f}, {df['steering'].max():.2f}]")
    #         except pd.errors.EmptyDataError:
    #             continue

    import pandas as pd
    from pathlib import Path

    session = "20260804-222242"
    csv_path = Path("Data") / f"Session {session}" / f"Session {session} telemtry.csv"
    df = pd.read_csv(csv_path)
    df_center = df[df['camera'] == 'Center']

    near_zero_speed = (df_center['speed'].abs() < 0.5).sum()
    high_speed_sharp_turn = ((df_center['speed'].abs() > 40) & (df_center['steering'].abs() > 0.5)).sum()
    likely_recovery_loose = ((df_center['speed'].abs() >= 0.5) & (df_center['speed'].abs() <= 40) & (
                df_center['steering'].abs() > 0.15)).sum()
    ordinary = len(df_center) - near_zero_speed - high_speed_sharp_turn - likely_recovery_loose

    print(f"total frames in session: {len(df_center)}")
    print(f"near-zero speed (stop-sign related): {near_zero_speed}")
    print(f"high speed + sharp turn: {high_speed_sharp_turn}")
    print(f"likely recovery (moderate speed, large steering): {likely_recovery_loose}")
    print(f"everything else: {ordinary}")

#note, this function was simply for testing and is not used anywhere in the program, however, I left it for later debugging if needed
def checkPredictionSpread():
    import torch
    import numpy as np
    from Training import Net
    from DataCollection import DrivingDataset
    from torch.utils.data import DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)
    model.load_state_dict(torch.load("Models/MorePara no filp.pth", map_location=device))
    model.eval()

    dataset = DrivingDataset()
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    # all_true, all_pred = [], []
    # with torch.no_grad():
    #     for images, speeds, speed_limits, targets in loader:
    #         images, speeds, speed_limits = images.to(device), speeds.to(device), speed_limits.to(device)
    #         preds = model(images, speeds, speed_limits)
    #         all_true.extend(targets[:, 0].tolist())
    #         all_pred.extend(preds[:, 0].cpu().tolist())
    #
    # print(f"true steering  std: {np.std(all_true):.4f}, range: [{min(all_true):.2f}, {max(all_true):.2f}]")
    # print(f"pred steering  std: {np.std(all_pred):.4f}, range: [{min(all_pred):.2f}, {max(all_pred):.2f}]")
    #//////////////////////
    # all_true_pedal, all_pred_pedal = [], []
    # with torch.no_grad():
    #     for images, speeds, speed_limits, targets in loader:
    #         images, speeds, speed_limits = images.to(device), speeds.to(device), speed_limits.to(device)
    #         preds = model(images, speeds, speed_limits)
    #         all_true_pedal.extend(targets[:, 1].tolist())
    #         all_pred_pedal.extend(preds[:, 1].cpu().tolist())
    #
    # print(
    #     f"true pedal  std: {np.std(all_true_pedal):.4f}, range: [{min(all_true_pedal):.2f}, {max(all_true_pedal):.2f}]")
    # print(
    #     f"pred pedal  std: {np.std(all_pred_pedal):.4f}, range: [{min(all_pred_pedal):.2f}, {max(all_pred_pedal):.2f}]")
#     ///////////////////////////////////
    target = "20260731-132821 Right frame_001254.jpg"
    for i, path in enumerate(dataset.image_paths):
        if target in path:
            idx_red = i

    target = "20260731-132821 Right frame_001255.jpg"
    for i, path in enumerate(dataset.image_paths):
        if target in path:
            idx_green = i



    for i in [idx_red, idx_green]:   # fill in real indices from your CSV
        image, speed, speed_limit, target = dataset[i]
        pred = model(image.unsqueeze(0).to(device), speed.unsqueeze(0).to(device), speed_limit.unsqueeze(0).to(device))
        print(f"speed={speed.item():.2f}  true_pedal={target[1]:.2f}  pred_pedal={pred[0,1].item():.2f}")


#note, this function was simply for testing and is not used anywhere in the program, however, I left it for later debugging if needed
def checkCamera(): #NOTE IMPROVE THIS LATER TO PROMPT FOR SEISSION AND PHOTO
    # menu_parent.enabled = False
    # Open menu of all models currently saved
    import cv2
    import matplotlib.pyplot as plt
    from DataCollection import DrivingDataset

    img = cv2.imread("Data/Session 20260801-183421/20260801-183421 Center frame_000420.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    processed = DrivingDataset.preprocess(img)
    processed_display = (processed + 1.0) / 2.0  # undo the [-1,1] normalization just for viewing

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(img);
    axes[0].set_title("raw")
    axes[1].imshow(processed_display);
    axes[1].set_title("what the network actually sees")
    plt.savefig("crop_check5.png")


def buildMenu():
    """All Ursina construction lives here, called only from the __main__ guard.

    Nothing in this function may run at import time -- see the note at the top of the
    file. A DataLoader worker re-imports this module and must reach the end of it
    without creating a window.
    """
    global app, menu_parent, simulation_parent, title, hoverHelp

    import MenuScreens

    app = Ursina()
    menu_parent = Entity(parent=camera.ui)
    simulation_parent = Entity(enabled=False)
    title = Text(text="Autopilot Sim", parent=menu_parent, y=0.40, x=-0.30,
                 scale=2, origin=(0, 0))

    # Description panel on the right, filled in on hover.
    hoverHelp = MenuScreens.HoverHelp(parent=menu_parent)

    entries = [
        ("Start Simulation", color.red, startSimulation,
         "Drive it yourself",
         "Opens a world you control.\n\nAsks first how densely stop\nsigns should spawn - dense is\nfor collecting approach data.\n\nIn the sim:\n  0  autopilot on/off\n  -  choose a model\n  9  record\n  8  corrections only\n  1/2  camera\n  esc  weather, camera, exit"),
        ("Train Model", color.blue, runTraining,
         "Train on recorded data",
         f"Runs up to {TRAIN_EPOCHS} epochs with\n{TRAIN_WORKERS} loader workers.\n\nRuns on a background thread,\nso the window stays responsive\nand shows progress.\n\nSaves BEST, BESTSTEER and EMA\ncheckpoints. Expect tens of\nminutes."),
        ("Evaluate Model", color.green, evaluateModel,
         "Score models on a fixed course",
         "Every model drives the same\nseeded course, so results are\ncomparable.\n\nQueue several and they run back\nto back, then a comparison table\nshows them side by side.\n\nUse 240s+ and dense signs to get\na meaningful stop-sign sample."),
        ("View Current Models", color.yellow, viewCurrentModels,
         "Browse saved models",
         "Lists every .pth in Models/\nwith its size and how many\nevaluations it has.\n\nSelect one to see its results\nand compare its runs."),
        ("Data check", color.orange, checkData,
         "Inspect the dataset",
         "How many frames you have and\nhow much of the rare material:\nstop signs, braking, resuming,\nsharp turns.\n\nClasses under 2% are flagged -\nthose are the ones that tend to\nbe learned badly."),
    ]

    for i, (label, col, action, helpTitle, helpBody) in enumerate(entries):
        b = Button(text=label, text_color=color.black, parent=menu_parent,
                   scale=(0.34, 0.085), x=-0.30, y=0.18 - i * 0.105,
                   on_click=action, color=col)
        MenuScreens.attachHover(b, hoverHelp, helpTitle, helpBody)

    return app


if __name__ == '__main__':
    multiprocessing.freeze_support()
    buildMenu().run()


