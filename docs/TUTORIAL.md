# Robot Learning with Isaac Lab and Isaac Sim: An End-to-End Guide

![chicgrasp_isaacsim workflow](diagrams/TUTORIAL_FLOWCHART_d72ed720.png)

## 0. Get the repository

Install [Git for Windows](https://git-scm.com/downloads/win) and
[Git LFS](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage), then open a new PowerShell window.

Copy the repo's URL and paste below.

```powershell
$repositoryUrl = Read-Host "Paste this repository's HTTPS URL"
git clone $repositoryUrl chicgrasp_isaacsim
cd chicgrasp_isaacsim
git lfs install --local
git lfs pull
git lfs fsck
```

After obtaining the repository, complete NVIDIA's installation using the official guides below. This tutorial then walks through configuring this project, reviewing the scene, recording demonstrations, training a new model, and testing that model with three videos. It describes future commands, not a completed runtime validation.

- [Isaac Sim 5.1 workstation installation guide](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_workstation.html)
- [Isaac Lab v2.3.2 binary-installation guide](https://isaac-sim.github.io/IsaacLab/v2.3.2/source/setup/installation/binaries_installation.html)

### Install Python 3.11

Install **Python 3.11 (64-bit)** using the
[official Python installation guide for Windows](https://docs.python.org/3.11/using/windows.html#installation-steps).
This is the Python version used with this project's Isaac Sim 5.1 setup.

Check your installation:

```powershell
python --version
```

It should show `Python 3.11.x`. If another version is selected, use `py -3.11`
in place of `python` in the commands below.

## 1. Set up the project

```powershell
python setup_project.py
```

If setup reports missing project packages, run:

```powershell
python setup_project.py --install-extras
```

Then check the configuration:

```powershell
python setup_project.py --check
```

Continue when the status is `READY`. The scripts use the configured Isaac Sim
Python automatically.

If an installation is not found, supply your folders (replace these examples):

```powershell
python setup_project.py --isaac-sim "C:\isaacsim" --isaac-lab "C:\robotics\IsaacLab"
```

To change previously saved installation paths, replace the example folders and run:

```powershell
python setup_project.py --isaac-sim "C:\isaacsim" --isaac-lab "C:\robotics\IsaacLab" --update-config
```

For other local options, run `python setup_project.py --help`.

## 2. Create your experiment configuration

Choose a new experiment name, using letters, numbers, hyphens or underscores:

```powershell
$name = Read-Host "Experiment name"
$experiment = python create_experiment.py $name --json | ConvertFrom-Json
```

This creates `configs/experiments/<name>.json` and assigns all output paths under
`research_outputs/<name>/`. Existing experiments are never overwritten.
Keep this PowerShell window open: the commands below use `$experiment` to select
the generated configuration and paths automatically.

## 3. Keep the default setup or customize it

Keep the supplied setup to use the default table scene, demonstration and camera
presets. Training and test positions are loaded automatically from the saved
position sets.

For a compatible custom scene, edit [scene_setup.py](../scene_setup.py).

| Optional change | File |
| --- | --- |
| Scene setup | [scene_setup.py](../scene_setup.py) |
| Demonstration motion | [scripted_run.py](../scripted_run.py) |
| HAND, LEFT_HD and RIGHT_HD cameras | [camera presets](../camera_presets/) |

Actions are `[x, y, z, gL, gR]`: XYZ is the TCP target in robot-base metres;
jaw values are `1 = open`, `0 = closed`.

## 4. Load and approve the scene

Show the scene-loading plan in PowerShell without opening Isaac Sim:

```powershell
python load_scene.py --config $experiment.config
```

Open the scene in Isaac Sim for visual inspection and save the inspection results:

```powershell
python load_scene.py --config $experiment.config --execute --output $experiment.outputs.scene
```

Click **Finish inspection** in the **Scene inspection** window, then wait for
PowerShell to return. Use this button instead of the window X or Ctrl+C. Check `summary.json` in the printed
`output_directory` for `completion: COMPLETED`, then approve the latest inspection:

```powershell
python load_scene.py --config $experiment.config --approve $experiment.outputs.scene
```

Reuse the same inspection and approval commands each time. After changing
`scene_setup.py`, `scripted_run.py` or camera preset files, inspect and approve again.

### 4b. Watch the scripted demonstration

After approving the scene, open a visible preview:

```powershell
python scripted_run.py --config $experiment.config --position R000002 --preview
```

Click **Start demonstration**, watch the full motion, then click **Finish preview**.
Use **Stop preview** to end early.

## 5. Record demonstrations

The recorder executes the script in the background and saves training images and
actions. It does not create a playback video. By default, it records one episode
at each of the 50 supplied training positions:

```powershell
python recorder.py --config $experiment.config
python recorder.py --config $experiment.config --execute --output $experiment.outputs.collection
```

For fewer demonstrations, choose a count from 1 to 50:

```powershell
$demonstrations = 5
python recorder.py --config $experiment.config --limit $demonstrations
python recorder.py --config $experiment.config --limit $demonstrations --execute --output $experiment.outputs.collection
```

## 6. Train the model

The workflow uses diffusion. To use another model, connect its training,
prediction and checkpoint functions to the workflow while keeping the same input
and action formats. Scene setup, recording and live evaluation can then be reused.

Choose the total number of optimizer updates, then check the plan and start training:

```powershell
$updates = 142858
python train.py --config $experiment.config --updates $updates
python train.py --config $experiment.config --execute --dataset $experiment.dataset --updates $updates --output $experiment.outputs.training
```

Recovery state is saved to `model/LAST.pt` every 1,000 updates and at
the final update.

### Resume training after an interruption or shutdown

Open PowerShell and return to your repository folder. Check setup; reinstallation
is normally unnecessary. Select your existing experiment again because shell
variables do not survive a restart:

```powershell
cd "C:\path\to\chicgrasp_isaacsim"
python setup_project.py --check
$name = "my-experiment"
$config = "configs/experiments/$name.json"
$runRoot = "research_outputs/$name"
python train.py --config $config --resume "$runRoot/training001/model/LAST.pt" --execute --output "$runRoot/training002"

# Select the resumed model for Step 7.
$experiment = @{
    config = $config
    artifact = "$runRoot/training002/model/artifact.json"
    outputs = @{ live = "$runRoot/live001" }
}
```

Replace the folder and experiment name with yours. Resume restores the saved
training state and original total update target; it does not add another full
budget or restart warmup. Keep the original dataset and checkpoint folder intact.

## 7. Run the trained model and save the videos

### Set up video recording

Install [FFmpeg](https://ffmpeg.org/download.html), including `ffprobe.exe`.
Keep both executables in the same folder.

If setup has not found FFmpeg, replace the example path below with yours:

```powershell
python setup_project.py --ffmpeg "C:\Tools\ffmpeg\bin\ffmpeg.exe" --update-config --require-video
```

Check video readiness:

```powershell
python setup_project.py --check --require-video
```

Continue when `status` and `video_status` are `READY`.

### Run an episode

Test one position with one seed:

```powershell
python live_run.py --config $experiment.config --position HC07 --seed 15345
python live_run.py --config $experiment.config --execute --position HC07 --seed 15345 --artifact $experiment.artifact --output $experiment.outputs.live
```

The videos are saved here:

```text
research_outputs/<name>/live001/HC07_seed15345/native/capture/experiment/media/
    HAND.mp4
    LEFT_HD.mp4
    RIGHT_HD.mp4
    MEDIA_RECEIPT.json
```

Video settings are 1280 x 960, 30 fps and H.264 CRF 16 by default.
