# AutoNet-NAS Vision Studio

Detect people, classify photos, and compare CNN architecture experiments in Streamlit.

## Update an existing Windows installation

Stop Streamlit with `Ctrl+C` and extract `AutoNetNAS_Updated.zip`. Open PowerShell inside the extracted folder that contains `apply_update.py`, then run these commands.

```powershell
$installer = Join-Path $PWD.Path "apply_update.py"
Set-Location "$env:USERPROFILE\OneDrive\Desktop\AutoNet-NAS"
.\.venv\Scripts\python.exe $installer --project .
```

The installer verifies the packaged files and backs up the files it replaces. Existing continued-training checkpoints and evaluation outputs take priority over the bundled defaults. You can apply this package directly to the original project or to the previous prediction update.

## Open Vision Studio on Windows

Paste this command into PowerShell from any folder. It starts the project with its own Python environment and opens the app in your browser.

```powershell
& "$env:USERPROFILE\OneDrive\Desktop\AutoNet-NAS\Start_AutoNet.cmd"
```

You can also double-click `Start_AutoNet.cmd` in the project folder. Press `Ctrl+C` in the terminal to stop the app.

## Scan a photo

Open **Image Studio**, choose a scan mode, upload a JPG, PNG, or WEBP, and select **Scan image**.

| Mode | What it does |
| --- | --- |
| Smart scan | Checks for people first. If it finds none above the score threshold, it classifies the main object. |
| People | Finds people, draws a box around each detection, and shows a count. You can download a marked copy. |
| Objects | Classifies one prominent object using 1,000 ImageNet labels. |
| CIFAR-10 | Runs one of the trained CNNs on a photo or a labelled test example. The ten classes are airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. |

For people in a solo or group photo, use **Smart scan** or **People**. Adjust the minimum person score under **Detection settings** to review stronger or weaker detections. The app corrects phone photo orientation before scanning.

Person detection uses pretrained SSDLite with a MobileNetV3 backbone and COCO weights. Object classification uses pretrained EfficientNet-B0. Both sets of weights are bundled for offline use after the Python dependencies are installed.

For a reproducible classification check, choose **CIFAR-10 → Labelled test example**. The fixed random sample includes mistakes. Each image displays its dataset label separately from its prediction.

The **Model Results**, **Architecture Lab**, and **Training History** pages show the saved CIFAR-10 experiments.

## Verified results in this update

All four checkpoints were evaluated on the same 10,000 labelled CIFAR-10 test images.

| Model | Correct predictions | Test accuracy |
| --- | ---: | ---: |
| Original NAS | 3,439 / 10,000 | 34.39% |
| Improved NAS | 5,440 / 10,000 | 54.40% |
| Advanced CNN | 6,772 / 10,000 | 67.72% |
| Continued CNN | 7,060 / 10,000 | **70.60%** |

The continuation trained for five more epochs on 48,000 images with batch size 64. Its fixed validation score rose from 67.05% to 71.20%. Checkpoint selection used validation accuracy; the test set was evaluated afterward. The gain over Advanced CNN is **2.88 percentage points** (288 additional correct test predictions).

The test benchmark measures CIFAR-10 performance. The pretrained models have separate functional checks in `results/person_detection_check.json` and `results/photo_model_check.json`.

## Scan from the terminal

The default command classifies an everyday photo. Use `--mode smart` to check for people first or `--mode people` for detection only.

```powershell
python -m src.predict_file "C:\path\to\photo.jpg"
python -m src.predict_file "C:\path\to\photo.jpg" --mode smart --output marked.png
```

The second command writes `marked.png` when it detects people.

## Verify the saved model

Your existing `data` folder is used. This command evaluates the current Continued CNN, or Advanced CNN when no Continued CNN exists, on all 10,000 CIFAR-10 test images:

```powershell
python -m src.verify_predictions
```

Verify every saved experiment:

```powershell
python -m src.verify_predictions --all
```

Add `--download` if CIFAR-10 is absent. Results include `results/verified_evaluation.json`, confusion-matrix CSV files, 24 labelled PNG examples, and `results/prediction_examples.png`. The report records checkpoint hashes so it cannot silently describe different weights.

## Continue training the CNN

```powershell
python -m src.continue_training --epochs 10 --batch-size 64
```

This starts from `models/continued_cnn.pth` when present, otherwise from `models/advanced_cnn.pth`. It uses up to 48,000 training images while preserving the original 2,000 validation images. Batch size is 64 instead of the previous 8. The learning rate starts at 0.0003 and follows a cosine schedule. Default CPU threads are capped at four; use `--threads` to change this.

The best checkpoint is chosen on validation accuracy. If training never improves it, the starting weights are retained. The test set is evaluated after checkpoint selection. Training does not overwrite the original NAS, Improved NAS, or Advanced CNN weights.

New output files:

- `models/continued_cnn.pth`
- `results/continued_history.json`
- `results/continued_evaluation.json`

Stopping with Ctrl+C retains the saved best model and evaluates it. Running the command again fine-tunes those saved weights with a fresh optimizer. It does not restore an interrupted optimizer state.

## Fresh environment

From the extracted ZIP folder, create a virtual environment and install the dependencies. The package uses the PyTorch 2.15.0 CPU and Torchvision 0.29.2 CPU versions from the original project.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\Start_AutoNet.cmd
```

Optional regression checks:

```powershell
python -m pip install pytest
python -m pytest tests
```

## Model sources

- Original CNN weights and experiment records: the uploaded AutoNet-NAS project.
- EfficientNet-B0: Torchvision `EfficientNet_B0_Weights.IMAGENET1K_V1`.
- Person detection: Torchvision `SSDLite320_MobileNet_V3_Large_Weights.COCO_V1`.
- Official photo weights: https://download.pytorch.org/models/efficientnet_b0_rwightman-7f5810bc.pth
- Official model documentation: https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b0.html
- CIFAR-10: https://www.cs.toronto.edu/~kriz/cifar.html

CIFAR-10 benchmark evidence is in `results/verified_evaluation.json`, which records the checkpoint hash for each model.
