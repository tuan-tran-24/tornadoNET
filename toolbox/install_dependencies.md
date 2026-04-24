# Install Dependencies to Use ArcGIS Toolboxes

1. Open ArcGIS Python Command Prompt

2. Activate your cloned environment
```
activate arcgispro-py3-clone
```

3. Go to your project folder
```
cd /d C:\Path\To\Your\Project
```

4. Upgrade pip tools
```
python -m pip install --upgrade pip setuptools wheel
```

5. Install dependencies
```
python -m pip install Pillow
python -m pip install --no-deps torch torchvision
python -m pip install ^
timm ^
torchcam ^
albumentations ^
opencv-python ^
matplotlib ^
pandas ^
scikit-learn ^
tqdm ^
ipython ^
jupyter ^
ipykernel ^
notebook
```
