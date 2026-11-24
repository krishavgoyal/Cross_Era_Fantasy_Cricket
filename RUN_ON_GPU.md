# Run the full pipeline (WSL Ubuntu, RTX 5060)

## Once: give WSL enough RAM
By default WSL gets half the laptop's RAM (7.4 GB on a 16 GB laptop); loading the models needs more and the
process is killed without an error. In **Windows PowerShell**:
```
@"
[wsl2]
memory=12GB
swap=16GB
"@ | Out-File -Encoding ascii $env:USERPROFILE\.wslconfig
wsl --shutdown
```
Reopen Ubuntu; `free -h` should show about 11Gi total.

## Once: generation env
```
cd <repo folder>
uv venv ~/gen --python 3.11 && source ~/gen/bin/activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install -r requirements-generation.txt "numpy<3" "opencv-python<5" "Pillow<12" "scipy<2" onnxruntime "huggingface-hub<1.0" pytest
uv pip install --no-deps easy-dwpose==1.0.2
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Every run
```
source ~/gen/bin/activate
export HF_HUB_DISABLE_XET=1 PYTHONUNBUFFERED=1
python master_pipeline.py --clip-id bowler_delivery_001 --image data/historical_images/modern_bowler.jpeg --role bowler --auto-window
python master_pipeline.py --clip-id good_cover_drive_final --image data/historical_images/bradman.jpg --role batsman
```
First run downloads ~6 GB of models. Results: `runs/<clip>_<role>/`.

## Without a GPU: use videos Student 2 already generated
```
python master_pipeline.py --clip-id bowler_delivery_001 --image data/historical_images/modern_bowler.jpeg --role bowler --generated outputs/m3_bowler_delivery.mp4
python master_pipeline.py --clip-id good_cover_drive_final --image data/historical_images/bradman.jpg --role batsman --generated data/generated_clips/test_run/m3_magicanimate.mp4
```
Runs colour, 24 fps, demo and evaluation on CPU in under a minute.
