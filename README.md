# DDPM
## Folder structure
    ```
    .
    ├── scripts/
    │   ├── _pycache_/
    │   ├── lightning_logs/
    │   ├── dataset.py
    │   ├── diffusion.py
    │   ├── main.py
    │   ├── model.py
    │   └── parse_argue.py
    ├── train_ckpt/
    ├── images/
    ├── mnist/
    ├── README.md
    ├── mnist.npz
    ├── 311513015.png
    └── requirements.txt
    ```

## Environment
- Python 3.6 or later version
    ```sh
    pip install -r requirements.txt
    ```

## Train
```sh
python ./scripts/main.py --train
```

## Sample
Before you make a prediction, go to check ./train_ckpt folder and get the i-th epoch you want.
```sh
python ./scripts/main.py --test --epoch <i-th epoch>
```
The output images is in `images` folder.