# AI Neural Style Transfer (AdaIN)

A web app and training pipeline for fast, arbitrary neural style transfer, built around
**Adaptive Instance Normalization (AdaIN)** — the method from
[*Huang & Belongie, "Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization", ICCV 2017*](https://arxiv.org/abs/1703.06868).

Upload any content image and any style image, and the model blends them into a new,
stylized image in a single forward pass — no per-image optimization loop required.

## Demo

| Content | Style | Result |
|---|---|---|
| ![content](Demo_IO_Images/i-p/content.jpg) | ![style](Demo_IO_Images/i-p/style_1.png) | ![output](Demo_IO_Images/o-p/output_style_1.jpg) |
| ![content](Demo_IO_Images/i-p/content.jpg) | ![style](Demo_IO_Images/i-p/style_2.jpg) | ![output](Demo_IO_Images/o-p/output_style_2.jpg) |

## How it works

AdaIN performs style transfer entirely in feature space, using a fixed pretrained
encoder and a small trainable decoder:

![AdaIN architecture](NST_Code/adain_algo.png)

1. A **VGG-19 encoder** (frozen, pretrained, never updated) extracts feature maps from
   both the content image and the style image.
2. An **AdaIN layer** rescales the content feature map's channel-wise mean and
   standard deviation to match the style feature map's mean and standard deviation.
   This single statistical alignment step is what transfers the "style" — no learned
   parameters are needed here.
3. A **decoder** (the only part of the network that gets trained) inverts the
   AdaIN output back into a normal RGB image.
4. An **alpha** parameter (0 to 1) controls style strength by interpolating between
   the original content features and the fully stylized features before decoding.

Because steps 1–3 are a single forward pass, this runs much faster than classic
Gatys-style optimization-based style transfer, which re-optimizes pixels for every
new image pair.

## Project structure

```
ai-nst-project/
├── NST_Code/
│   ├── app.py                  # Flask web app (upload content + style, get stylized image)
│   ├── train.py                # Training script for the decoder
│   ├── utils/
│   │   ├── models.py           # VGGEncoder and Decoder model definitions
│   │   └── utils.py            # AdaIN math, dataset loader, transforms
│   ├── templates/
│   │   └── index.html          # Web UI
│   ├── static/uploads/         # Images uploaded/generated at runtime (a few demo samples ship here)
│   ├── content_data/           # Sample content images for training/testing
│   ├── style_data/             # Sample style images for training/testing
│   ├── examples/               # Images shown in the "Examples" section of the web UI
│   ├── experiment/final_exp/   # Trained decoder checkpoint + training logs/samples
│   ├── vgg_normalised.pth      # Pretrained, frozen VGG-19 encoder weights
│   └── adain_algo.png          # Architecture diagram (from the AdaIN paper)
├── Demo_IO_Images/             # Input/output images shown in this README
├── code.ipynb                  # Notebook visualizing VGG feature activations across layers
├── requirements.txt
├── Procfile                    # For Heroku-style deployment (gunicorn)
└── README.md
```

## Setup

```bash
git clone <your-repo-url>
cd ai-nst-project
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

The pretrained VGG-19 encoder (`vgg_normalised.pth`) and the trained decoder
(`experiment/final_exp/decoder_final.pth`) are already included in this repo, so no
extra downloads are needed to run the app.

## Running the web app

```bash
cd NST_Code
python app.py
```

Then open `http://localhost:5000` in your browser. Upload a content image and a style
image, adjust the alpha slider to control style strength, and submit.

For production deployment (e.g. Heroku), the included `Procfile` runs the app with
`gunicorn` instead of Flask's dev server.

## Training your own decoder

The encoder is frozen — only the decoder is trained, using a combination of content
loss and style loss computed from VGG feature statistics.

```bash
cd NST_Code
python train.py \
    --content_dir content_data \
    --style_dir style_data \
    --vgg vgg_normalised.pth \
    --experiment my_experiment \
    --epochs 200 \
    --batch_size 16
```

Useful flags:

| Flag | Default | Description |
|---|---|---|
| `--content_dir` | `content_data` | Folder of content training images |
| `--style_dir` | `style_data` | Folder of style training images |
| `--vgg` | `vgg_normalised.pth` | Path to pretrained VGG weights |
| `--final_size` | `256` | Output image size during training |
| `--lr` | `1e-4` | Learning rate |
| `--style_weight` | `5` | Weight on the style loss term |
| `--content_weight` | `1.0` | Weight on the content loss term |
| `--epochs` | `1` | Number of training epochs |
| `--save_interval` | `2` | Save a checkpoint every N epochs |
| `--resume` | `False` | Resume from `--decoder_path` / `--optimizer_path` |

Checkpoints, sample outputs, and the training args used are saved under
`NST_Code/experiment/<experiment_name>/`. The shipped `experiment/final_exp/` folder
shows exactly this output from the run that produced the included `decoder_final.pth`.

For a larger-scale training run you'll want a bigger content/style dataset than the
small sample folders included here (e.g. a subset of COCO for content images, and a
painting dataset such as WikiArt for style images).

## Exploring VGG feature activations

`code.ipynb` is a small notebook that loads a few example images, runs them through the
VGG encoder, and visualizes the activation maps at each of the four encoder stages
(`relu1_1` through `relu4_1`). It's a good way to see what the encoder is actually
"looking at" before AdaIN ever touches the features. Run it from the repository root
so its relative imports/paths resolve correctly.

## Tech stack

Python · PyTorch · torchvision · Flask · Flask-Bootstrap · WTForms

## Notes

- `vgg_normalised.pth` (~77 MB) and `decoder_final.pth` (~14 MB) are committed directly
  to this repo as model weights. If you fork this project and want to keep the repo
  lighter long-term, consider moving them to [Git LFS](https://git-lfs.github.com/) or
  hosting them externally (e.g. a GitHub Release) and downloading them in a setup step.
- GPU is used automatically if available (`torch.cuda.is_available()`); otherwise
  everything falls back to CPU.
