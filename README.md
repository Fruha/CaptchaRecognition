## Description
- Dataset generation
- Model training (**EfficientNetB0+TransformerEncoder->CTCLoss**)

## Example predictions

<img src="https://github.com/Fruha/CaptchaRecognition/blob/master/git_images/example_predictions.png" width="60%">

## Installation

```bash
git clone https://github.com/Fruha/CaptchaRecognition
cd CaptchaRecognition
pip install -r requirements.txt
```
## Usage

### Generate data.
```bash
python captcha_gen.py
```
You can use any tff formats, using **path_to_ttfs** in captcha_gen.py

### Training. 
Change **hparams** in captcha_gen.py
```bash
python captcha_gen.py
```

### Visualization

```bash
tensorboard --logdir tb_logs
```


## Plot (Accuracy of words)/(count dataset)

<img src="https://github.com/Fruha/CaptchaRecognition/blob/master/git_images/plot_accuracy.png" width="60%">