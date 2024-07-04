
<!-- ![ID-CompressAI-logo](assets/CRA5LOGO.svg =750x140) -->
<a href="url"><img src="assets/CRA5LOGO.svg" align="center"></a>

[![License](https://img.shields.io/github/license/InterDigitalInc/CompressAI?color=blue)](https://github.com/InterDigitalInc/CompressAI/blob/master/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/cra5?color=brightgreen)](https://pypi.org/project/compressai/)
[![Downloads](https://pepy.tech/badge/cra5)](https://pypi.org/project/cra5/#files)

[Paper:CRA5: Extreme Compression of ERA5 for Portable Global Climate and Weather Research via an Efficient Variational Transformer](https://arxiv.org/abs/2405.03376)

# Introduction and get started

CRA5 is a extreme **compressed dataset** of the most popular weather dataset ERA5. The repository also includes **compression models**, **forecasting model** for researchers to conduct portable weather and climate research.

CRA5 currently provides:

* A customized variaitional transformer (VAEformer) for climate data compression
* A dataset CRA5 less than 1 TiB, but contains the same information with 300 TiB ERA5 dataset. Covering ERA5 from year 1979 to 2023 at an houly interval.    
* A pre-trained Auto-Encoder on the climate/weather data to support some potential weather research.



> **Note**: Multi-GPU support is now experimental.

## Installation

CRA5 supports python 3.8+ and PyTorch 1.7+.

**pip**:

```bash
pip install cra5
```

> **Note**: wheels are available for Linux and MacOS.

**From source**:

A C++17 compiler, a recent version of pip (19.0+), and common python packages are also required (see `setup.py` for the full list).

To get started locally and install the development version of CRA5, run the following commands in a [virtual environment](https://docs.python.org/3.6/library/venv.html):

```bash
git https://github.com/taohan10200/CRA5
cd CRA5
pip install -U pip && pip install -e .
```

For a custom installation, you can also run one of the following commands:
* `pip install -e '.[dev]'`: install the packages required for development (testing, linting, docs)
* `pip install -e '.[tutorials]'`: install the packages required for the tutorials (notebooks)
* `pip install -e '.[all]'`: install all the optional packages

> **Note**: Docker images will be released in the future. Conda environments are not
officially supported.

<!-- ## Documentation -->

<!-- * [Installation](https://interdigitalinc.github.io/CompressAI/installation.html)
* [CompressAI API](https://interdigitalinc.github.io/CompressAI/)
* [Training your own model](https://interdigitalinc.github.io/CompressAI/tutorials/tutorial_train.html)
* [List of available models (model zoo)](https://interdigitalinc.github.io/CompressAI/zoo.html) -->

# Usages

## 1. CRA5 dataset is an outcome of the VAEformer in the atmospheric science. We explore this to facilitate the research in weather and climate. 

* **Train the large data-driven numerical weather forecasting models with our CRA5**

> **Note**: For researches who do not have enough disk space to store the 300 TiB+ ERA5 dataset, but have interests to to train a large weather forecasting model, like [FengWu-GHR](https://arxiv.org/abs/2402.00059),  this research can help you save it into less than 1 TiB disk.  

Our preliminary attemp has proven that the CRA5 dataset can train the very very similar NWP model compared with the original ERA5 dataset. Also, with this dataset, you can easily train a Nature published forecasting model, like [Pangu-Weather](https://www.nature.com/articles/s41586-023-06185-3). 

<!-- ![ID-CompressAI-logo](assets/rmse_acc_bias_activity.png =400x140) -->
<a href="url"><img src="assets/rmse_acc_bias_activity.png" align="center"></a>

## 2. VAEformer is a powerful compression model, we hope it can be extended to other domains, like image and video compression.

<!-- ![ID-CompressAI-logo](assets/MSE_supp_new.png =400x140) -->
<a href="url"><img src="assets/MSE_supp_new.png" align="center"></a>


*  **We here demonstrate how to use it for weather data compression and decompression**


```python
import os 
import torch
from cra5.models.compressai.zoo import vaeformer_pretrained
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
net = vaeformer_pretrained(quality=268, pretrained=True).eval().to(device)
input_data_norm = torch.rand(1,268, 721,1440) #This is a proxy weather data. It actually should be a 
x = torch.from_numpy(input_data_norm).unsqueeze(0).to(device)
print(x.shape)
with torch.no_grad():
    out_net = net.compress(x) 
    
print(out_net)
```
or directly using our API


```python
from cra5.api import cra5_api
cra5_API = cra5_api()
# This command will download two ERA5 netcdf files 
# data/ERA5/2024/2024-06-01T00:00:00_pressure.nc (513MiB) and data/ERA5/2024/2024-06-01T00:00:00_single.nc (18MiB) 
# and then compress it into a tiny binary file `./data/cra5/2024/2024-06-01T00:00:00.bin` (**1.8Mib**)
cra5_API.encoder_era5(time_stamp="2024-06-01T00:00:00") 

# If you aready have the compressed binary file,  this command will help you get the reconstructed weather data.
cra5_data = cra5_API.decode_from_bin(time_stamp="2024-06-01T00:00:00")

# show some variables for the constructed data
cra5_API.show_image(
	reconstruct_data=cra5_data.cpu().numpy(), 
	time_stamp="2024-06-01T00:00:00", 
	show_variables=['z_500', 'q_500', 'u_500', 'v_500', 't_500', 'w_500'])

```
<!-- ![ID-CompressAI-logo](assets/CRA5LOGO.svg =400x140) -->
<a href="url"><img src="assets/2024-06-01T00:00:00.png" align="center"></a>



## 3 VAEformer is based on the Auto-Encoder-Decoder, we provide a pretrained VAE for the weather research, you can use our VAEformer to get the latents for downstream research, like diffusion-based or other generation-based forecasting methods.

* **Using it as a Auto-Encoder-Decoder**

> **Note**: For people who are intersted in diffusion-based or other generation-based forecasting methods, we can provide an Auto Encoder and decoder for the weather research, you can use our VAEformer to get the latents for downstream research.


```python
from cra5.api import cra5_api
cra5_API = cra5_api()
# This command will download two ERA5 netcdf files 
# data/ERA5/2024/2024-06-01T00:00:00_pressure.nc (513MiB) and data/ERA5/2024/2024-06-01T00:00:00_single.nc (18MiB) 
# and then compress it into a tiny binary file `./data/cra5/2024/2024-06-01T00:00:00.bin` (**1.8Mib**)
cra5_API.encoder_era5(time_stamp="2024-06-01T00:00:00") 

# If you aready have the compressed binary file,  this command will help you get the latent of the weather data.
latent = cra5_API.decode_from_bin("2024-06-01T00:00:00", return_format='latent')

# show some variables for the constructed data
cra5_API.show_latent(
	latent=latent.squeeze(0).cpu().numpy(), 
	time_stamp="2024-06-01T00:00:00", 
	show_channels=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150])

```
<!-- ![ID-CompressAI-logo](assets/2024-06-01T00:00:00_latent.png =400x140) -->
<a href="url"><img src="assets/2024-06-01T00:00:00_latent.png" align="center"></a>


Script and notebook examples can be found in the `examples/` directory.

To encode/decode images with the provided pre-trained models, run the
`codec.py` example:

```bash
python3 examples/codec.py --help
```

An examplary training script with a rate-distortion loss is provided in
`examples/train.py`. You can replace the model used in the training script
with your own model implemented within CompressAI, and then run the script for a
simple training pipeline:

```bash
python3 examples/train.py -d /path/to/my/image/dataset/ --epochs 300 -lr 1e-4 --batch-size 16 --cuda --save
```
> **Note:** the training example uses a custom [ImageFolder](https://interdigitalinc.github.io/CompressAI/datasets.html#imagefolder) structure.

A jupyter notebook illustrating the usage of a pre-trained model for learned image
compression is also provided in the `examples` directory:

```bash
pip install -U ipython jupyter ipywidgets matplotlib
jupyter notebook examples/
```

### Evaluation

To evaluate a trained model on your own dataset, CompressAI provides an
evaluation script:

```bash
python3 -m compressai.utils.eval_model checkpoint /path/to/images/folder/ -a $ARCH -p $MODEL_CHECKPOINT...
```

To evaluate provided pre-trained models:

```bash
python3 -m compressai.utils.eval_model pretrained /path/to/images/folder/ -a $ARCH -q $QUALITY_LEVELS...
```

To plot results from bench/eval_model simulations (requires matplotlib by default):

```bash
python3 -m compressai.utils.plot --help
```

<!-- To evaluate traditional codecs:

```bash
python3 -m compressai.utils.bench --help
python3 -m compressai.utils.bench bpg --help
python3 -m compressai.utils.bench vtm --help
```

For video, similar tests can be run, CompressAI only includes ssf2020 for now:

```bash
python3 -m compressai.utils.video.eval_model checkpoint /path/to/video/folder/ -a ssf2020 -p $MODEL_CHECKPOINT...
python3 -m compressai.utils.video.eval_model pretrained /path/to/video/folder/ -a ssf2020 -q $QUALITY_LEVELS...
python3 -m compressai.utils.video.bench x265 --help
python3 -m compressai.utils.video.bench VTM --help
python3 -m compressai.utils.video.plot --help
``` -->

<!-- ## Tests

Run tests with `pytest`:

```bash
pytest -sx --cov=compressai --cov-append --cov-report term-missing tests
```

Slow tests can be skipped with the `-m "not slow"` option. -->


## License

CompressAI is licensed under the BSD 3-Clause Clear License

## Contributing

We welcome feedback and contributions. Please open a GitHub issue to report
bugs, request enhancements or if you have any questions.

Before contributing, please read the CONTRIBUTING.md file.

## Authors

* Tao Han ([hantao10200@gmail.com](mailto:hantao10200@gmail.com)) 
* Zhenghao Chen.

## Citation

If you use this project, please cite the relevant original publications for the models and datasets, and cite this project as:

```
@article{han2024cra5extremecompressionera5,
      title={CRA5: Extreme Compression of ERA5 for Portable Global Climate and Weather Research via an Efficient Variational Transformer}, 
      author={Tao Han and Zhenghao Chen and Song Guo and Wanghan Xu and Lei Bai},
      year={2024},
      eprint={2405.03376},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2405.03376}, 
}
```

For any work related to the forecasting models, please cite
```
@article{han2024fengwughr,
title={FengWu-GHR: Learning the Kilometer-scale Medium-range Global Weather Forecasting}, 
author={Tao Han and Song Guo and Fenghua Ling and Kang Chen and Junchao Gong and Jingjia Luo and Junxia Gu and Kan Dai and Wanli Ouyang and Lei Bai},
year={2024},
eprint={2402.00059},
archivePrefix={arXiv},
primaryClass={cs.LG}
}
```

## Related links
 * Tensorflow compression library by _Ballé et al._: https://github.com/tensorflow/compression
 * Range Asymmetric Numeral System code from _Fabian 'ryg' Giesen_: https://github.com/rygorous/ryg_rans
 * BPG image format by _Fabrice Bellard_: https://bellard.org/bpg
 * HEVC HM reference software: https://hevc.hhi.fraunhofer.de
 * VVC VTM reference software: https://vcgit.hhi.fraunhofer.de/jvet/VVCSoftware_VTM
 * AOM AV1 reference software: https://aomedia.googlesource.com/aom
 * Z. Cheng et al. 2020: https://github.com/ZhengxueCheng/Learned-Image-Compression-with-GMM-and-Attention
 * Kodak image dataset: http://r0k.us/graphics/kodak/
