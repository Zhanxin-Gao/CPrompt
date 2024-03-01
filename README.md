
# CPrompt

This repo contains the official code of the project "Consistent Prompting for Rehearsal-Free Continual Learning" (CVPR2024).
Paper link.

![The proposed consistent prompting](https://github.com/Zhanxin-Gao/CPrompt/blob/main/CPrompt_Illustration.pdf)

## 1.Dependent Packages and Platform

First we recommend to create a conda environment with all the required packages by using the following command.

```
conda env create -f environment.yml
```

This command creates a conda environment named `CPrompt`. You can activate the conda environment with the following command:

```
conda activate CPrompt
```

In the following sections, we assume that you use this conda environment or you manually install the required packages.

Note that you may need to adapt the `environment.yml/requirements.txt` files to your infrastructure. The configuration of these files was tested on Linux Platform with a GPU (RTX3080 Ti).

If you see the following error, you may need to install a PyTorch package compatible with your infrastructure.

```
RuntimeError: No HIP GPUs are available or ImportError: libtinfo.so.5: cannot open shared object file: No such file or directory
```

For example if your infrastructure only supports CUDA == 11.1, you may need to install the PyTorch package using CUDA11.1.

```
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
```

## 2.Dataset

We have implemented the pre-processing of `Split StanfordCars `, `Split ImageNet-R`, `Split CIFAR-100` and `Split DomainNet`. When training on `Split CIFAR-100`, this framework will automatically download it. When training on other datasets, you should specify the folder of your dataset in `utils/data.py`.

```python
    def download_data(self):
        rootdir = "[DATA-PATH]"
```

- StanfordCars: retrieve from: [https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset?resource=download](https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset?resource=download)
- ImageNet-R: retrieve from: [https://github.com/hendrycks/imagenet-r](https://github.com/hendrycks/imagenet-r)
- DomainNet: retrieve from: [http://ai.bu.edu/M3SDA/](http://ai.bu.edu/M3SDA/)
## 3.Run
<!-- `To run the testing code:`

Due to the limitation of the supplementary material size, we only upload the results of CIFAR100.

- Test CIFAR100

    ```
    python main.py --config=mafdrc-cifar100.json --test True
    ``` -->

<!-- `To run the training code:` -->
- Train Split StanfordCars

    ```
    python main.py --config=./exps/stanfordcars.json
    ```

- Train Split ImageNet-R

    ```
    python main.py --config=./exps/imagenetr.json
    ```

- Train Split CIFAR-100

    ```
    python main.py --config=./exps/cifar100_vit.json
    ```

- Train Split DomainNet

    ```
    python main.py --config=./exps/domainnet.json
    ```


<!-- ## 3.Results

`CIFAR100:`

Tasks | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80 | 90 | 100 | Avg |
:---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
Ours (MAFDRC) | **93.4** | **83.0** | **80.9** | **76.72** | **74.18** | **71.3** | **69.17** | **65.45** | **63.49** | **62.04** | **73.97**

Note:

If you use other versions of pytorch, it will get a different result. -->

## 4.Citation

If you find this code useful, please kindly cite the following paper:

```
@article{,
  title={Consistent Prompting for Rehearsal-Free Continual Learning},
  author={Zhanxin Gao, Jun CEN, Xiaobin Chang},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition(CVPR)},
  year={2024}
}
```
