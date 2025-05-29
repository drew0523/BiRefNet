<h1 align="center">Bilateral Reference for High-Resolution Dichotomous Image Segmentation</h1>

<div align='center'>
    <a href='https://scholar.google.com/citations?user=TZRzWOsAAAAJ' target='_blank'><strong>Peng Zheng</strong></a><sup> 1,4,5,6</sup>,&thinsp;
    <a href='https://scholar.google.com/citations?user=0uPb8MMAAAAJ' target='_blank'><strong>Dehong Gao</strong></a><sup> 2</sup>,&thinsp;
    <a href='https://scholar.google.com/citations?user=kakwJ5QAAAAJ' target='_blank'><strong>Deng-Ping Fan</strong></a><sup> 1*</sup>,&thinsp;
    <a href='https://scholar.google.com/citations?user=9cMQrVsAAAAJ' target='_blank'><strong>Li Liu</strong></a><sup> 3</sup>,&thinsp;
    <a href='https://scholar.google.com/citations?user=qQP6WXIAAAAJ' target='_blank'><strong>Jorma Laaksonen</strong></a><sup> 4</sup>,&thinsp;
    <a href='https://scholar.google.com/citations?user=pw_0Z_UAAAAJ' target='_blank'><strong>Wanli Ouyang</strong></a><sup> 5</sup>,&thinsp;
    <a href='https://scholar.google.com/citations?user=stFCYOAAAAAJ' target='_blank'><strong>Nicu Sebe</strong></a><sup> 6</sup>
</div>

## Usage

#### Environment Setup

```shell
# PyTorch==2.5.1+CUDA12.4 (or 2.0.1+CUDA11.8) is used for faster training (~40%) with compilation.
conda create -n birefnet python=3.10 -y && conda activate birefnet
pip install -r requirements.txt
```

## Run

```shell
# Train & Test & Evaluation
./train_test.sh RUN_NAME GPU_NUMBERS_FOR_TRAINING GPU_NUMBERS_FOR_TEST
# Example: ./train_test.sh tmp-proj 0,1,2,3,4,5,6,7 0

# See train.sh / test.sh for only training / test-evaluation.
# After the evaluation, run `gen_best_ep.py` to select the best ckpt from a specific metric (you choose it from Sm, wFm, HCE (DIS only)).
```

### :pen: Fine-tuning on Custom Data

> A video of the tutorial on BiRefNet fine-tuning has been released on my YouTube channel ⬇️

[![BiRefNet Fine-tuning Tutorial](https://img.youtube.com/vi/FwGT_0V9E-k/0.jpg)](https://youtu.be/FwGT_0V9E-k)

> Suppose you have some custom data, fine-tuning on it tends to bring improvement.

1. **Pre-requisites**: you have put your datasets in the path `${data_root_dir}/TASK_NAME/DATASET_NAME`. For example, `${data_root_dir}/DIS5K/DIS-TR` and `${data_root_dir}/General/TR-HRSOD`, where `im` and `gt` are both in each dataset folder.
2. **Change an existing task to your custom one**: replace all `'General'` (with single quotes) in the whole project with `your custom task name` as the screenshot of vscode given below shows:<img src="https://drive.google.com/thumbnail?id=1J6gzTmrVnQsmtt3hi6ch3ZrH7Op9PKSB&sz=w400" />
3. **Adapt settings**:
   + `sys_home_dir`: path to the root folder, which contains codes / datasets / weights / ... -- project folder / data folder / backbone weights folder are `${sys_home_dir}/codes/dis/BiRefNet / ${sys_home_dir}/datasets/dis/General / ${sys_home_dir}/weights/cv/swin_xxx`, respectively.
   + `testsets`: your validation set.
   + `training_set`: your training set.
   + `lambdas_pix_last`: adapt the weights of different losses if you want, especially for the difference between segmentation (classification task) and matting (regression task).
4. **Use existing weights**: if you want to use some existing weights to fine-tune that model, please refer to the `resume` argument in `train.py`. Attention: the epoch of training continues from the epochs the weights file name indicates (e.g., `244` in `BiRefNet-general-epoch_244.pth`), instead of `1`. So, if you want to fine-tune `50` more epochs, please specify the epochs as `294`. `\#Epochs, \#last epochs for validation, and validation step` are set in `train.sh`.
5. Good luck to your training :) If you still have questions, feel free to leave issues (recommended way) or contact me.



### Citation

```
@article{zheng2024birefnet,
  title={Bilateral Reference for High-Resolution Dichotomous Image Segmentation},
  author={Zheng, Peng and Gao, Dehong and Fan, Deng-Ping and Liu, Li and Laaksonen, Jorma and Ouyang, Wanli and Sebe, Nicu},
  journal={CAAI Artificial Intelligence Research},
  volume = {3},
  pages = {9150038},
  year={2024}
}
```
