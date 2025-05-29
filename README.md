<h1 align="center">Bilateral Reference for High-Resolution Dichotomous Image Segmentation</h1>

## Usage for Fine-tuning ~ torch2onnx

#### Clone Repository

```shell
git clone https://github.com/ZhengPeng7/BiRefNet.git
cd BiRefNet
```

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

> fine-tuning 튜토리얼 영상: 사진 클릭 ⬇️

[![BiRefNet Fine-tuning Tutorial](https://img.youtube.com/vi/FwGT_0V9E-k/0.jpg)](https://youtu.be/FwGT_0V9E-k)

> Suppose you have some custom data, fine-tuning on it tends to bring improvement.

1. **Pre-requisites**: 커스텀 데이터셋을 다음의 경로에 저장: `${data_root_dir}/TASK_NAME/DATASET_NAME` 예시: `${data_root_dir}/custom/test_custom` 또는 `${data_root_dir}/General/train_custom` 그리고 각 dataset 폴더에는 `im` 와 `gt` 에 해당하는 데이터셋 폴더를 가져야 함. im: 이미지 / gt: 이미지에 대응되는 matting ground truth.

> > #### 예시
> > ![image](https://github.com/user-attachments/assets/f22b4c14-5a33-4c97-aeae-b40542b83e24)

> > 같은 폴더 내 im / gt 폴더에 대해서 파일명을 동일하게 하여 dataset을 넣어야 합니다.
> > ![image](https://github.com/user-attachments/assets/2ab9420a-46d3-4a1c-8499-90bf23f44aeb)
> > ![image](https://github.com/user-attachments/assets/bb9e10aa-a0af-4cee-ae99-98b520a29ef5)




2. **Change an existing task to your custom one**: replace all `'General'` (with single quotes) in the whole project with `your custom task name` as the screenshot of vscode given below shows:<img src="https://drive.google.com/thumbnail?id=1J6gzTmrVnQsmtt3hi6ch3ZrH7Op9PKSB&sz=w400" />

위의 사진과 같이 공식 repository를 clone한 경우 `'General'`을  `your custom task name`으로 변경해주면 됩니다.

> 현재 수정한 config.py를 쓸 경우 이 부분은 skip해도 됩니다. 대신 반드시, 1번에서의 예시 사진과 똑같이 설정한 폴더 구조와 폴더 명을 따라야합니다. task 명 혹은 dataset 폴더명을 바꾸게 될 경우 아래 config.py에 대응되는 부분을 수정하면 됩니다. 
> ![image](https://github.com/user-attachments/assets/5f5b4801-3b7c-4601-a648-b7a296d5d92b)



3. **Adapt settings**:
   + `sys_home_dir': root 폴더에 대응되는 path
   + `data_root_dir': dataset이 존재하는 root 디렉토리
   ![image](https://github.com/user-attachments/assets/ab7ddcbf-9e9f-40ec-b000-92fdbc8de88f)

> 또한, 현재 task는 `matting` task로 해당 task에 맞게 loss function을 config.py에서 설정한 상황입니다. task를 `segmentation`으로 변경할 경우 이 부분 수정하면 됩니다. 이외의 learning rate 등도 추가적으로 수정이 필요할 경우 config.py에서 수정 진행하면 됩니다.
    + `lambdas_pix_last`: adapt the weights of different losses if you want, especially for the difference between segmentation (classification task) and matting (regression task).
> ![image](https://github.com/user-attachments/assets/3dff2140-411e-47b3-ae1b-d69825fa1672)

   + `testsets`: your test set.
   + `training_set`: your training set.
   + `validation_set`: your validation set.
     ![image](https://github.com/user-attachments/assets/4cd0f4e9-6594-4a4b-9be9-7f944958d56b)



5. **Use existing weights**: if you want to use some existing weights to fine-tune that model, please refer to the `resume` argument in `train.py`. Attention: the epoch of training continues from the epochs the weights file name indicates (e.g., `244` in `BiRefNet-general-epoch_244.pth`), instead of `1`. So, if you want to fine-tune `50` more epochs, please specify the epochs as `294`. `\#Epochs, \#last epochs for validation, and validation step` are set in `train.sh`.
6. Good luck to your training :) If you still have questions, feel free to leave issues (recommended way) or contact me.



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
