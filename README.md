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


### :pen: Fine-tuning on Custom Data

> fine-tuning 튜토리얼 영상: 사진 클릭 ⬇️

[![BiRefNet Fine-tuning Tutorial](https://img.youtube.com/vi/FwGT_0V9E-k/0.jpg)](https://youtu.be/FwGT_0V9E-k)

> Suppose you have some custom data, fine-tuning on it tends to bring improvement.

1. **Pre-requisites**: 커스텀 데이터셋을 다음의 경로에 저장: `${data_root_dir}/TASK_NAME/DATASET_NAME` 예시: `${data_root_dir}/custom/test_custom` 또는 `${data_root_dir}/General/train_custom` 그리고 각 dataset 폴더에는 `im` 와 `gt` 에 해당하는 데이터셋 폴더를 가져야 함. im: 이미지 / gt: 이미지에 대응되는 matting ground truth.

> > #### 예시
> > ![image](https://github.com/user-attachments/assets/7ba935b2-3e0c-42d4-b084-9acaa31b518e)

> > 같은 폴더 내 im / gt 폴더에 대해서 파일명을 동일하게 하여 dataset을 넣어야 합니다.
> > ![image](https://github.com/user-attachments/assets/2ab9420a-46d3-4a1c-8499-90bf23f44aeb)
> > ![image](https://github.com/user-attachments/assets/bb9e10aa-a0af-4cee-ae99-98b520a29ef5)




2. **Change an existing task to your custom one**: replace all `'General'` (with single quotes) in the whole project with `your custom task name` as the screenshot of vscode given below shows:<img src="https://drive.google.com/thumbnail?id=1J6gzTmrVnQsmtt3hi6ch3ZrH7Op9PKSB&sz=w400" />

> >위의 사진과 같이 공식 repository를 clone한 경우 `'General'`을  `your custom task name`으로 변경해주면 됩니다.

> >현재 수정한 config.py를 쓸 경우 이 부분은 skip해도 됩니다. 대신 반드시, 1번에서의 예시 사진과 똑같이 설정한 폴더 구조와 폴더 명을 따라야합니다. task 명 혹은 dataset 폴더명을 바꾸게 될 경우 아래 config.py에 대응되는 부분을 수정하면 됩니다. 
> > ![image](https://github.com/user-attachments/assets/5f5b4801-3b7c-4601-a648-b7a296d5d92b)



3. **Adapt settings**:
   + `sys_home_dir': root 폴더에 대응되는 path
   + `data_root_dir': dataset이 존재하는 root 디렉토리
   ![image](https://github.com/user-attachments/assets/ab7ddcbf-9e9f-40ec-b000-92fdbc8de88f)

   + 또한, 현재 task는 `matting` task로 해당 task에 맞게 loss function을 config.py에서 설정한 상황입니다. task를 `segmentation`으로 변경할 경우 이 부분 수정하면 됩니다. 이외의 learning rate 등도 추가적으로 수정이 필요할 경우 config.py에서 수정 진행하면 됩니다.

   + `lambdas_pix_last`: adapt the weights of different losses if you want, especially for the difference between segmentation (classification task) and matting (regression task).
     
      ![image](https://github.com/user-attachments/assets/3dff2140-411e-47b3-ae1b-d69825fa1672)

   + `testsets`: your test set.
   + `training_set`: your training set.
   + `validation_set`: your validation set.
     
     ![image](https://github.com/user-attachments/assets/4cd0f4e9-6594-4a4b-9be9-7f944958d56b)



4. **Use existing weights**: pretrained weight를 가지고 fine-tuning 을 진행하기 위해 train.sh 파일을 수정합니다.
  아래 사진과 같이 weight에 대한 path를 지정합니다. 이때 현재 weight의 최종 학습 epoch이 100이고 추가적으로 10 epoch을 학습시키고 싶을 경우 `100 + 10` 과 같이 입력해줍니다.
   `\#Epochs, \#last epochs for validation, and validation step` are set in `train.sh`.
   ![image](https://github.com/user-attachments/assets/57d321e8-2d8f-4e08-8efc-8b10d34e27df)

 
## Run
> 학습을 위한 setting은 마무리되어 train 및 test까지 실행합니다. 아래 쉘스크립트를 실행하면 됩니다.
```shell
# Train & Test & Evaluation
./train_test.sh RUN_NAME GPU_NUMBERS_FOR_TRAINING GPU_NUMBERS_FOR_TEST
```
### Example: ![image](https://github.com/user-attachments/assets/b18baeb0-2311-4ea1-9e9b-564bf45fe369)
> 이 경우 gpu 4, 5를 가지고 학습을 진행 후 4로 test를 진행합니다. 이후 ckpt/z_no-freeze/ 폴더에 학습 완료된 model이 저장됩니다.

> ![image](https://github.com/user-attachments/assets/fa59b3e5-762d-46e3-8258-05637008f67c)


### :pen: torch2onnx
#### notebook version
> 해당 ipynb를 통해 변환을 수행하기 위해서는 최소 vram 20GB가 요구됩니다. (Colab 무료 불가)

> 또한, 해당 과정을 수정한 config.py 를 사용할 경우 알 수 없는 에러가 발생하여 `새로운 디렉토리에 다시 git clone`을 후 진행하였습니다. 아래의 ipynb를 새로운 디렉토리에서 실행하면 이상 없이 작동합니다.

### 결과
![image](https://github.com/user-attachments/assets/3d148796-d18b-47bd-bf95-42490408d457)

[biref_torch2onnx.ipynb](https://drive.google.com/file/d/1katt9le45K35n1GL8ZQocqRJiJlqNWFA/view?usp=sharing)

#### python file version
> 1. 앞선 사항과 마찬가지로 깔끔한 변환 환경을 위해 새롭게 clone을 진행하고 conda, requirement 등 environment를 setting합니다.
> 2. 추가적으로 export 하기 위해 필요한 사항들을 설치합니다.
> ```shell
> pip install -q gdown onnx onnxscript onnxruntime-gpu==1.18.1
> ```
> 3. 이후 BiRefNet 폴더에 [export_onnx.py](https://drive.google.com/file/d/1FB_kSi9u_4pXGtcQNUAtEeWzFkpGrTU4/view?usp=sharing) 파일을 업로드합니다.
> ![image](https://github.com/user-attachments/assets/82841425-a7b9-4c25-a4c9-d788b0ebc8b0)
> 4. 이후 현재 위치에서 다음을 실행합니다.
> ```shell
> python3 export_onnx.py --pt_path "{pth 파일 경로}" --onnx_save_path "{onnx 저장경로}" --device {사용할 gpu/cpu}
> ```
> ![image](https://github.com/user-attachments/assets/faa20cf3-d46c-4cdc-b9e0-aa913552d77e)

> 변환 완료
> ![image](https://github.com/user-attachments/assets/aa8ac210-dc93-4f16-bf48-9557231d753c)



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
