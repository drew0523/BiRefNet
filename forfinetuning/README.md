# BiRefNet 사용자 정의 데이터 기반 Fine-tuningg 및 Onnx 변환 가이드

## 1. Clone Repository

```shell
git clone https://github.com/ZhengPeng7/BiRefNet.git
cd BiRefNet
```

## 2. Environment Setup

```shell
# PyTorch==2.5.1+CUDA12.4 (or 2.0.1+CUDA11.8) is used for faster training (~40%) with compilation.
conda create -n birefnet python=3.10 -y && conda activate birefnet
pip install -r requirements.txt
```


## 3. Fine-tuning on Custom Data

- Fine-tuning 튜토리얼 영상
[![BiRefNet Fine-tuning Tutorial](https://img.youtube.com/vi/FwGT_0V9E-k/0.jpg)](https://youtu.be/FwGT_0V9E-k)

### 3.1 데이터셋 준비

- 사용자 정의 데이터는 다음과 같은 구조로 정리되어야 합니다:

  ```perl
  ${data_root_dir}/TASK_NAME/DATASET_NAME/
  ├── im/   # 입력 이미지
  └── gt/   # 대응되는 GT (matting ground truth)
  ```

- 예시:
  ```perl
  ./data_root/General/train_custom/
  ├── im/
  └── gt/
  ```

> > 이미지 예시

> > ![image](https://github.com/user-attachments/assets/f22b4c14-5a33-4c97-aeae-b40542b83e24)

> > `im/`과 `gt/` 폴더 내부의 파일명은 반드시 동일해야 합니다.

> > ![image](https://github.com/user-attachments/assets/2ab9420a-46d3-4a1c-8499-90bf23f44aeb)

> > ![image](https://github.com/user-attachments/assets/bb9e10aa-a0af-4cee-ae99-98b520a29ef5)




### 3-2. 프로젝트 태스크 변경

- 기존 레포지토리는 'General' 태스크를 기준으로 설정되어 있습니다. 사용자 정의 태스크로 변경하려면 프로젝트 전체에서 'General' 문자열을 원하는 태스크명으로 일괄 치환해주시기 바랍니다.

- VSCode 활용 예시

  <img src="https://drive.google.com/thumbnail?id=1J6gzTmrVnQsmtt3hi6ch3ZrH7Op9PKSB&sz=w400" />

- 단, 수정된 config.py를 사용 중이라면 이 단계는 생략 가능하며, 대신 데이터셋 구조 및 폴더 명칭이 정확히 일치해야 합니다.

  ![image](https://github.com/user-attachments/assets/5f5b4801-3b7c-4601-a648-b7a296d5d92b)



### 3-3. 설정 파일 수정 (`config.py`)
  - `sys_home_dir`: root 폴더에 대응되는 path
  - `data_root_dir`: dataset이 존재하는 root 디렉토리
    ![image](https://github.com/user-attachments/assets/ab7ddcbf-9e9f-40ec-b000-92fdbc8de88f)

  -  또한, 현재 task는 `matting` task로 해당 task에 맞게 loss function을 config.py에서 설정한 상황입니다. task를 `segmentation`으로 변경할 경우 이 부분 수정하면 됩니다. 이외의 learning rate 등도 추가적으로 수정이 필요할 경우 config.py에서 수정 진행하면 됩니다.

  - `lambdas_pix_last`: adapt the weights of different losses if you want, especially for the difference between segmentation (classification task) and matting (regression task).
     
      ![image](https://github.com/user-attachments/assets/3dff2140-411e-47b3-ae1b-d69825fa1672)

   + `testsets`: your test set.
   + `training_set`: your training set.
   + `validation_set`: your validation set.
     
     ![image](https://github.com/user-attachments/assets/4cd0f4e9-6594-4a4b-9be9-7f944958d56b)



### 3-4. 사전 학습된 모델 응용
- `train.sh` 파일 내에 pretrained weight 경로를 지정하고, 이어서 학습할 epoch 수를 설정합니다.
- 아래 사진과 같이 weight에 대한 path를 지정합니다. 이때 현재 weight의 최종 학습 epoch이 100이고 추가적으로 10 epoch을 학습시키고 싶을 경우 `100 + 10` 과 같이 입력해줍니다.

   ![image](https://github.com/user-attachments/assets/57d321e8-2d8f-4e08-8efc-8b10d34e27df)

 
## 4. 학습 및 테스트 실행

```shell
./train_test.sh RUN_NAME GPU_NUMBERS_FOR_TRAINING GPU_NUMBERS_FOR_TEST
```
- 예시
  
  ![image](https://github.com/user-attachments/assets/b18baeb0-2311-4ea1-9e9b-564bf45fe369)
> 이 경우 gpu 4, 5를 가지고 학습을 진행 후 4로 test를 진행합니다. 이후 ckpt/z_no-freeze/ 폴더에 학습 완료된 model이 저장됩니다.

> ![image](https://github.com/user-attachments/assets/fa59b3e5-762d-46e3-8258-05637008f67c)


## 5. Pytorch -> Onnx 변환
- Onnx로 변환을 수행하기 위해서는 최소 VRAM 18~20GB가 요구됩니다.

- 1. 새로운 디렉터리에서 새로 Clone을 하고 환경 세팅 후 다음 패키지를 설치합니다.

  ```bash
  pip install -q gdown onnx onnxscript onnxruntime-gpu==1.18.1
  ```

- 2. 이후 BiRefNet 폴더에 [export_onnx.py](https://drive.google.com/file/d/1FB_kSi9u_4pXGtcQNUAtEeWzFkpGrTU4/view?usp=sharing) 파일을 업로드합니다.

- 3. 이후 현재 위치에서 다음을 실행합니다.
  
  ```shell
  python3 export_onnx.py --pt_path "{pth 파일 경로}" --onnx_save_path "{onnx 저장경로}" --device {사용할 gpu/cpu}
  ```

- 4. 변환 완료
  ![image](https://github.com/user-attachments/assets/aa8ac210-dc93-4f16-bf48-9557231d753c)



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
