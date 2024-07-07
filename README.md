
# 프로젝트 개요
이 프로젝트는 Transformer 모델을 사용하여 텍스트를 생성하는 데 초점을 맞추고 있습니다. 이 프로젝트에는 모델 정의, 데이터 로드 및 전처리, 모델 학습 및 평가, 그리고 텍스트 생성 기능이 포함되어 있습니다.

## 파일 설명
- `dataset.py`: 데이터셋을 로드하고 전처리하는 코드가 포함되어 있습니다.
- `main.ipynb`: 프로젝트의 주요 실행 파일로, 모델 학습 및 평가, 텍스트 생성 과정을 포함합니다.
- `model.py`: Transformer 모델 정의가 포함되어 있습니다.
- `utils.py`: 유틸리티 함수들이 포함되어 있습니다.
- `README.md`: 프로젝트 설명 및 사용법이 포함되어 있습니다.

## 사전 준비
1. 필수 라이브러리 설치
    ```bash
    pip install torch pandas sentencepiece tqdm wandb
    ```

2. 프로젝트 디렉토리 구조
    ```
    project_root/
    │
    ├── dataset.py
    ├── main.ipynb
    ├── model.py
    ├── utils.py
    ├── README.md
    └── weight/
        └── best_model_name.pt  # 학습된 모델 가중치 파일
    ```

## 사용법

### 모델 학습
모델을 학습시키려면 `main.ipynb` 노트북을 열고, 각 셀을 순서대로 실행하세요. 노트북 내에는 데이터 로드, 모델 정의, 학습, 평가 등의 단계가 포함되어 있습니다.

### 모델 가중치 로드 및 텍스트 생성
모델 학습 후, 학습된 가중치를 사용하여 텍스트를 생성할 수 있습니다. 다음은 저장된 가중치를 로드하고 텍스트를 생성하는 예시 코드입니다.

    ```python
    import torch
    import sentencepiece as spm
    from model import *
    from utils import *

    # 모델 인스턴스 생성
    model = Transformer(n_seq, n_vocab, n_dim, n_head, n_layer, device=device).to(device)

    # 저장된 가중치 로드
    model_name = "your_model_name"
    save_path = f'./weight/best_{model_name}.pt'
    state_dict = torch.load(save_path)

    # 모델에 가중치 로드
    model.load_state_dict(state_dict)

    # 모델을 평가 모드로 전환 (옵션)
    model.eval()

    # SentencePiece 모델 로드
    sp = spm.SentencePieceProcessor(model_file='your_spm_model.model')

    # 입력 문장
    prompt = "내일 날씨 어때?"
    output = generate_sentc(model, sp, n_seq=50, prompt=prompt, device=device)

    print("입력 문장:", prompt)
    print("출력 문장:", output)
    

## wandb 설정
모델 학습 중 메트릭을 추적하려면 [wandb](https://wandb.ai/) 계정을 생성하고, 아래와 같이 설정합니다.

    ```python
    import wandb

    wandb.init(project="your_project_name")
    wandb.config = {
        "learning_rate": 0.001,
        "epochs": 100,
        "batch_size": 32
    }

    # 모델 학습 중 메트릭 로깅 예시
    train_metrics = {'train_loss': train_loss / len(train_dataloader)}
    wandb.log(train_metrics, step=epoch)

    val_metrics = {'val_loss': val_loss / len(val_dataloader)}
    wandb.log(val_metrics, step=epoch)
    

위 코드에서는 `wandb.init()`을 통해 프로젝트를 초기화하고, 학습 중 각 에폭마다 `wandb.log()`를 사용하여 학습 및 검증 손실을 기록합니다.

## 참고 문헌
- [PyTorch 공식 문서](https://pytorch.org/docs/stable/index.html)
- [SentencePiece 공식 문서](https://github.com/google/sentencepiece)
- [W&B 공식 문서](https://docs.wandb.ai/)

이 문서가 프로젝트를 이해하고 사용하는 데 도움이 되기를 바랍니다. 추가적인 질문이나 문제가 발생하면 언제든지 문의해 주세요.
