# Pixel Art Image Generator

GPT와 Z-Image-Turbo 모델을 활용한 픽셀아트 이미지 생성 파이프라인입니다.

## Overview

1. GPT-4o-mini로 한국어 주제 생성 (중복 방지를 위한 임베딩 유사도 체크)
2. 주제를 영어 픽셀아트 프롬프트로 변환
3. Z-Image-Turbo 모델로 1024x1024 픽셀아트 이미지 생성
4. HuggingFace 데이터셋으로 업로드

## Setup

```bash
pip install -r requirements.txt
```

`.env` 파일 생성:
```
OPENAI_API_KEY=your_openai_key
HF_TOKEN=your_huggingface_token
```

## Usage

```bash
# 전체 파이프라인 실행 (프롬프트 + 이미지 생성)
python main.py --num_images 10

# 프롬프트만 생성
python generate_prompts.py --num_prompts 10

# 기존 프롬프트로 이미지 생성
python generate_images.py --prompts_file data/prompts.jsonl

# 특정 인덱스부터 이미지 생성
python generate_images.py --prompts_file data/prompts.jsonl --start_index 500

# HuggingFace에 업로드
python upload_to_hf.py --dataset_dir data/dataset --repo_name username/dataset-name
```

## Project Structure

```
├── main.py              # 메인 파이프라인
├── generate_prompts.py  # 프롬프트 생성 (GPT + 임베딩 유사도)
├── generate_images.py   # 이미지 생성 (Z-Image-Turbo)
├── upload_to_hf.py      # HuggingFace 업로드
├── requirements.txt     # 의존성
└── data/
    ├── prompts.jsonl    # 생성된 프롬프트
    └── dataset/         # 생성된 이미지 + metadata
```

## Features

- **주제 중복 방지**: OpenAI 임베딩 + 코사인 유사도로 비슷한 주제 필터링 (threshold: 0.85)
- **이어서 생성**: 중단 시 metadata.jsonl 기반으로 자동 resume
- **HuggingFace 호환**: ImageFolder 형식으로 바로 업로드 가능

## Requirements

- Python 3.10+
- CUDA GPU (이미지 생성용)
- OpenAI API Key
- HuggingFace Token (업로드 시)

## Dataset

생성된 데이터셋: [mks0813/pixel_image_dataset](https://huggingface.co/datasets/mks0813/pixel_image_dataset)

## License

MIT License
