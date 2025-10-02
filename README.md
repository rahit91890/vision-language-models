# Vision Language Models Toolkit

Computer Vision and Multimodal AI toolkit featuring CLIP, LLaVA, BLIP-2, and CogVLM integrations.

## 🚀 Overview

This repository provides a unified interface for working with state-of-the-art Vision Language Models (VLMs). Integrate powerful multimodal AI capabilities into your applications with support for:

- **CLIP** by OpenAI - Text-image understanding model
- **LLaVA** - Large Language and Vision Assistant combining language and visual processing
- **BLIP-2** - Bootstrapping language-image pre-training
- **CogVLM** - Cognitive visual language model with deep fusion techniques

## 📁 Project Structure

```
vision-language-models/
├── clip/                  # OpenAI CLIP integration
│   ├── __init__.py
│   ├── model.py
│   └── utils.py
├── llava/                 # LLaVA integration
│   ├── __init__.py
│   ├── model.py
│   └── utils.py
├── blip2/                 # BLIP-2 integration
│   ├── __init__.py
│   ├── model.py
│   └── utils.py
├── cogvlm/                # CogVLM integration
│   ├── __init__.py
│   ├── model.py
│   └── utils.py
├── adapters/              # Model adapters and bridges
│   ├── __init__.py
│   └── base_adapter.py
├── pipeline/              # Main pipeline orchestration
│   ├── __init__.py
│   └── main_pipeline.py
├── .env.example           # Environment variables template
├── .gitignore
├── requirements.txt       # Python dependencies
└── README.md
```

## 🔧 Installation

### Prerequisites

- Python 3.8+
- pip or conda package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/rahit91890/vision-language-models.git
cd vision-language-models
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables:
```bash
cp .env.example .env
```

4. Edit `.env` with your API keys and configurations (see Configuration section below).

## ⚙️ Configuration

### Environment Variables

Copy `.env.example` to `.env` and configure the following:

```bash
# OpenAI CLIP Configuration
CLIP_MODEL_NAME=ViT-B/32
CLIP_DEVICE=cuda  # or 'cpu'

# LLaVA Configuration
LLAVA_MODEL_PATH=path/to/llava/model
LLAVA_API_KEY=your_llava_api_key_here
LLAVA_DEVICE=cuda

# BLIP-2 Configuration
BLIP2_MODEL_TYPE=Salesforce/blip2-opt-2.7b
BLIP2_API_KEY=your_blip2_api_key_here
BLIP2_DEVICE=cuda

# CogVLM Configuration
COGVLM_MODEL_PATH=path/to/cogvlm/model
COGVLM_API_KEY=your_cogvlm_api_key_here
COGVLM_DEVICE=cuda

# General Settings
MAX_BATCH_SIZE=32
CACHE_DIR=./model_cache
LOG_LEVEL=INFO
```

### API Keys

- **OpenAI CLIP**: Available through Hugging Face or OpenAI
- **LLaVA**: Obtain from LLaVA project repository
- **BLIP-2**: Available through Hugging Face
- **CogVLM**: Obtain from CogVLM project repository

### Dependencies

Key dependencies (automatically installed via requirements.txt):
- `torch>=2.0.0`
- `transformers>=4.30.0`
- `pillow>=9.0.0`
- `numpy>=1.20.0`
- `python-dotenv>=0.19.0`
- `openai-clip`
- `accelerate>=0.20.0`

## 🎯 Usage

### Basic Example

```python
from pipeline.main_pipeline import VisionLanguagePipeline

# Initialize the pipeline
pipeline = VisionLanguagePipeline(model='clip')

# Process an image with text
image_path = 'path/to/image.jpg'
text = 'A photo of a cat'

result = pipeline.process(image_path, text)
print(result)
```

### Using Different Models

```python
# Use LLaVA
llava_pipeline = VisionLanguagePipeline(model='llava')

# Use BLIP-2
blip2_pipeline = VisionLanguagePipeline(model='blip2')

# Use CogVLM
cogvlm_pipeline = VisionLanguagePipeline(model='cogvlm')
```

## 📝 License

MIT License - see LICENSE file for details

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

Rahit Biswas - [@Rahit1996](https://twitter.com/Rahit1996) - r.codaphics@gmail.com

Project Link: [https://github.com/rahit91890/vision-language-models](https://github.com/rahit91890/vision-language-models)
