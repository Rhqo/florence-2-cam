### 1. ollama 설치
```
    # ollama 설치
    apt-get install curl
    curl -fsSL https://ollama.com/install.sh | sh

    # ollama-python 설치
    pip install ollama
```

### 2. ollama의 model 설치
```
    # gemma3:4b는 gemma3 라고 작성해야 합니다.
    ollama run gemma3:1b
    ollama run gemma3
    ollama run llama3.2-vision
    # ...
```
사용 가능한 모델들(vision): https://ollama.com/search?c=vision

### 3. `main_local.py` 에 ollama_model 작성
```
    # ollama 모델 설정
    # 가능한 모델 : 'gemma3:1b', 'gemma3', 'gemma3:12b', 'llama3.2-vision', ...

    ollama_model = 'gemma3'
```

### 4. 코드 실행