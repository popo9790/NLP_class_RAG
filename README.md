
### 1. 建立 Conda 環境

使用 `r.yml` 快速建立執行環境：

```bash
conda env create -f r.yml
conda activate rag_environment
```

### 2. 下載 NLTK 必要資源

在執行搜尋之前，請先確保 NLTK 的模型已下載：

```bash
python download_nltk_data.py
```

---

## 🚀 使用方法

### 1. 向量搜尋 (Vector Search)

基於語意相似度進行搜尋


**使用：**
engine = VectorSearchEngine()
```bash
    engine.build_from_json(json_file_path)
    query = "Given the total pre-training GPU hours and the number of GPUs used, estimate the total wall-clock time in days required to pre-train the JetMoE-8B model."
    search_results = engine.get(query, k=5)
```

### 2. 名詞搜尋 (Noun Search)

基於關鍵字搜尋

```bash
    engine = NounSearchEngine(json_file_path)
    query = "what are the key advantages of JetMoE-8B model?"
    search_results = engine.search(query, top_k=5)

```


