import os
import json
import glob
import torch
import pickle
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ==========================================
# 1. 設定區
# ==========================================
# 輸入資料夾 (剛剛 Qwen 跑完的 JSON 位置)
INPUT_DIR = "/home/maxwell/data/nlp/final_project/dataset/WattBot2025/local_llm_processed"
# 輸出資料夾 (存 Embedding 的位置)
OUTPUT_DIR = "/home/maxwell/data/nlp/final_project/dataset/WattBot2025/embeddings"

# 使用的 Embedding 模型 (推薦 BAAI/bge-m3，支援長文本與多語言)
MODEL_NAME = "BAAI/bge-m3"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ==========================================
# 2. Encoder 類別
# ==========================================
class WattBotEncoder:
    def __init__(self, model_name=MODEL_NAME):
        print(f"🚀 正在載入 Embedding 模型: {model_name}...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=device)
        print(f"✅ 模型載入完成 (Device: {device})")

    def encode(self, texts):
        """
        將文字列表轉為向量
        Output: Numpy array
        """
        # normalize_embeddings=True 對於計算 Cosine Similarity 很重要
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False, batch_size=32)

# ==========================================
# 3. 資料處理函式 (包含錯誤修復)
# ==========================================
def process_file(json_file_path, encoder):
    filename = os.path.basename(json_file_path)
    doc_id = os.path.splitext(filename)[0]
    
    with open(json_file_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print(f"⚠️ [Skip] 無法讀取 JSON: {filename}")
            return None

    # 用來收集這一份文件裡所有要 embedding 的文字
    chunks_text = []
    chunks_metadata = []

    for item in data:
        # --- 🛡️ 關鍵修復：安全檢查 ---
        content = item.get('content')
        caption = item.get('caption')
        
        # 1. 如果 content 是 None，給它空字串，避免 AttributeError
        if content is None:
            content = ""
        
        # 2. 確保是字串型態 (有時候可能是數字或物件)
        if not isinstance(content, str):
            content = str(content)
            
        # 3. 去除空白
        text_to_embed = content.strip()
        
        # 4. 如果是表格或圖片，把 Caption 加進去一起 embedding，增加檢索準確度
        if caption and isinstance(caption, str) and caption.strip():
            text_to_embed = f"{caption.strip()}\n{text_to_embed}"

        # 5. 如果最後文字是空的（例如有些圖片描述失敗），就跳過
        if not text_to_embed:
            continue
            
        chunks_text.append(text_to_embed)
        chunks_metadata.append(item)

    if not chunks_text:
        return None

    # --- 批量進行 Encoding ---
    # 這樣比一筆一筆 encode 快非常多
    embeddings = encoder.encode(chunks_text)

    # --- 整合結果 ---
    processed_results = []
    for i, meta in enumerate(chunks_metadata):
        # 把向量存進去 (轉成 List 方便存 JSON/Pickle)
        meta['embedding'] = embeddings[i].tolist()
        # 為了確認我們 embed 了什麼，把組合好的文字也存回去
        meta['embedded_text'] = chunks_text[i] 
        processed_results.append(meta)

    return processed_results

# ==========================================
# 4. 主程式
# ==========================================
def main():
    # 1. 初始化 Encoder
    encoder = WattBotEncoder()

    # 2. 抓取所有 JSON
    json_files = glob.glob(os.path.join(INPUT_DIR, "*.json"))
    print(f"📂 找到 {len(json_files)} 個 JSON 檔案，開始 Embedding...")

    all_documents_data = []

    # 3. 處理每個檔案
    for json_file in tqdm(json_files, desc="Encoding Files"):
        result = process_file(json_file, encoder)
        if result:
            all_documents_data.extend(result)

    # 4. 存檔 (儲存為 Pickle，因為讀寫向量最快)
    output_pkl = os.path.join(OUTPUT_DIR, "corpus_embeddings.pkl")
    print(f"💾 正在儲存 {len(all_documents_data)} 筆向量資料到 {output_pkl} ...")
    
    with open(output_pkl, 'wb') as f:
        pickle.dump(all_documents_data, f)

    # 5. 額外存一份 JSONL (方便人類檢查，不含向量以免檔案太大)
    output_jsonl = os.path.join(OUTPUT_DIR, "corpus_text_only.jsonl")
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for entry in all_documents_data:
            # 複製一份不含 embedding 的資料
            entry_copy = entry.copy()
            del entry_copy['embedding']
            f.write(json.dumps(entry_copy, ensure_ascii=False) + '\n')

    print(f"🎉 全部完成！")
    print(f"   - 向量檔: {output_pkl}")
    print(f"   - 文字檔: {output_jsonl}")

if __name__ == "__main__":
    main()