import json

def jsonl_to_json(input_jsonl_path, output_json_path):
    """
    將 JSONL 檔案轉換為單一的 JSON 陣列檔案，並在轉換時將 'content' 欄位（如果是列表）合併成單一字串。
    """
    data_list = []

    # 1. 讀取 JSONL 檔案 (一行一行讀取)
    try:
        with open(input_jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line.strip())
                        
                        # --- 關鍵的內容預處理邏輯 START ---
                        if 'content' in record:
                            content_value = record['content']
                            
                            if isinstance(content_value, list):
                                # 如果 'content' 是列表，將列表中的所有元素用空格連接成一個單一字串
                                # 使用 map(str, ...) 確保列表中的元素即使不是字串也能被處理
                                record['content'] = " ".join(map(str, content_value)).strip()
                            
                            elif not isinstance(content_value, str):
                                # 如果 'content' 既不是列表也不是字串 (例如是數字、布林值)，強制轉為字串
                                record['content'] = str(content_value).strip()

                            # 確保處理完畢後，如果內容是空字串，我們將其設為 None 或進行進一步過濾
                            if not record['content']:
                                record['content'] = None 
                        # --- 關鍵的內容預處理邏輯 END ---

                        data_list.append(record)
                        
                    except json.JSONDecodeError as e:
                        print(f"警告：跳過無效的 JSON 行: {line.strip()}，錯誤: {e}")
    except FileNotFoundError:
        print(f"錯誤：找不到檔案 {input_jsonl_path}")
        return

    # 2. 將 Python 物件列表 (List) 寫入 JSON 檔案
    print(f"成功讀取和處理 {len(data_list)} 筆記錄。")
    with open(output_json_path, 'w', encoding='utf-8') as f:
        # indent=4 使輸出檔案具有良好的縮排格式
        json.dump(data_list, f, ensure_ascii=False, indent=4)
    
    print(f"✅ 轉換完成！輸出檔案為：{output_json_path}")


# --- 執行範例 ---
# 請將這裡的路徑替換為您實際的檔案名稱
INPUT_FILE = 'corpus_text_only.jsonl'
OUTPUT_FILE = 'corpus_text_only.json'

jsonl_to_json(INPUT_FILE, OUTPUT_FILE)