import os
import json
import torch
import re
import glob
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import fitz  # PyMuPDF
from PIL import Image
import io

# 嘗試匯入 json_repair，這是處理 LLM 爛 JSON 的神器
try:
    import json_repair
    HAS_JSON_REPAIR = True
except ImportError:
    HAS_JSON_REPAIR = False
    print("⚠️ 建議執行 'pip install json_repair' 以獲得最強的容錯能力！")

# ==========================================
# 1. 設定區 & 模型載入
# ==========================================
# 設定資料路徑
BASE_DIR = "/home/maxwell/data/nlp/final_project/dataset/WattBot2025"
PDF_DIR = os.path.join(BASE_DIR, "pdf")
OUTPUT_DIR = os.path.join(BASE_DIR, "local_llm_processed")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print("🚀 [Init] 正在載入 Qwen2-VL-7B-Instruct 到 RTX 5090...")

# 載入模型
# 使用 bfloat16 以節省顯存並加速 (5090 完美支援)
# 使用 flash_attention_2 進行極致加速
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto"
)

# 載入處理器
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

print("✅ [Init] 模型載入完成！準備開始 GPU 運算。")

# ==========================================
# 2. 定義 Prompt (針對 Qwen 優化)
# ==========================================
SYSTEM_PROMPT = """
You are an expert in digitizing academic documents. Analyze this image of a paper page.
Extract ALL content into a structured JSON list following the reading order (Left column first, then Right column).

**Extraction Rules:**
1. **Reading Order:** Strictly follow the logical reading order of a scientific paper.
2. **Structure:** Extract Text, Headers, Tables, and Figures.
3. **Noise:** Ignore running headers, page numbers, and decorative lines.

**Output Format (Strict JSON List of Objects):**
[
  {"type": "header", "content": "Section Title (e.g., 1. Introduction)"},
  {"type": "text", "content": "Full text of the paragraph..."},
  {"type": "table", "caption": "Table 1: Title...", "content": "Markdown representation of the table"},
  {"type": "figure", "caption": "Figure 1: Title...", "content": "Detailed description of the image content/trends"}
]

Return ONLY the valid JSON string. Do not add markdown code blocks (```json).
"""

# ==========================================
# 3. 核心功能 (本地推論)
# ==========================================
def extract_with_local_vlm(page_image):
    """
    使用 Qwen2-VL 進行本地推論
    """
    # 建構訊息格式
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": page_image, # 直接傳入 PIL Image
                },
                {"type": "text", "text": SYSTEM_PROMPT},
            ],
        }
    ]

    # 1. 預處理輸入
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    
    # 移至 GPU
    inputs = inputs.to("cuda")

    # 2. 生成 (Inference)
    # max_new_tokens 設為 4096 確保長文本 (如表格) 不會被切斷
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=4096)
    
    # 3. 解碼輸出
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    return output_text

def parse_json_output(text):
    """
    強壯的 JSON 解析器 (v2.0)：
    1. 使用 Regex 抓取 JSON 區塊
    2. 使用 json_repair 自動修復格式錯誤
    """
    text = text.strip()
    
    # --- 防護 1: 使用 Regex 抓取最外層的 List [...] ---
    # 防止模型在 JSON 前後講廢話，或者沒寫 markdown block
    pattern = r"\[\s*\{.*\}\s*\]"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        text = match.group()
    else:
        # 如果 Regex 沒抓到完整的，嘗試從第一個 '[' 抓到最後
        start_idx = text.find('[')
        if start_idx != -1:
            text = text[start_idx:]

    # --- 防護 2: 嘗試解析 ---
    try:
        # 先試標準解析
        return json.loads(text)
    except json.JSONDecodeError as e:
        # --- 防護 3: 如果失敗，使用 json_repair ---
        if HAS_JSON_REPAIR:
            try:
                # json_repair 可以修補未閉合的引號、缺少的逗號等常見 LLM 錯誤
                repaired_obj = json_repair.loads(text)
                print(f"   🔧 JSON 格式有誤 (Raw len: {len(text)})，已自動修復！")
                return repaired_obj
            except Exception:
                pass # 如果連 repair 都失敗，就往下走
        
        print(f"   ⚠️ JSON 解析嚴重失敗 (Raw text length: {len(text)})")
        # 回傳包含原始文字的錯誤物件，避免資料遺失
        return [{
            "type": "error_parsing", 
            "content": "JSON Parsing Failed",
            "raw_content": text
        }]

def process_pdf(pdf_path):
    doc_id = os.path.splitext(os.path.basename(pdf_path))[0]
    output_json_path = os.path.join(OUTPUT_DIR, f"{doc_id}.json")
    
    # 檢查是否已存在，支援中斷後續傳
    if os.path.exists(output_json_path):
        print(f"⏩ {doc_id} 已存在，跳過。")
        return

    print(f"🚀 開始處理: {doc_id}")
    
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"❌ 無法開啟: {e}")
        return

    all_data = []
    global_idx = 0  # 全域 ID 計數器

    for page_num, page in enumerate(doc):
        current_page = page_num + 1
        print(f"   -> Page {current_page}/{len(doc)}...", end="", flush=True)

        # 1. 轉圖片 (300 DPI 對 OCR 很重要)
        pix = page.get_pixmap(dpi=300)
        img = Image.open(io.BytesIO(pix.tobytes("png")))

        # 2. 本地推論 (速度取決於你的 GPU，5090 應該非常快)
        try:
            raw_output = extract_with_local_vlm(img)
            extracted_items = parse_json_output(raw_output)
            
            if extracted_items:
                # 計算有效區塊 (排除 error_parsing)
                valid_items = [x for x in extracted_items if x.get('type') != 'error_parsing']
                print(f" ✅ 抓到 {len(valid_items)} 個區塊")
                
                # 補上 metadata
                for item in extracted_items:
                    # 如果是解析錯誤，保留它以便 Debug，但不給 id
                    if item.get('type') != 'error_parsing':
                        item['id'] = global_idx
                        global_idx += 1
                    
                    item['doc_id'] = doc_id
                    item['page'] = current_page
                    all_data.append(item)
            else:
                print(f" ⚠️ 空內容")
                
        except Exception as e:
            print(f" ❌ GPU Inference Error: {e}")

        # 不需要 sleep，因為是本地運算

    # 存檔
    with open(output_json_path, "w", encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)
    
    print(f"🎉 完成！存檔至: {output_json_path}\n")

# ==========================================
# 4. 執行 (批量處理)
# ==========================================
if __name__ == "__main__":
    # 使用 glob 抓取資料夾下所有 .pdf 檔案
    pdf_files = glob.glob(os.path.join(PDF_DIR, "*.pdf"))
    
    if not pdf_files:
        print(f"❌ 在 {PDF_DIR} 找不到任何 PDF 檔案。")
    else:
        print(f"📂 找到 {len(pdf_files)} 個 PDF 檔案，5090 引擎全開！")
        print("="*50)
        
        for i, pdf_path in enumerate(pdf_files):
            filename = os.path.basename(pdf_path)
            print(f"🔄 [{i+1}/{len(pdf_files)}] 正在處理: {filename}")
            process_pdf(pdf_path)
            print("-" * 50)
            
        print("✅ 所有 PDF 處理完成！")