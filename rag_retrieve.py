import json
import numpy as np
import os
from typing import List, Dict, Optional, Tuple

# 確保已安裝必要的套件
try:
    import faiss
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("缺少必要的套件。請執行:")
    print("pip install sentence-transformers faiss-cpu")
    exit(1)

class VectorSearchEngine:
    """
    一個封裝了句子嵌入、FAISS 索引和語意搜尋的向量搜尋引擎。
    """
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        初始化引擎，載入指定的 SentenceTransformer 模型。
        :param model_name: 要使用的預訓練模型名稱。
        """
        print(f"正在初始化模型 '{model_name}'...")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index: Optional[faiss.Index] = None
        self.documents: List[str] = []
        self.urls: List[str] = []
        self.document_ids: List[int] = []
        print("模型初始化完成。")

    def _load_and_filter_documents(self, json_path: str) -> Tuple[List[str], List[int]]:
        """從 JSON 檔案中讀取資料、過濾重複內容並提取 'content' 欄位。"""
        print(f"從 {json_path} 載入並過濾文件...")
        documents = []
        ids = []
        urls = []
        seen_content = set()
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for item in data:
            if "content" in item and item["content"] and item["content"].strip():
                content = item["content"]
                if content not in seen_content:
                    documents.append(content)
                    ids.append(item.get('doc_id'))
                    urls.append(item.get('url', ''))
                    seen_content.add(content)
        print(f"成功載入 {len(documents)} 個不重複的文件片段。")
        return documents, ids, urls

    def build_from_json(self, json_path: str):
        """
        從 JSON 檔案建立完整的 FAISS 索引。
        :param json_path: JSON 檔案的路徑。
        """
        self.documents, self.document_ids, self.urls = self._load_and_filter_documents(json_path)
        if not self.documents:
            print("錯誤: 文件中沒有找到可處理的內容。")
            return

        print("正在將文件編碼為向量...")
        embeddings = self.model.encode(self.documents, show_progress_bar=True)
        
        print("建立 FAISS 索引...")
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(np.array(embeddings))
        print(f"索引建立完成，共包含 {self.index.ntotal} 個向量。")

   
        
    # 重新編寫 insert 方法
    def insert(self, new_documents: List[str]):
        """
        向現有索引中插入新的文件。
        :param new_documents: 一個包含新文本字串的列表。
        """
        # 1. 過濾重複
        if not self.documents:
            # 第一次插入
            filtered_docs = list(set(new_documents))
        else:
            filtered_docs = [doc for doc in new_documents if doc not in self.documents]

        if not filtered_docs:
            print("沒有新的不重複文件可供插入。")
            return

        print(f"正在插入 {len(filtered_docs)} 個新文件...")
        new_embeddings = self.model.encode(filtered_docs, show_progress_bar=True)
        
        if self.index is None:
            self.index = faiss.IndexFlatL2(self.dimension)

        self.index.add(np.array(new_embeddings))
        
        # 生成 ID
        start_id = max(self.document_ids) + 1 if self.document_ids else 0
        new_ids = [start_id + i for i in range(len(filtered_docs))]
        
        self.documents.extend(filtered_docs)
        self.document_ids.extend(new_ids)
        
        print(f"插入完成。索引現在包含 {self.index.ntotal} 個向量。")

    def get(self, query: str, k: int = 5) -> List[Dict]:
        """
        在索引中執行語意搜尋。
        :param query: 查詢的文本字串。
        :param k: 要回傳的最相似結果數量。
        :return: 一個包含結果的字典列表，每個字典包含 'score' 和 'text'。
        """
        if self.index is None or self.index.ntotal == 0:
            print("索引為空，無法執行搜尋。")
            return []

        # print(f"\n執行搜尋: \"{query}\"")
        query_embedding = self.model.encode([query])
        distances, indices = self.index.search(np.array(query_embedding), k)

        results = []
        if len(indices[0]) > 0:
            for i, doc_index in enumerate(indices[0]):
                if doc_index < len(self.documents):
                    result = {
                        "id": self.document_ids[doc_index],
                        "score": float(distances[0][i]),
                        "text": self.documents[doc_index],
                        "url": self.urls[doc_index]
                    }
                    results.append(result)
        return results

def main():
    """主執行函數，展示如何使用 VectorSearchEngine 類別。"""
    json_file_path = "./corpus_text_only.json"
    if not os.path.exists(json_file_path):
        print(f"錯誤: 找不到檔案 '{json_file_path}'")
        return

    # 1. 建立引擎實例
    engine = VectorSearchEngine()

    # 2. 從 JSON 檔案建立初始索引
    engine.build_from_json(json_file_path)

    # 3. 執行查詢
    query = "Given the total pre-training GPU hours and the number of GPUs used, estimate the total wall-clock time in days required to pre-train the JetMoE-8B model."
    search_results = engine.get(query, k=5)

    print("\n--- 搜尋結果 ---")
    if not search_results:
        print("找不到相關結果。")
    else:
        for i, result in enumerate(search_results):
            print(f"\n結果 {i+1}:")
            print(f"  - ID: {result['id']}")
            print(f"  - 相似度分數 (L2 距離): {result['score']:.4f}")
            print(f"  - 相關文本: \"{result['text']}\"")
    
    


if __name__ == "__main__":
    main()
