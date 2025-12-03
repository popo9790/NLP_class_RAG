import json
import os
from collections import defaultdict
from typing import List, Dict, Set, Tuple

# 嘗試導入 NLTK 並在需要時下載其必要組件
try:
    import nltk
    # 檢查 'punkt' (用於分詞) 和 'averaged_perceptron_tagger' (用於詞性標註) 是否存在
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("NLTK 'punkt' tokenizer not found. Downloading...")
        nltk.download('punkt', quiet=True)
        print("'punkt' downloaded.")
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        print("NLTK 'averaged_perceptron_tagger' not found. Downloading...")
        nltk.download('averaged_perceptron_tagger', quiet=True)
        print("'averaged_perceptron_tagger' downloaded.")

    try:
        nltk.data.find('averaged_perceptron_tagger_eng')
    except LookupError:
        print("NLTK 'averaged_perceptron_tagger_eng' not found. Downloading...")
        nltk.download('averaged_perceptron_tagger_eng', quiet=True)
        print("'averaged_perceptron_tagger_eng' downloaded.")
except ImportError:
    print("錯誤: NLTK 套件未安裝。")
    print("請執行: pip install nltk")
    exit(1)

class NounSearchEngine:
    """
    一個透過比對名詞來找出相關文件的搜尋引擎。
    """
    def __init__(self, json_path: str):
        """
        初始化引擎，讀取 JSON 檔案並建立名詞索引。
        :param json_path: JSON 資料來源的路徑。
        """
        print("正在初始化名詞搜尋引擎...")
        self.documents: Dict[int, str] = {}
        # 反向索引: { 'noun': {id1, id2, ...} }
        self.inverted_index: Dict[str, Set[int]] = defaultdict(set)
        
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"錯誤: 找不到檔案 '{json_path}'")
            
        self._build_index(json_path)
        print("引擎初始化完成。")

    def _extract_nouns(self, text: str) -> Set[str]:
        """
        從給定文本中提取所有名詞。
        :param text: 輸入的文本字串。
        :return: 一個包含所有名詞的集合 (set)。
        """
        # 不先轉換為小寫，以保留專有名詞 (NNP) 的特徵供詞性標註使用
        tokens = nltk.word_tokenize(text)
        pos_tags = nltk.pos_tag(tokens)
        
        nouns = set()
        # 'NN' (名詞), 'NNS' (複數名詞), 'NNP' (專有名詞), 'NNPS' (複數專有名詞)
        noun_tags = {'NN', 'NNS', 'NNP', 'NNPS'}
        for word, tag in pos_tags:
            if tag in noun_tags and len(word) > 1: # 過濾掉單一字母的名詞
                # 在加入集合時才轉為小寫，確保搜尋時不區分大小寫
                nouns.add(word.lower())
        return nouns

    def _build_index(self, json_path: str):
        """
        讀取 JSON 檔案並建立反向名詞索引。
        """
        print(f"正在從 '{json_path}' 建立索引...")
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            raise TypeError("JSON 檔案的根結構必須是一個列表。")

        for item in data:
            # 確保物件是字典且包含 'doc_id' 和 'content'
            if isinstance(item, dict) and 'doc_id' in item and 'content' in item:
                doc_id = item['doc_id']
                content = item['content']
                
                if not content or not content.strip():
                    continue

                self.documents[doc_id] = content
                nouns = self._extract_nouns(content)
                
                for noun in nouns:
                    self.inverted_index[noun].add(doc_id)
        
        print(f"索引建立完成。共處理 {len(self.documents)} 個文件，發現 {len(self.inverted_index)} 個獨特名詞。")

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        根據查詢中的名詞，搜尋最相關的文件 ID。
        :param query: 查詢的文本字串。
        :param top_k: 要回傳的最相關結果數量。
        :return: 一個包含結果的字典列表。
        """
        print(f"\n執行名詞搜尋: \"{query}\"")
        query_nouns = self._extract_nouns(query)
        
        if not query_nouns:
            print("查詢中未找到可比對的名詞。")
            return []
        
        print(f"查詢中的名詞: {query_nouns}")

        # 計算每個文件的匹配分數
        # defaultdict(int) 會在鍵不存在時自動給予預設值 0
        doc_scores: Dict[int, int] = defaultdict(int)
        for noun in query_nouns:
            if noun in self.inverted_index:
                for doc_id in self.inverted_index[noun]:
                    doc_scores[doc_id] += 1
        
        # 根據分數進行排序
        # sorted() 回傳一個元組列表 (doc_id, score)
        sorted_docs: List[Tuple[int, int]] = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)
        
        # 準備回傳結果
        results = []
        for doc_id, score in sorted_docs[:top_k]:
            results.append({
                "id": doc_id,
                "score": score, # 匹配到的名詞數量
                "content": self.documents.get(doc_id, "內容遺失")
            })
            
        return results

def main():
    """主執行函數，展示如何使用 NounSearchEngine 類別。"""
    json_file_path = "corpus_text_only.json"
    
    try:
        # 1. 建立引擎實例，它會自動讀取檔案並建立索引
        engine = NounSearchEngine(json_file_path)

        # 2. 執行查詢
        query = "what are the key advantages of JetMoE-8B model?"
        search_results = engine.search(query, top_k=5)

        print("\n--- 搜尋結果 ---")
        if not search_results:
            print("找不到相關結果。")
        else:
            for i, result in enumerate(search_results):
                print(f"\n結果 {i+1}:")
                print(f"  - 文件 ID: {result['id']}")
                print(f"  - 匹配分數: {result['match_score']} (個共同名詞)")
                print(f"  - 相關文本: \"{result['content']}\"")

    except (FileNotFoundError, TypeError) as e:
        print(f"發生錯誤: {e}")
    except Exception as e:
        print(f"發生未預期的錯誤: {e}")


if __name__ == "__main__":
    main()
