import nltk
import ssl

def download_nltk_resources():
    """
    下載 NLTK 所需的核心資源，並處理可能的 SSL 憑證問題。
    """
    # 定義資源及其對應的子目錄
    resources = {
        'punkt': 'tokenizers',
        'averaged_perceptron_tagger': 'taggers',
        'punkt_tab': 'tokenizers'
    }
    
    print("正在檢查並下載 NLTK 資源...")
    
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        # 如果系統不支援或不需要，則忽略
        pass
    else:
        # 建立一個不驗證 SSL 憑證的 context，以避免下載時出錯
        ssl._create_default_https_context = _create_unverified_https_context

    for resource, subdir in resources.items():
        try:
            # 嘗試尋找資源
            nltk.data.find(f'{subdir}/{resource}')
            print(f"- 資源 '{resource}' 已存在。")
        except LookupError:
            # 如果找不到，則下載
            print(f"- 正在下載資源 '{resource}'...")
            try:
                nltk.download(resource, quiet=True)
                print(f"- '{resource}' 下載完成。")
            except Exception as e:
                print(f"- 下載 '{resource}' 時發生錯誤: {e}")
            
    print("\n所有必要的 NLTK 資源檢查完畢！")

if __name__ == "__main__":
    download_nltk_resources()
