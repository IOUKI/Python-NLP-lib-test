from nltk import FreqDist
import jieba

chineseSentence = "自然語言處理是人工智慧領域的一部分。這是一個很有趣的領域。"
chineseTokens = jieba.lcut(chineseSentence) # 中文分詞
print(chineseTokens)

freqDist = FreqDist(chineseTokens) # 計算詞頻
print('詞頻：', freqDist.most_common(2)) # 取得最常出現的兩個詞