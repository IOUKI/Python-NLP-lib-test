from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk import FreqDist
from nltk.stem import PorterStemmer
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# nltk.download('punkt') # 下載斷詞庫
# nltk.download('stopwords') # 下載停用詞庫
nltk.download('vader_lexicon') # 下載情感分析庫
text = "Hello, how are you doing? I'm doing fine. What's your name?"
tokens = word_tokenize(text) # 英文斷詞
print('英文斷詞', tokens)
sentences = sent_tokenize(text) # 英文斷句
print('英文斷句：', sentences)
stopwords = set(['how']) # 英文停用詞
filteredTokens = [word for word in tokens if word.lower() not in stopwords] # 過濾停用詞
print('過濾停用詞：', filteredTokens)
freqDist = FreqDist(filteredTokens) # 計算詞頻
print('詞頻：', freqDist.most_common(2)) # 取得最常出現的兩個詞

words = ['running', 'jumps', 'quickly', 'restart', 'programing', 'swimming', 'easily']
stemmer = PorterStemmer()
stemmedWords = [stemmer.stem(word) for word in words] # 英文詞幹提取
print('詞幹提取：', stemmedWords)

sia = SentimentIntensityAnalyzer()
sentimentScore = sia.polarity_scores('You ought to be ashamed of yourself.') # 情感分析
sentiment = ''
if sentimentScore['compound'] >= 0.05:
    sentiment = 'positive'
elif sentimentScore['compound'] <= -0.05:
    sentiment = 'negative'
else:
    sentiment = 'neutral'

print('情感分析：', sentimentScore, sentiment)