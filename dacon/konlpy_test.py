from konlpy.tag import Okt
import numpy as np

okt = Okt()

# abc = '아버지가 방에 입장하셨닼ㅋㅋㅋㅋ'
# a = okt.normalize(abc)
# a = okt.pos(a)
# print(a)

abc = np.array([
    '아버지가 방에 입장하셨닼ㅋㅋㅋㅋ',
    '심심한 김에 만들어보는 테스트!',
    '에이비씨디이는 영어 단어이다!!'
])
print(abc)
for i, j in enumerate(abc):
    clean_words = []
    j = okt.normalize(j)
    print(j)
    for word in okt.pos(j, stem=False):
        if word[1] in ['Noun', 'Verb', 'Adjective']:
            clean_words.append(word[0])
    print(clean_words)
    j = ' '.join(clean_words)
    abc[i] = j
print(abc)
