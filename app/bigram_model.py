from collections import defaultdict, Counter
import random, re
from typing import Iterable, Union

TextLike = Union[str, Iterable[str]]

class BigramModel:
    def __init__(self, text: TextLike):
        toks = self._tokenize(text)
        self.table = self._build(toks)

    def _tokenize(self, text: TextLike):
        # 兼容 list/tuple：你之前的报错是因为传入了 list，list 没有 lower()
        if isinstance(text, (list, tuple)):
            text = " ".join(text)
        if not isinstance(text, str):
            text = str(text)
        text = text.lower()
        # 单词或句末标点
        return re.findall(r"[a-z']+|[.!?]", text)

    def _build(self, tokens):
        tb = defaultdict(Counter)
        for w1, w2 in zip(tokens, tokens[1:]):
            tb[w1][w2] += 1
        return tb

    def _next_word(self, word: str) -> str:
        if word not in self.table or not self.table[word]:
            return random.choice(list(self.table.keys()))
        words, weights = zip(*self.table[word].items())
        return random.choices(words, weights=weights, k=1)[0]

    # 与讲义保持同名接口
    def generate_text(self, start_word: str, length: int) -> str:
        if length <= 0:
            return ""
        w = start_word.lower()
        out = [w]
        for _ in range(length - 1):
            w = self._next_word(w)
            out.append(w)
        return " ".join(out)
