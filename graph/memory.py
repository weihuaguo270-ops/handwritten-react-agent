"""
记忆系统 — 语义记忆 + 自动遗忘 + 语义去重更新

核心功能：
  - add():          新增记忆（去重后追加）
  - add_or_update(): 语义去重后写入——如果找到相似记忆且主体相同但内容冲突，用新值替换旧值
  - query():        语义检索
  - remove()/clear(): 删除
  - _prune():       LRU 自动遗忘

面试话术（记忆冲突解决）：
  背景：用户说"我叫张三"，过一会又说"我的名字是李四"。
  问题：两条矛盾记忆共存，查询时老信息可能覆盖新信息。
  方案：存入时对已有记忆做语义相似度扫描。
    - 相似度高（>0.85）：主体相同，直接跳过（去重）
    - 相似度中等（0.6~0.85）：可能主体相同但内容不同 → 用新内容替换旧条目
    - 相似度低（<0.6）：确认为不同事实，作为新条目追加
"""

import json
import os
import time
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# 语义相似度阈值
EXACT_MATCH_THRESHOLD = 0.85    # 高于此值视为同一事实
CONFLICT_THRESHOLD = 0.60       # 高于此值视为主体相同，需检查是否冲突


class Memory:
    """语义记忆，支持增删查 + LRU 自动遗忘 + 语义去重更新"""

    MAX_FACTS = 100

    def __init__(self):
        save_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "memory.json")
        self.save_path = save_path
        self.facts = []
        self.vecs = []
        self.access_count = []
        self.last_access = []
        self.model = SentenceTransformer('BAAI/bge-small-zh-v1.5')
        self._load()

    def _load(self):
        try:
            with open(self.save_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.facts = data.get("facts", [])
            self.vecs = [np.array(v) for v in data.get("vecs", [])]
            self.access_count = data.get("access_count", [0] * len(self.facts))
            self.last_access = data.get("last_access", [0] * len(self.facts))
        except Exception:
            pass

    def _save(self):
        with open(self.save_path, "w", encoding="utf-8") as f:
            json.dump({
                "facts": self.facts,
                "vecs": [[round(float(x), 4) for x in v] for v in self.vecs],
                "access_count": self.access_count,
                "last_access": self.last_access,
            }, f, ensure_ascii=False, separators=(",", ":"))

    def add(self, fact: str) -> bool:
        """
        直接追加一条记忆（简单去重）。

        只做精确去重（字符串完全相同才跳过），
        不做语义去重。语义去重请用 add_or_update()。
        """
        if fact not in self.facts:
            self.facts.append(fact)
            self.vecs.append(self.model.encode(fact))
            self.access_count.append(0)
            self.last_access.append(0)
            self._prune()
            self._save()
            return True
        return False

    def add_or_update(self, new_fact: str) -> tuple:
        """
        语义去重后写入记忆。

        流程：
          1. 对新事实做向量化
          2. 与已有所有记忆做余弦相似度对比
          3. 根据相似度决定策略：

            相似度       | 判断         | 行为
            ------------|-------------|-----------------------------
            >= 0.85     | 同一事实     | 跳过（已存在，无需更新）
            0.60 ~ 0.85 | 主体相似     | 用新内容替换旧条目（更新）
            < 0.60      | 不同事实     | 作为新条目追加

        参数:
            new_fact: 要存入的新事实字符串

        返回:
            ("skipped", reason)  /  ("updated", old_fact)  /  ("added", None)
        """
        if not new_fact.strip():
            return ("skipped", "空内容")

        new_vec = self.model.encode(new_fact)

        if not self.facts:
            # 记忆库为空，直接追加
            self.facts.append(new_fact)
            self.vecs.append(new_vec)
            self.access_count.append(0)
            self.last_access.append(0)
            self._save()
            return ("added", None)

        # 与所有已有记忆做语义相似度对比
        scores = cosine_similarity([new_vec], self.vecs)[0]
        best_idx = int(scores.argsort()[::-1][0])
        best_score = float(scores[best_idx])

        if best_score >= EXACT_MATCH_THRESHOLD:
            # 语义高度相似，视为同一事实——跳过
            return ("skipped", f"与已有记忆重复（相似度 {best_score:.2f}）")

        if best_score >= CONFLICT_THRESHOLD:
            # 主体相似但内容可能不同——用新内容替换旧条目
            old_fact = self.facts[best_idx]
            self.facts[best_idx] = new_fact
            self.vecs[best_idx] = new_vec
            self.access_count[best_idx] = 0
            self.last_access[best_idx] = time.time()
            self._save()
            return ("updated", old_fact)

        # 完全不同的事实——追加为新条目
        self.facts.append(new_fact)
        self.vecs.append(new_vec)
        self.access_count.append(0)
        self.last_access.append(0)
        self._prune()
        self._save()
        return ("added", None)

    def query(self, question: str, top_k: int = 3) -> list:
        if not self.facts:
            return []
        try:
            q_vec = self.model.encode(question)
            scores = cosine_similarity([q_vec], self.vecs)[0]
            results = []
            for idx in scores.argsort()[::-1][:top_k]:
                if scores[idx] > 0.3:
                    results.append({"fact": self.facts[idx], "score": float(scores[idx])})
                    self.access_count[idx] += 1
                    self.last_access[idx] = time.time()
            if results:
                self._save()
            return results
        except Exception:
            return []

    def remove(self, fact_or_query: str) -> int:
        if fact_or_query in self.facts:
            self._remove_at(self.facts.index(fact_or_query))
            self._save()
            return 1
        for i, f in enumerate(self.facts):
            if fact_or_query in f:
                self._remove_at(i)
                self._save()
                return 1
        try:
            q_vec = self.model.encode(fact_or_query)
            scores = cosine_similarity([q_vec], self.vecs)[0]
            best = scores.argsort()[::-1][0]
            if scores[best] > 0.4:
                self._remove_at(best)
                self._save()
                return 1
        except Exception:
            pass
        return 0

    def _remove_at(self, idx: int):
        self.facts.pop(idx)
        self.vecs.pop(idx)
        self.access_count.pop(idx)
        self.last_access.pop(idx)

    def _prune(self):
        while len(self.facts) > self.MAX_FACTS:
            now = time.time()
            scores = []
            for i in range(len(self.facts)):
                age = now - self.last_access[i] if self.last_access[i] > 0 else 999999
                scores.append((self.access_count[i], -age, i))
            scores.sort()
            idx = scores[0][2]
            self._remove_at(idx)

    def clear(self):
        self.facts.clear()
        self.vecs.clear()
        self.access_count.clear()
        self.last_access.clear()
        self._save()


MEMORY = Memory()
