
"""
Memory — 独立的语义记忆模块
============================
基于 BGE-small-zh-v1.5 的语义记忆系统，支持：
- 增: add() / auto_extract()
- 删: remove() / clear()
- 查: query()
- 自动遗忘: _prune()
"""
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class Memory:
    MAX_FACTS = 100

    def __init__(self, save_path=None):
        if save_path is None:
            import os
            save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "memory.json")
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
            print(f"[记忆] 已加载 {len(self.facts)} 条记忆")
        except:
            self.facts = []
            self.vecs = []
            self.access_count = []
            self.last_access = []

    def _save(self):
        with open(self.save_path, "w", encoding="utf-8") as f:
            json.dump({
                "facts": self.facts,
                "vecs": [[round(float(x), 4) for x in v] for v in self.vecs],
                "access_count": self.access_count,
                "last_access": self.last_access,
            }, f, ensure_ascii=False, separators=(",", ":"))

    def add(self, fact):
        if fact not in self.facts:
            self.facts.append(fact)
            vec = self.model.encode(fact)
            self.vecs.append(vec)
            self.access_count.append(0)
            self.last_access.append(0)
            self._prune()
            self._save()
            return True
        return False

    # ================================================================
    # 新增：语义去重更新（与 graph/memory.py 的 add_or_update 一致）
    # ================================================================
    # 相似度阈值
    _EXACT_MATCH = 0.85
    _CONFLICT = 0.60

    def add_or_update(self, new_fact):
        """
        语义去重后写入记忆。

        相似度 >= 0.85 → 同一事实，跳过
        相似度 0.60~0.85 → 主体相同但内容不同，用新内容替换旧条目
        相似度 < 0.60 → 不同事实，作为新条目追加

        返回:
            ("skipped", reason) / ("updated", old_fact) / ("added", None)
        """
        if not new_fact.strip():
            return ("skipped", "空内容")

        new_vec = self.model.encode(new_fact)

        if not self.facts:
            self.facts.append(new_fact)
            self.vecs.append(new_vec)
            self.access_count.append(0)
            self.last_access.append(0)
            self._save()
            return ("added", None)

        scores = cosine_similarity([new_vec], self.vecs)[0]
        best_idx = int(scores.argsort()[::-1][0])
        best_score = float(scores[best_idx])

        if best_score >= self._EXACT_MATCH:
            return ("skipped", f"与已有记忆重复（相似度 {best_score:.2f}）")

        if best_score >= self._CONFLICT:
            old_fact = self.facts[best_idx]
            self.facts[best_idx] = new_fact
            self.vecs[best_idx] = new_vec
            self.access_count[best_idx] = 0
            self.last_access[best_idx] = __import__("time").time()
            self._save()
            return ("updated", old_fact)

        self.facts.append(new_fact)
        self.vecs.append(new_vec)
        self.access_count.append(0)
        self.last_access.append(0)
        self._prune()
        self._save()
        return ("added", None)

    def query(self, question, top_k=3):
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
                    self.last_access[idx] = __import__("time").time()
            if results:
                self._save()
            return results
        except Exception:
            return []

    def remove(self, fact_or_query):
        # 1) 精确匹配
        if fact_or_query in self.facts:
            self._remove_at(self.facts.index(fact_or_query))
            self._save()
            return 1
        # 2) 关键词包含
        for i, f in enumerate(self.facts):
            if fact_or_query in f:
                self._remove_at(i)
                self._save()
                print(f"[记忆] 已删除: {f}")
                return 1
        # 3) 语义匹配
        try:
            q_vec = self.model.encode(fact_or_query)
            scores = cosine_similarity([q_vec], self.vecs)[0]
            best = scores.argsort()[::-1][0]
            if scores[best] > 0.4:
                self._remove_at(best)
                self._save()
                print(f"[记忆] 已删除: {self.facts[best]}")
                return 1
        except:
            pass
        return 0

    def _remove_at(self, idx):
        self.facts.pop(idx)
        self.vecs.pop(idx)
        self.access_count.pop(idx)
        self.last_access.pop(idx)

    def _prune(self):
        while len(self.facts) > self.MAX_FACTS:
            scores = []
            now = __import__("time").time()
            for i in range(len(self.facts)):
                count = self.access_count[i]
                age = now - self.last_access[i] if self.last_access[i] > 0 else 999999
                scores.append((count, -age, i))
            scores.sort()
            idx = scores[0][2]
            removed = self.facts[idx]
            self._remove_at(idx)
            print(f"[记忆] 自动遗忘: {removed}")

    def clear(self):
        self.facts.clear()
        self.vecs.clear()
        self.access_count.clear()
        self.last_access.clear()
        self._save()
