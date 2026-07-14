"""
Memory — 独立的语义记忆模块
============================
有 `[rag]` 依赖时：基于 BGE-small-zh-v1.5 的语义记忆。
无依赖时：降级为关键词匹配（保证核心 Agent 可安装、可跑）。

支持：
- 增: add() / auto_extract()
- 删: remove() / clear()
- 查: query()
- 自动遗忘: _prune()

注意：BGE 模型采用懒加载，首次写/查记忆时才会加载。
"""
import json
import time

try:
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    _HAS_VECTOR = True
except ImportError:  # pragma: no cover - exercised when installed without [rag]
    np = None
    cosine_similarity = None
    _HAS_VECTOR = False


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
        self._model = None  # 懒加载：首次使用时才加载
        self._load()

    def _semantic_ready(self) -> bool:
        return _HAS_VECTOR

    def _get_model(self):
        """首次访问时加载 BGE 模型，后续复用"""
        if not _HAS_VECTOR:
            raise ImportError(
                '语义记忆需要: pip install -e ".[rag]"'
            )
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            print("  [记忆] 加载语义模型...")
            self._model = SentenceTransformer('BAAI/bge-small-zh-v1.5')
        return self._model

    def _load(self):
        try:
            with open(self.save_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.facts = data.get("facts", [])
            raw_vecs = data.get("vecs", [])
            if _HAS_VECTOR and np is not None:
                self.vecs = [np.array(v) for v in raw_vecs]
            else:
                self.vecs = raw_vecs
            self.access_count = data.get("access_count", [0] * len(self.facts))
            self.last_access = data.get("last_access", [0] * len(self.facts))
            if self.facts:
                mode = "语义" if _HAS_VECTOR else "关键词降级"
                print(f"[记忆] 已加载 {len(self.facts)} 条记忆（{mode}）")
        except Exception:
            self.facts = []
            self.vecs = []
            self.access_count = []
            self.last_access = []

    def _save(self):
        with open(self.save_path, "w", encoding="utf-8") as f:
            serial_vecs = []
            for v in self.vecs:
                if _HAS_VECTOR and np is not None and hasattr(v, "__iter__"):
                    serial_vecs.append([round(float(x), 4) for x in v])
                else:
                    serial_vecs.append(v if isinstance(v, list) else [])
            json.dump({
                "facts": self.facts,
                "vecs": serial_vecs,
                "access_count": self.access_count,
                "last_access": self.last_access,
            }, f, ensure_ascii=False, separators=(",", ":"))

    def add(self, fact):
        if fact not in self.facts:
            self.facts.append(fact)
            if self._semantic_ready():
                try:
                    vec = self._get_model().encode(fact)
                    self.vecs.append(vec)
                except ImportError:
                    self.vecs.append([])
            else:
                self.vecs.append([])
            self.access_count.append(0)
            self.last_access.append(0)
            self._prune()
            self._save()
            return True
        return False

    _EXACT_MATCH = 0.85
    _CONFLICT = 0.60

    def add_or_update(self, new_fact):
        if not new_fact.strip():
            return ("skipped", "空内容")
        if not self._semantic_ready():
            if new_fact in self.facts:
                return ("skipped", "与已有记忆重复")
            self.facts.append(new_fact)
            self.vecs.append([])
            self.access_count.append(0)
            self.last_access.append(0)
            self._prune()
            self._save()
            return ("added", None)
        new_vec = self._get_model().encode(new_fact)
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
            self.last_access[best_idx] = time.time()
            self._save()
            return ("updated", old_fact)
        self.facts.append(new_fact)
        self.vecs.append(new_vec)
        self.access_count.append(0)
        self.last_access.append(0)
        self._prune()
        self._save()
        return ("added", None)

    def _keyword_query(self, question, top_k=3):
        tokens = [t for t in question.lower().replace("？", " ").split() if len(t) > 1]
        scored = []
        for f in self.facts:
            fl = f.lower()
            hit = sum(1 for t in tokens if t in fl) if tokens else (1 if question in f else 0)
            if hit or (question and question in f):
                scored.append((hit, f))
        scored.sort(key=lambda x: -x[0])
        return [{"fact": f, "score": float(s)} for s, f in scored[:top_k]]

    def query(self, question, top_k=3):
        if not self.facts:
            return []
        if not self._semantic_ready():
            return self._keyword_query(question, top_k)
        try:
            q_vec = self._get_model().encode(question)
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
            return self._keyword_query(question, top_k)

    def remove(self, fact_or_query):
        if fact_or_query in self.facts:
            self._remove_at(self.facts.index(fact_or_query))
            self._save()
            return 1
        for i, f in enumerate(self.facts):
            if fact_or_query in f:
                self._remove_at(i)
                self._save()
                print(f"[记忆] 已删除: {f}")
                return 1
        if not self._semantic_ready():
            return 0
        try:
            q_vec = self._get_model().encode(fact_or_query)
            scores = cosine_similarity([q_vec], self.vecs)[0]
            best = scores.argsort()[::-1][0]
            if scores[best] > 0.4:
                removed = self.facts[best]
                self._remove_at(best)
                self._save()
                print(f"[记忆] 已删除: {removed}")
                return 1
        except Exception:
            pass
        return 0

    def _remove_at(self, idx):
        self.facts.pop(idx)
        if idx < len(self.vecs):
            self.vecs.pop(idx)
        self.access_count.pop(idx)
        self.last_access.pop(idx)

    def _prune(self):
        while len(self.facts) > self.MAX_FACTS:
            scores = []
            now = time.time()
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
