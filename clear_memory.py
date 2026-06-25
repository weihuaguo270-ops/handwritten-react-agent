import json
import os
import sys
sys.path.insert(0, r"D:\agent_learning\repo")

# 直接调用 Memory 类的 clear 方法
memory_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "memory.json")

# 清空记忆
with open(memory_path, "w", encoding="utf-8") as f:
    json.dump({"facts": [], "vecs": [], "access_count": [], "last_access": []}, f, ensure_ascii=False, separators=(",", ":"))

print("记忆已清空！")
print(f"文件路径: {memory_path}")
if os.path.exists(memory_path):
    print(f"文件大小: {os.path.getsize(memory_path)} 字节")
else:
    print("文件不存在")
