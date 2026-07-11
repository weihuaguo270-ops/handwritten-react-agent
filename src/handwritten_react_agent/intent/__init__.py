"""intent — 任务意图自动分类

自动判断用户输入属于：
  - functional_test:  测试工具/功能是否正常 → 直接返回用户
  - generative_task:  复杂生成式任务 → 进入 Eval Loop
"""
