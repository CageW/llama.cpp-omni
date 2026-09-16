"""多 session 实时双工压测工具。

模块分工:

- arrival  按分布生成会话到达时刻(什么时候来)
- activity 会话内的说话/静音/抢话节奏(来了之后怎么用)
- client   单个会话的协议生命周期与观测
- metrics  逐会话指标与播放缓冲模型
- runner   投放、汇总、报告
- fake_server 不需要 GPU 的假服务端,用于工具自检
"""

__all__ = ["arrival", "activity", "client", "metrics", "runner", "fake_server"]
