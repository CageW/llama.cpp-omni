"""什么时候有新的 session 连上来。

压测里最容易失真的一个环节:如果所有 session 在同一秒连上,那么「并发 N」在开头那一段
其实是「N 个会话同时抢冷启动」,和稳态下的「N 个会话各自在第几十秒的某个时刻」完全不是
一件事。真实流量里会话是在一段窗口内陆续到来的,到达模式直接决定瞬时并发怎么爬升、
峰值有多尖。这个模块只负责「第几个 session 在第几秒上线」,不关心它上线之后干什么。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ArrivalPlan:
    """一次压测的完整投放计划。

    offsets 是从压测开始算起每个 session 的上线时刻(秒),按时间升序;
    target_concurrency 是本次计划打算维持的并发数(会话数上限),仅用于报告;
    它会和 duration 一起决定自然到达该在什么时候停止投放。
    """

    offsets: List[float]
    target_concurrency: int
    duration_s: float
    mode: str
    seed: Optional[int] = None
    notes: str = ""

    @property
    def n_sessions(self) -> int:
        return len(self.offsets)


def _poisson_offsets(rate: float, until: float, rng: random.Random) -> List[float]:
    """泊松到达:相邻两次到达的间隔独立同分布,服从均值 1/rate 的指数分布。

    这是排队论里的标准模型(M/M/c),也是 vLLM 压测 `--request-rate` 的默认口径。
    它给出的瞬时并发会有明显起伏,更接近真实通话随机接入的样子。
    """
    out: List[float] = []
    t = 0.0
    while True:
        t += rng.expovariate(rate)
        if t > until:
            return out
        out.append(t)


def _uniform_ramp_offsets(count: int, ramp_s: float, start: float, rng: random.Random) -> List[float]:
    """阶梯式(斜坡)到达:把 count 个 session 均匀铺在 ramp_s 秒里。

    这是闭环压测工具(Locust、k6 的 ramping VUs)的做法:并发是「我要它等于几」
    而不是「到达率是几」。爬升期确定、没有随机抖动,适合做可比的回归基线。
    """
    if count <= 0:
        return []
    if ramp_s <= 0:
        return [start] * count
    step = ramp_s / count
    return [start + i * step for i in range(count)]


def _burst_offsets(count: int, ramp_s: float, start: float, rng: random.Random) -> List[float]:
    """突发到达:同一时刻成批上线,批内间隔极小。

    用来复现实时语音里最常见的尖峰——推送通知、整点活动、开会时一群人同时进。
    它和斜坡到达的差别在峰值:同样的会话总数,突发会在开头打出远高于目标的瞬时并发。
    """
    if count <= 0:
        return []
    batch = max(1, int(count * 0.5))
    out: List[float] = []
    for i in range(count):
        if i < batch:
            out.append(start + rng.uniform(0.0, max(ramp_s * 0.02, 0.05)))
        else:
            out.append(start + rng.uniform(ramp_s * 0.5, ramp_s))
    return sorted(out)


def _trickle_offsets(count: int, ramp_s: float, start: float, rng: random.Random) -> List[float]:
    """零散到达:每个 session 在窗口内均匀随机取一个时刻。

    比斜坡更松、比泊松更平,适合观察「会话互相错开」时的稳态成本。
    """
    if count <= 0:
        return []
    return sorted(start + rng.uniform(0.0, ramp_s) for _ in range(count))


def expected_sessions(
    *,
    target_concurrency: int,
    duration_s: float,
    mean_lifetime_s: float,
    retry_mean_s: float,
    abandon_probability: float,
) -> int:
    """补位模式下大概会投出多少会话,只用于事前提示和结果对照。

    口径是稳态的直白推算:不再重连的比例是 abandon_probability,所以平均一个会话
    槽位要经历 1/(1-abandon) 次会话才彻底空出来;每个会话占用的时间是一个平均时长
    加上一次补位间隔。乘上目标并发与时长就是总量。这是个粗略估计,它的用途只是让
    人知道这次会投出几十个还是几百个会话,不是精确计划。
    """
    if target_concurrency <= 0 or mean_lifetime_s <= 0 or abandon_probability >= 1.0:
        return 0
    per_session_s = mean_lifetime_s + retry_mean_s
    cycles = duration_s / max(per_session_s, 1e-6)
    return int(target_concurrency * cycles / max(1e-6, 1.0 - abandon_probability))


def check_load_feasibility(
    *,
    mode: str,
    target_concurrency: int,
    duration_s: float,
    ramp_s: Optional[float],
    mean_lifetime_s: float,
    reuse: bool,
    n_planned_sessions: int,
) -> Optional[str]:
    """判断这份配置能不能达到它声称的并发,不能就给出人话解释。

    判据是 Little 定律的直白形式:稳态并发 ≈ 到达率 × 平均停留时长。不补位时
    「到达率」= 计划会话数 / 到达窗口,算出来的稳态并发若明显低于目标值,这次压测
    压的就不是它标称的那个并发,报告里会给出一个好看的失败率,实际上压根没压到。
    """
    if reuse or mode == "all_at_once":
        return None
    window = ramp_s if ramp_s is not None else max(duration_s / 3.0, 1.0)
    if window <= 0 or mean_lifetime_s <= 0:
        return None
    arrival_rate = n_planned_sessions / window
    expected_concurrency = arrival_rate * mean_lifetime_s
    if expected_concurrency < target_concurrency * 0.8:
        suggested = int(max(1, target_concurrency * window / max(mean_lifetime_s, 1e-6)))
        return (
            f"这份配置达不到目标并发 {target_concurrency}。按 Little 定律,到达率 "
            f"{arrival_rate:.3f} 会话/秒 乘以平均会话时长 {mean_lifetime_s:.1f}s,稳态并发只有约 "
            f"{expected_concurrency:.1f}。要么加 --reuse 让会话退出后补位,要么把投放数提到约 "
            f"{suggested} 个,要么把到达窗口压到 {target_concurrency * mean_lifetime_s / max(n_planned_sessions, 1):.1f}s 以内。"
        )
    return None


def build_arrival_plan(
    *,
    mode: str = "poisson",
    target_concurrency: int = 8,
    duration_s: float = 300.0,
    ramp_s: Optional[float] = None,
    rate: Optional[float] = None,
    reuse: bool = False,
    seed: Optional[int] = None,
    mean_lifetime_s: float = 60.0,
    lifetime_dist: str = "lognormal",
    lifetime_min_s: float = 5.0,
    lifetime_max_s: float = 600.0,
    retry_mean_s: float = 3.0,
    abandon_probability: float = 0.0,
) -> ArrivalPlan:
    """按指定到达分布生成投放计划。

    Args:
        mode: poisson | ramp | burst | trickle | all_at_once
        target_concurrency: 稳态要维持的会话数。
        duration_s: 压测时长(秒)。到达窗口默认取 duration_s 的三分之一。
        ramp_s: 到达窗口长度(秒)。不补位时窗口决定投放节奏;补位时它只作用于爬升段。
        rate: 泊松到达率(会话/秒)。给定它就忽略 target_concurrency 推导出的默认值。
        reuse: 会话结束后是否补投新会话。True 时把这套工具当成持续加负载的发生器,
            稳态并发顶在 target_concurrency 附近;False 时只投一轮,并发爬到峰值后
            随会话各自结束而下降,而这个峰值由会话时长随机抽到什么决定,不稳。
        seed: 随机种子。给定后同一份配置可复现,回归对比才有意义。
        mean_lifetime_s / lifetime_dist / lifetime_min_s / lifetime_max_s:
            会话时长模型。补位模式下要靠它推算每个会话什么时候退出(从而什么时候补位),
            所以必须和真正跑的时候用的是同一套参数,否则计划里的会话数和实际投放的
            对不上。runner 会把 RunConfig 里的这几个值原样传进来。
        retry_mean_s: 补位模式下,会话退出后隔多久补上(指数分布均值)。
        abandon_probability: 补位模式下,用户退出后不再重连的概率。
    """
    rng = random.Random(seed)
    window = ramp_s if ramp_s is not None else max(duration_s / 3.0, 1.0)

    if mode == "all_at_once":
        # 开头就满并发,复现冷启动叠加的最坏情况。它看的是峰值,不是稳态,所以不补位。
        offsets: List[float] = [0.0] * target_concurrency
        return ArrivalPlan(
            offsets=sorted(offsets),
            target_concurrency=target_concurrency,
            duration_s=duration_s,
            mode=mode,
            seed=seed,
            notes="全部 session 同一时刻上线,复现冷启动叠加的最坏情况",
        )

    if reuse:
        # 补位模式下这个计划只覆盖爬升段:把会话铺在窗口内上线。之后「谁退出就补谁」
        # 由运行时看着实际退出事件做,不在计划阶段预演。原因是会话时长是随机抽的,
        # 计划阶段抽一次、运行阶段又抽一次,两次不可能重合,拿预演去驱动补位会让
        # 实际并发缓慢跑偏(实测会从 8 掉到 3)。
        #
        # 爬升人数要按「占空比」放大。补位不是瞬时的:会话平均跑 mean_lifetime_s,
        # 退出后再隔 retry_mean_s 才补上,所以一个槽位平均只有
        # mean/(mean+retry) 的时间真的有会话在跑。想让实际并发达到 target_concurrency,
        # 就要多开这么多槽位,否则实际并发会稳定地低于目标值(实测 8 只能到 5.7)。
        # 再乘上不再重连的损耗:每次退出有 abandon_probability 的概率永久空一个槽。
        duty = mean_lifetime_s / max(mean_lifetime_s + max(retry_mean_s, 0.0), 1e-6)
        duty *= max(1e-6, 1.0 - abandon_probability)
        oversubscribe = 1.0 / duty if duty > 0 else 1.0
        ramp_count = max(1, int(math.ceil(target_concurrency * oversubscribe)))
        offsets = _uniform_ramp_offsets(ramp_count, window, 0.0, rng)
        return ArrivalPlan(
            offsets=offsets,
            target_concurrency=target_concurrency,
            duration_s=duration_s,
            mode=f"{mode}+refill",
            seed=seed,
            notes=(
                f"爬升 {window:.1f}s 投出 {ramp_count} 个会话(目标并发 {target_concurrency} "
                f"除以占空比 {duty:.2f} 得到);之后会话退出即补位,平均间隔 {retry_mean_s:.1f}s"
                + (f",{abandon_probability:.0%} 的会话退出后不再重连" if abandon_probability else "")
                + ";实际投放总数运行时才会定"
            ),
        )

    if mode == "poisson":
        if rate is None:
            # 默认口径:到达窗口内平均投出 target_concurrency 个会话。
            rate = target_concurrency / window
        offsets = _poisson_offsets(rate, window, rng)
        return ArrivalPlan(
            offsets=offsets,
            target_concurrency=target_concurrency,
            duration_s=duration_s,
            mode=mode,
            seed=seed,
            notes=(
                f"泊松到达,平均间隔 {1.0 / rate:.3f}s({rate:.4f} 会话/秒);"
                f"不补位,实际峰值并发取决于抽到的会话时长"
            ),
        )

    if mode == "ramp":
        offsets = _uniform_ramp_offsets(target_concurrency, window, 0.0, rng)
    elif mode == "trickle":
        offsets = _trickle_offsets(target_concurrency, window, 0.0, rng)
    elif mode == "burst":
        offsets = _burst_offsets(target_concurrency, window, 0.0, rng)
    else:
        raise ValueError(f"未知到达模式: {mode}")

    return ArrivalPlan(
        offsets=offsets,
        target_concurrency=target_concurrency,
        duration_s=duration_s,
        mode=mode,
        seed=seed,
        notes=f"到达窗口 {window:.1f}s,共 {len(offsets)} 个会话,不补位",
    )


@dataclass
class SessionLifecycle:
    """单个 session 的时长。

    真实通话时长是右偏的:多数几十秒结束,少数拖到几分钟。用指数分布会低估长尾,
    用对数正态或韦布尔更接近实测。时长还受服务端硬上限约束(video 300s / audio 600s),
    超过上限的会话会被服务端以 timeout 关闭,这里按上限截断,免得把服务端的正常
    超时当成错误统计进去。
    """

    mean_s: float = 60.0
    dist: str = "lognormal"
    min_s: float = 5.0
    max_s: float = 600.0
    seed: Optional[int] = None
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def sample(self) -> float:
        if self.dist == "fixed":
            return min(max(self.mean_s, self.min_s), self.max_s)
        if self.dist == "exponential":
            # 指数分布:均值 mean_s,方差大,长尾比对数正态更重。
            value = self._rng.expovariate(1.0 / self.mean_s)
        elif self.dist == "weibull":
            # 韦布尔形状 1.5:比指数衰减得稍快,适合「通话很快结束」占多数的情形。
            scale = self.mean_s / math.gamma(1.0 + 1.0 / 1.5)
            value = scale * self._rng.weibullvariate(1.0, 1.5)
        else:
            # 对数正态:标准差取均值的 0.6 倍,给出一个中等偏重的长尾。
            sigma = 0.6
            mu = math.log(self.mean_s) - sigma * sigma / 2.0
            value = self._rng.lognormvariate(mu, sigma)
        return min(max(value, self.min_s), self.max_s)
