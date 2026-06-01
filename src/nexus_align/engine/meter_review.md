# `meter.py` 代码评审

对 `WindowMeter` 的评审与改进建议。整体评价:**能用、可读性尚可**,但有几处**潜在 bug**(尤其子串过滤)和一个**结构性坏味道**(`self.meters` 异构字典),修掉后会明显更稳、更好维护。

按严重程度从高到低。

---

## 🔴 正确性 / 潜在 bug

### 1. 用子串判断来排除 timing 指标,很脆(`latest_exp_info` L78-84、`info` L302-308)
```python
and "step" not in k
and "epoch" not in k
```
任何**名字里含 `step`/`epoch` 的用户指标**都会被静默排除。比如 `add_new_meter("kl_per_step")`、`reward_epoch_avg`,就再也不会出现在 `latest_exp_info()` 和 `info()` 里 —— 而且不报错,极难发现。

**建议**:像 `hardware_meters` 那样用**显式集合**标记 timing 指标,不要靠子串:
```python
self.timing_meters = {"epoch", "step"}
...
and k not in self.hardware_meters
and k not in self.timing_meters
```

### 2. 同一段过滤逻辑重复两遍(`latest_exp_info` 与 `info` 的 exp 段)
两处的判定谓词完全一样(`isinstance(dict)` + 排除 hardware + 排除 timing)。一旦规则要改(比如上面的第 1 点),很容易只改一处。

**建议**:抽一个内部生成器,两边复用:
```python
def _experiment_meters(self):
    """Yield (name, meter_dict) for user experiment metrics."""
    for k, v in self.meters.items():
        if (isinstance(v, dict)
                and k not in self.hardware_meters
                and k not in self.timing_meters):
            yield k, v
```

### 3. `nvmlInit()` 没有对应的 `nvmlShutdown()`(L32)
`__init__` 里 `pynvml.nvmlInit()`,但 `stop_hardware_monitoring` 只停线程、不关 NVML → handle 泄漏。对长生命周期 / 反复创建实例的场景有影响。

**建议**:在 `stop_hardware_monitoring()` 里补 `pynvml.nvmlShutdown()`,或实现 `__enter__/__exit__` 当上下文管理器用。

### 4. `update_mean` 对空 deque 会 `ZeroDivisionError`(L63-66)
```python
self.meters[meter]["mean"] = sum(data_list) / len(data_list)
```
目前只被 `update()` 在 append 之后调用(非空,安全),但它是 public 方法,单独调用即崩。

**建议**:加一行保护 `if not data_list: return`,或别暴露成 public。

---

## 🟠 设计 / 结构

### 5. `self.meters` 是异构字典 —— 最大的坏味道
现在一个 dict 里混了三种东西:
- 纯标量计数器:`exp_start_time`(None/float)、`total_step`(int)、`current_train_steps`(int)
- 标准 meter:`{data, mean, report_mean, decimal}`
- 特殊 meter:`epoch`/`step` 还额外塞了 `num`、`start_time`(见 `add_epoch_step` L198-204)

这正是导致**到处 `isinstance(v, dict)`**、**子串排除**、**`epoch/step` 形状与别人不同**的根因。

**建议**(较大改动,值得做):
- 用一个轻量 `dataclass Meter`(`data: deque`、`mean`、`decimal`、`report_mean`)表示窗口指标;
- 计数器 `total_step` / `current_train_steps` / `exp_start_time` 提升为**普通实例属性**,不要塞进 `meters`;
- timing 的 `num` / `start_time` 单独管理,别挂在 meter 上。

这样 `isinstance` 判断、子串排除全都消失。

### 6. `info()` 硬件段四块 f-string 高度重复(L266-295)
GPU mem / GPU util / CPU mem / CPU util 四段几乎一模一样,~30 行。

**建议**:做成数据驱动 —— 用一张小表描述「标签、used 指标、peak 指标、可选 total、单位」,循环生成:
```python
specs = [
    ("gpu_mem", "gpu_mem_used", "gpu_mem_peak", "gpu_mem_total", "G"),
    ("gpu_uti", "gpu_util", "gpu_util_peak", None, "%"),
    ...
]
```

### 7. `mean` 字段类型不一致(初始 `"N/A"` 字符串,之后变 float)
`info()`(L316-323)只能靠 try/except 把初始的 `"N/A"` 格式化失败再兜底成 `(N/A)`,属于「用异常处理控制流」。

**建议**:`mean` 初始化为 `None`,格式化前判 `is None`,语义更清晰、也不靠抛异常。

---

## 🟡 健壮性 / 可移植性

### 8. `start`/`end` 用 `assert` 校验参数(L214、L222)
`python -O` 下 assert 会被剥离,校验失效。

**建议**:改成显式 `raise ValueError("meter must be 'step' or 'epoch'")`。

### 9. 顶层无条件 `import pynvml, psutil`(L9-10),即使 `hardware=False` 也强依赖
不想做硬件监控的人(比如 CPU-only 单测)也被迫装这两个包。

**建议**:把 `import pynvml/psutil` 挪进 `hardware=True` 的分支内做**惰性导入**。

### 10. hardware 相关属性只在 `hardware=True` 时创建(L29-42)
`device_id` / `nvml_handle` / `_monitor_thread` / `_stop_monitoring` 在 `hardware=False` 时根本不存在,别处一旦访问就 `AttributeError`。

**建议**:在 `__init__` 顶部对所有路径统一初始化为 `None`。

### 11. 后台线程与主线程的读写竞争(轻微)
监控线程 `update()` 写硬件 meter,主线程 `info()` 读。靠 GIL 基本安全(最多读到稍旧的值),但**前提是监控启动后不要再增删 `self.meters` 的键**。

**建议**:文档里写明「`add_new_meter` 等结构性修改要在 `start_hardware_monitoring` 之前完成」;若确需运行时增减,加一把 `threading.Lock`。

---

## 🟢 命名 / 文档(小)

### 12. `latest_exp_info` → 建议 `latest_metrics`
和我们前面统一的术语一致:loss/grad_norm/reward 这些是 **metric**。`info(exp_info=True)` 的参数也可改 `show_metrics`。`exp_info` 这个词含糊。
> 注意:`tracker.py` 已不再调用它(改吃 plain dict),所以这个改名**不会牵连** tracker;但若以后别处调用,记得同步。

### 13. `total_step` vs `current_train_steps` 语义不清(L189-194、L228-229)
两者在 `end("step")` 里**同步 +1**,看起来完全重复。如果设计意图是「`total_step` 跨断点续训累计、`current_train_steps` 本次进程重置」,请在 docstring 写明;否则考虑删一个。

### 14. 部分 docstring 偏循环/空洞
`_get_val` "Get the value to print."、`update_train_state` "Update the train state." 没说清输入输出。`update_train_state` 可注明读取的键是 `epoch/step/total_step`。

### 15. `psutil.cpu_percent(interval=None)` 首次调用返回 0.0(L150)
第一次采样是无意义的 0(它测的是「距上次调用」的占用)。可接受,但可在 docstring 提一句,或启动时先 warm-up 调一次丢弃。

---

## 建议优先级清单

| 优先级 | 项 | 工作量 |
|---|---|---|
| **P0 先修** | #1 子串过滤(真隐患)、#4 空 deque 崩溃 | 小 |
| **P1** | #2 抽公共过滤、#3 nvmlShutdown、#8 assert→raise | 小 |
| **P2** | #5 异构字典重构(根因)、#6 硬件段数据驱动、#7 mean 类型 | 中~大 |
| **P3** | #9 惰性导入、#10 属性初始化、#11 线程文档、#12 改名 | 小~中 |
| **P3** | #13 澄清 total_step 语义、#14/#15 文档 | 小 |

P0/P1 都是低成本高收益,建议先做;P2 的异构字典重构是改善可维护性的关键,但改动面大,可单独排期。
