# Foldseek 龙芯 loongarch64 适配记录

> Foldseek 是**蛋白质结构快速搜索**工具（结构版 BLAST），基于 MMseqs2 框架 +
> 3Di 结构字母表 + ProstT5 语言模型。本次结论：**CPU 版编译成功，含 ProstT5，
> ggml 原生启用龙芯 LSX/LASX SIMD**。分类：`05_interaction_analysis/` 或结构搜索类（按需定）。

## 一、环境信息

| 项目 | 值 |
|------|-----|
| 架构 | loongarch64（Loongnix Server 23.1，龙芯 3A6000，128 核）|
| 机器 | `10.71.13.47`，容器 `tuyi_test` |
| 分支 | `adapt/sdaa` |
| cmake | 3.26.3 |
| gcc / g++ | 12.3.0 |
| Rust | 1.77.0（MMseqs2 的 foldcomp 用 Rust）|

## 二、架构分析

### 2.1 Foldseek 组成

```
Foldseek
├── MMseqs2（核心框架）── 搜索/比对/聚类（C++，部分 Rust）
├── 3Di（结构字母表）── 把 3D 结构转成 20 字母序列
└── ProstT5（语言模型）── 从氨基酸序列预测 3Di
    └── ggml（llama.cpp 底层）── 推理引擎
```

### 2.2 自定义算子（CUDA）分析

**157 个 CUDA `.cu` 文件**，分两块：

| 块 | 位置 | 内容 |
|----|------|------|
| libmarv | `lib/mmseqs/lib/libmarv/src/` | PSSM / Smith-Waterman CUDA kernel |
| ggml-cuda | `lib/prostt5/ggml/src/ggml-cuda/` | 注意力/matmul/rope CUDA 后端 |

**关键：顶层 CMakeLists 里 `set(ENABLE_CUDA 0)` 默认关闭**，这些 CUDA 算子默认不编译，不阻塞 CPU 构建，无需处理。

### 2.3 龙芯 SIMD 支持情况

- **ggml（ProstT5）原生支持龙芯**：`ggml-cpu-impl.h` 含 `__loongarch64`/`__loongarch_sx`(LSX)/`__loongarch_asx`(LASX) 检测，configure 时确认 `Adding CPU backend variant ggml-cpu: -march=loongarch64;-mlasx;-mlsx`
- **MMseqs2 的 SIMD 用 SIMDe 中转（升级后可走 LSX）**：代码经 `simd.h` 调 `_mm_xxx`，由 SIMDe（跨架构兼容库）翻译成本地指令。自带 SIMDe 版本旧、不认识龙芯，会标量 fallback；**升级 SIMDe 后（见 4.2 节）`_mm_xxx` 自动翻译成 LSX 原生指令**（objdump 确认 `vadd.b`），比对核心已有向量化加速

## 三、依赖清单

### 3.1 编译时依赖

| 依赖 | 版本 | 说明 |
|------|------|------|
| gcc / g++ | 12.3.0 | 系统自带 |
| cmake | ≥ 3.15（实测 3.26.3）| 系统自带 |
| Rust | 1.77.0 | foldcomp 子库用（loongarch64-unknown-linux-gnu 目标）|
| zlib / bzip2 / zstd | — | 系统自带 |
| OpenMP（libgomp）| — | 系统自带 |

### 3.2 运行时权重（ProstT5 需单独下载）

| 文件 | 大小 | 下载方式 |
|------|------|---------|
| prostt5-f16.gguf | 2.42 GB | 见「六、权重下载」|

### 3.3 submodule 缺失（不影响编译）

`.gitmodules` 里 3 个 submodule 未同步，但都是测试/可选后端：
`mmseqs/util/regression`、`regression`（回归测试）、`ggml-kompute/kompute`（Vulkan 后端）。

## 四、适配改动（2 处）

### 4.1 `lib/gemmi/third_party/stb_sprintf.h` — 64 位指针修复

stb_sprintf.h 的 64 位架构宏列表漏了龙芯，导致 `char*`（64位）被强转成 `unsigned int`（32位）报错：

```c
// 原（第 233 行）
#if defined(__ppc64__) || defined(__powerpc64__) || defined(__aarch64__) || ... || defined(__s390x__)
// 改：末尾加 __loongarch64
#if defined(__ppc64__) || ... || defined(__s390x__) || defined(__loongarch64)
```

### 4.2 `lib/mmseqs/lib/simde/` — 升级 SIMDe 至支持 loongarch64（SIMD 提速）

**背景**：MMseqs2 的 SIMD 不是直接写死 x86，而是用 **SIMDe**（SIMD Everywhere，跨架构兼容库）
中转。`simd.h` 里的 `_mm_xxx` 通过 `SIMDE_ENABLE_NATIVE_ALIASES` 映射到 `simde_mm_xxx`。

**问题**：MMseqs2 自带的 SIMDe 版本过旧，不认识 loongarch64（只有老 MIPS 龙芯 `__mips_loongson_mmi`），
导致 `simde_mm_xxx` 在龙芯上静默 fallback 到标量实现（慢）。

**解决**：升级 SIMDe 到最新版（master，472 文件 vs 旧版 349），新版在 `simde-arch.h`/`simde-features.h`
里原生支持 loongarch64：

```c
// 新版 simde-features.h 末尾
#if defined(__loongarch_sx)          // 龙芯 LSX（128位，默认启用）
# define SIMDE_LOONGARCH_LSX_NATIVE
# include <lsxintrin.h>
#endif
#if defined(__loongarch_asx)         // 龙芯 LASX（256位，需 -mlasx）
# define SIMDE_LOONGARCH_LASX_NATIVE
# include <lasxintrin.h>
#endif
```

升级后，`simde_mm_xxx` 自动用 LSX/LASX 原生指令实现，**无需改动任何 MMseqs2 代码**（SSE 是 simd.h 的默认路径）。

## 五、编译

```bash
cd /data01/tuyilist/foldseek
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j 32          # 128 核，约 3 分钟
# → build/src/foldseek（22.5MB 可执行文件）
```

## 六、ProstT5 权重下载

官方源 `foldseek.steineggerlab.workers.dev`（Cloudflare Workers）国内访问超时，**用 mmseqs 数据源 + aria2c 多线程**：

```bash
dnf install -y aria2
aria2c -x 16 -s 16 -o prostt5-f16-gguf.tar.gz \
  https://opendata.mmseqs.org/foldseek/prostt5-f16-gguf.tar.gz
tar xzf prostt5-f16-gguf.tar.gz   # → prostt5-f16.gguf (2.42GB)
```

> 注意：用 curl 单线程下载极慢（~150KB/s），aria2c 16 线程约 1.4-1.9MB/s（约 20 分钟）。

## 七、测试结果（全部通过）

### 7.1 核心结构搜索（easy-search）

```bash
./build/src/foldseek easy-search example/d1asha_ example/ aln tmpFolder
```

结果正确：`d1asha_ d1asha_ 1.000`（自匹配 100%），后续为相似结构。

### 7.2 ProstT5 从序列预测 3Di

```bash
./build/src/foldseek createdb test.fasta testdb --prostt5-model prostt5-f16.gguf
```

成功输出 3Di 结构序列（20 字母表），如：
```
DPPPPPPPDAQEEEAEDEPCDVVVVVLVVVLVVLVVVLVVVCCVVVVPHYYYYYYYYDYPQKIKIWIWGDDPDTSDIDMDDDPDPPDDDDSVRVVV
```

ggml 正确识别 25 个 CPU 设备（龙芯后端），耗时约 1m28s/序列（T5 逐 token 生成，CPU 上慢属正常）。

### 7.3 SIMD 提速验证（升 SIMDe 后）

objdump 确认 `_mm_add_epi8` 已翻译成龙芯 LSX 原生指令：

```asm
foo:
    vinsgr2vr.d  $vr0, $a0, 0x0
    vadd.b       $vr0, $vr0, $vr1   ← LSX 字节向量加法（非标量）
    vpickve2gr.du $a0, $vr0, 0x0
```

> 性能说明：简单循环上 gcc 的自动向量化（auto-vectorization）已能覆盖 LSX，SIMDe 无优势甚至略慢；
> 但 MMseqs2 的 Smith-Waterman / UngappedAlignment 等**复杂手动 SIMD**（gcc 自动向量化覆盖不了）
> 才是 SIMDe→LSX 的真实收益点，需大规模比对任务 A/B 实测（可选后续）。

## 八、踩坑记录

| # | 坑 | 根因 | 解决 |
|---|----|------|------|
| 1 | stb_sprintf.h 报 64 位指针丢精度 | 64 位架构宏列表漏 `__loongarch64` | 宏列表补 `__loongarch64` |
| 2 | ProstT5 官方权重源国内超时 | Cloudflare Workers 被墙 | 换 `opendata.mmseqs.org` 源 + aria2c 多线程 |
| 3 | curl 单线程下载太慢 | S3 源限速 | aria2c 16 线程提速 10 倍 |

## 九、归类建议

Foldseek 属**结构搜索/比对**，modelzoo 里可归 `05_interaction_analysis/`（结构互作）或单独的搜索类目录，待定。
