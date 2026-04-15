# I2_S 算子讲解

## 1. 数据布局（最关键）

I2_S 是 **2-bit 对称量化**，权重只有 3 个取值 `{-1, 0, +1}`，但在内存里用 2 位无符号编码 `{0, 1, 2}`（0 代表 -1，1 代表 0，2 代表 +1，见 `quantize_i2_s` line 60-66）。整行再共享一个 `float` scale。

一个 "block" = **128 个权重**（`QK_I2_S = 128`），压缩后占 **32 字节**（`QK_I2_S_PACKED_BYTES = 32`）。打包方式很特殊（line 75-83）：

```
byte i (i=0..31) 的 8 位 = [w_i | w_{i+32} | w_{i+64} | w_{i+96}]
                           bits:  7..6  5..4   3..2    1..0
```

也就是说一个字节的 4 个 2-bit 字段**不是 128 个连续位置的权重**，而是把 128 分成 4 个 32 元素组，byte i 同时打包了这 4 组里的第 i 个元素。这种交错打包让 SIMD 解包时只需对一个寄存器做 `>>6, >>4, >>2, &0x03` 就能一次性得到 4 组 32 个 2-bit 权重。

对应地，激活 y（int8）也按 128 为一块，布局成 `[y0..y31][y32..y63][y64..y95][y96..y127]`，刚好和解包出的 4 组对齐。

## 2. 核心思想

MAD（Multiply-ADd）路径的点积 = 展开 2-bit 权重成 int8 → 和 int8 激活做 dot product → 累加。
关键是 **uint8 向量可以并行解 4 路 2-bit**，所以一次 32B 加载就够出 128 个权重的 4 组 32 元素。

注意：权重解出来是 `{0,1,2}`，不是 `{-1,0,1}`。真正的有符号映射被推迟到模型层面（会在更上层通过减一个 bias/常数补回来），所以内核只做无符号×有符号点积即可。

## 3. 各架构实现对比

所有路径都长成同一个结构：`group32_num` 批次，每批 32 个 block 累加在 int16 域，再一次性 widen 到 int32，减少累加开销。

### AVX2（line 182-245）
- `_mm256_loadu_si256`：加载 32B 打包权重
- `_mm256_srli_epi16(x, 2/4/6)` + `_mm256_and_si256(mask=0x03)`：一次出 4 组 32 个 2-bit 权重
- `_mm256_maddubs_epi16(u8, i8)`：**核心指令**。把相邻两对 `u8*i8` 相乘并相加成 int16（pmaddubsw）。因为权重是 u8∈{0..2}、激活是 i8∈[-128,127]，乘积 ≤ 254，两个相加 ≤ 508，int16 安全。
- `_mm256_madd_epi16(., 1)`：再把相邻两个 int16 加成 int32，顺带 widen
- `hsum_i32_8`：最后横向求和
- **优化点**：用 `maddubs` 把"乘+加"合成一条指令，并且把 int16 累加延迟到外层才升位到 int32，省下大量 `madd`。

### ARM NEON（line 247-408）
- `vld1q_u8` / `vld1q_s8`：16B 加载（所以一次循环读两条 16B 得到 32B 打包）
- `vshrq_n_u8(x, 6/4/2)` + `vandq_u8(.,3)`：同样的移位解包
- 两条路径：
  - **有 `__ARM_FEATURE_DOTPROD`**：`vdotq_s32(acc, q, y)` —— 一条指令算 16 个 int8 乘积，4 个一组累加到 int32，直接跳过 int16 层。最快。
  - **无 dotprod 兜底**：`vmlal_s8(acc16, low/high)` —— 8×8→16 的 widening MAC，再 `vmovl_s16` 升位到 int32。
- 用 4 个 `accu_0..3` 累加器打破依赖链，帮助乱序发射。
- 结尾 `vaddlvq_s32` 做水平归约。

### RISC-V RVV（line 101-153）
- 先 `vsetvl_e8m2(32)` 请求 32B 宽度；若 VLEN 太小放不下就走纯 scalar fallback。
- `vle8_v_u8m2`：加载 32B 权重
- `vsrl_vx_u8m2(., 6/4/2)` + `vand_vx_u8m2(., 3)`：同样的解包套路
- **核心指令 `vwmacc_vv_i16m4(acc16, i8, i8)`**：widening multiply-accumulate，int8×int8→int16 并累加。这是 RVV 里对应 ARM `vmlal_s8` / x86 `pmaddubsw` 的东西。
- **优化点**（注释 line 99-100）：严格做过误差分析 —— int16 lane 每 block 最多累加 1024，**31 个 block** 后必达上界 31744 仍 < 32767，所以可以把 31 个 block 的 MAC 压在 int16 累加器里，**每 31 个 block 才 `vwadd_wv_i32m8` 升一次位**，大幅减少升位开销。和 AVX2/NEON 的策略对齐。
- 最后 `vredsum_vs_i32m8_i32m1` 水平归约。

## 4. 通用优化总结

1. **交错式打包**让解包退化为"一次加载 + 三次 shift+and"，没有 shuffle/gather。
2. **三层累加器**：int8 乘 → int16 batch 累加 → int32 升位 → int32 横向归约。每层都贴着溢出上界最大化停留时间。
3. **多累加器**（NEON 的 `accu_0..3`，AVX 的 `accu`/`accu32` 分离）打破依赖链。
4. **指令层面最划算的一条**：AVX `pmaddubsw`、ARM `sdot`（有 dotprod 时）、RVV `vwmacc` —— 都是 "乘完就同级累加/升位" 的复合指令。
5. **tail 走标量**（`ggml_i2_s_block_dot_scalar`），不给 SIMD 分支加负担。
