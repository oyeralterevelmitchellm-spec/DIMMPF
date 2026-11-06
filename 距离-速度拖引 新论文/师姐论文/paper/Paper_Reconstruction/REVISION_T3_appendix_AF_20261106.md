# REVISION · T3 附录窄改 + 消化 A–F · 2026-11-06

**Authority:** `D-t3-appendix-writing-open-20261106`（主人「进行」经副总控；Prior `D-t3-tuple-t1-appendix` + `D-fusion-AF-ACCEPT`）

## 改了哪些段
1. **主文** `Paper_Reconstruction/主文.tex`
   - §`sec:tuple` 末：新增 **位阶声明（T3）** — `tuple_t1` = 附录级；主配置仍冻结 B。
   - §`sec:fusion-ablation` 末：新增 **FE 族 A–F** 短段（A/B/D′/E/F1 数字；C 仅程序注；禁 C.`ref_ok`）。
2. **补充材料** `Paper_Reconstruction/补充材料.tex`
   - 新 `\subsection{…tuple_t1…}` → `sup:tuple-t1-appendix`
   - 新 `\subsection{…A--F…}` + `tab:fusion-af` → `sup:fusion-af`（插在 `sup:tlb-boundary` 前）

## 主张边界（一句）
默认主算法仍为冻结 B（diag+γ=0.8+γ_v=0+pred）；`tuple_t1` 与 F1「T1 有贡献」仅附录；FE 族可引用改锁增益与关 T1 掉点，**不**升主表、**不**写臂 C `ref_ok`。

## 未触碰的禁项（确认）
- 未写臂 C `ref_ok` 数值主张；未把 C.`gamma_v` CSV=0.5 当真跑 0.5  
- 未写 index0 救回；未 0.5-as-main；未 paper_promote  
- 未升 `tuple_t1` 为主算法/主表；未开新机时；未碰真·元组×改锁 / F2  
- **未 commit / 未 push**（待总控 ACCEPT 另令）

## Fingerprints
| 文件 | sha256 |
|---|---|
| 主文.tex（改后） | `f602ec35f0f3ef0ee4ef75dbf73db1590e14510b8047a642020b3ef104c09cb7` |
| 补充材料.tex（改后） | `80879c23ad49e9852bee7e03bebb52e42278f9eac548c6d68f4afd620128f0b2` |
| bak | `*.tex.bak-t3-appendix-20261106` |

## 证据钉
`FUSION_AF_ACCEPT_20261106.md`；`T3_tuple_t1_ruling_brief_20261106.md`；`T3_tuple_t1_APPENDIX_ACCEPT_20261106.md`；run `fusion_AF_20261106T025032`。

## 编译
本机 TeX Live 2025 `xelatex`：主文.pdf 52 页、补充材料.pdf 53 页，RC=0；`sup:tuple-t1-appendix` / `sup:fusion-af` / `tab:fusion-af` 经 xr-hyper 交叉引用无 undefined。**未 commit / 未 push**（工作树 `M` 两 tex）。

