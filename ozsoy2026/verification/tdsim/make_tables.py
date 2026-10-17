"""Print the markdown tables used in README.md from the check JSON files."""
import json
import math
import common

d3 = json.load(open(common.out("check3_stability.json")))
print("### check3 (ratios to a_zoa; D1 unless noted)\n")
print("| rpm | g | a_zoa mm | SD/ZOA-1 | SD(no helix)/ZOA-1 | ZOA+dt/2 (512)/ZOA-1 | TD512 n300 | TD512 n1000 | "
      "TD2048 n1000 | TD8192 n300 | D2 TD512 n300 | D2 TD2048 n300 | fc ZOA / TD (Hz) |")
print("|" + "---|" * 13)
for r in d3["cases"]:
    z = r["a_zoa"]
    f = lambda k: ("%+.3f" % (r[k] / z - 1)) if (r[k] is not None and not
                                                 (isinstance(r[k], float) and math.isnan(r[k]))) else "-"
    print(f"| {r['rpm']} | {r['g']:.0f} | {z*1e3:.3f} | {f('a_sd')} | {f('a_sd_nohelix')} | {f('a_zoh512')} | "
          f"{f('a_td512_n300')} | {f('a_td512_n1000')} | {f('a_td2048_n1000')} | {f('a_td8192_n300')} | "
          f"{f('a_tdD2_512_n300')} | {f('a_tdD2_2048_n300')} | {r['fc_zoa']:.1f} / {r['fc_td512']:.1f} |")
print("\nsummary:", json.dumps(d3["summary"], indent=1))

d3b = json.load(open(common.out("check3b_exit_chip.json")))
print("\n### check3b (exit pi - 0.35)\n")
print("| rpm | g | ZOA mm | SD mm | TD512 D1 | TD512 D2 | ZOA+dt/2 512 | TD2048 D1 | TD2048 D2 | ZOA+dt/2 2048 | "
      "Richardson TD / SD - 1 |")
print("|" + "---|" * 11)
for r in d3b["rows"]:
    ext = r["a_tdD1_2048"] + (r["a_tdD1_2048"] - r["a_tdD1_512"]) / 3
    print(f"| {r['rpm']} | {r['g']:.0f} | {r['a_zoa']*1e3:.3f} | {r['a_sd']*1e3:.3f} | {r['a_tdD1_512']*1e3:.3f} | "
          f"{r['a_tdD2_512']*1e3:.3f} | {r['a_zoh512']*1e3:.3f} | {r['a_tdD1_2048']*1e3:.3f} | "
          f"{r['a_tdD2_2048']*1e3:.3f} | {r['a_zoh2048']*1e3:.3f} | {ext / r['a_sd'] - 1:+.4f} |")
print("\nsummary:", json.dumps(d3b["summary"], indent=1))
