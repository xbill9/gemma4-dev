"""ARITHMETIC for the 3050 OEM article. Inputs are quoted from the evidence files beside this one."""
card = 179.97          # price-snapshots.txt
kwh = 0.182            # price-snapshots.txt (EIA STEO 2026)
peak_w = 110.93        # concurrency-summary.txt, max power over the whole sweep
single_w = 80.11       # cells.txt, fa1-f16-t6 end-of-cell power (serving config, single stream)
g4dn_od, g4dn_spot, v5e_spot = 0.526, 0.3559, 0.5779

def per_m(tps, watts):
    kwh_per_m = watts / tps * 1e6 / 3.6e6
    return kwh_per_m, kwh_per_m * kwh

rows = [
 ("3050 c=1 128/512", 104.74, single_w),
 ("3050 c=64 128/512", 618.12, peak_w),
 ("3050 c=64 512/128", 358.05, peak_w),
 ("3050 c=8 512/128", 224.18, peak_w),
]
for name, tps, w in rows:
    k, d = per_m(tps, w)
    print(f"{name}: {tps} tok/s at {w} W -> {k:.3f} kWh/M, ${d:.3f}/M electricity")
print(f"break-even card vs g4dn on-demand: {card/g4dn_od:.0f} h = {card/g4dn_od/24:.1f} days")
print(f"break-even card vs g4dn spot: {card/g4dn_spot:.0f} h = {card/g4dn_spot/24:.1f} days")
print(f"break-even card vs v5e-1 spot: {card/v5e_spot:.0f} h = {card/v5e_spot/24:.1f} days")
print(f"electricity per hour at peak: ${peak_w/1000*kwh:.4f}")
print(f"g4dn c=1 cost per M on-demand: ${g4dn_od/(42.36*3600)*1e6:.3f}; c=8 ${g4dn_od/(242.47*3600)*1e6:.3f}")
print(f"v5e-1 c=1 spot per M: ${v5e_spot/(120*3600)*1e6:.3f}; c=64 ${v5e_spot/(1382*3600)*1e6:.3f}")
print(f"3050 / g4dn c=1 512/128: {93.28/42.36:.2f}x; c=8: {224.18/242.47:.2f}x")
print(f"3050 c=1 128/512 vs v5e-1 c=1 ctx128: {104.74/120:.2f}x; peak 618.12 vs 1382: {618.12/1382:.2f}x")
cpu_tg = [19.02, 19.14, 19.03, 18.17]
print(f"i7-10750H v2 tg128 -t6 mean: {sum(cpu_tg)/4:.2f}")
print(f"3050 tg128 116.14 / cpu {sum(cpu_tg)/4:.2f} = {116.14/(sum(cpu_tg)/4):.1f}x")
print(f"3050 tg / 1650Ti tg (81.59-81.92 mean {(81.65+81.82+81.92+81.59)/4:.2f}): {116.14/((81.65+81.82+81.92+81.59)/4):.2f}x")
print(f"3050 / sdm850 4 gold: {116.14/5.00:.1f}x; / sc7180p 2 gold {116.14/8.33:.1f}x")
print(f"tok/s per W single stream: 3050 {116.14/single_w:.2f}; 1650Ti 76.14..77.14 at 40.43 W -> {76.14/40.43:.2f}..{77.14/40.43:.2f}")
print(f"E4B/E2B tg: {60.41/116.02:.3f}; E4B VRAM 6430/8192 = {6430/8192:.0%}; E2B 3610/8192 = {3610/8192:.0%}")
print(f"1650Ti E4B tg ~39.8 vs 3050 E4B 60.41: {60.41/39.8:.2f}x")
print(f"VRAM headroom at 131k: E2B {8192-3610} MiB free; E4B {8192-6430} MiB free")
print(f"1650 Ti KL divergence Google GGUF / exact v2: 0.054264 / 0.001680 = {0.054264/0.001680:.0f}x")
print(f"flash attention off vs on: prompt {3408.19/3859.96:.3f}x, generate {108.46/116.14:.3f}x")
for n, v in [("1650 Ti", 81.75), ("i7-10750H", 18.84), ("SC7180P", 8.33), ("SDM850", 5.00)]:
    print(f"{n} tg / 3050 tg: {v/116.14:.2f}x")
