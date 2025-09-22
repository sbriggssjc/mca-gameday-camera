# MCA Gameday Cleanup Report

Generated on 2025-09-21T15:16:01Z

## Summary

* Working tree consumes ~240.46 MiB; `.git/objects/pack` holds 108.39 MiB (45.1% of the tree) due to historical blobs.
* `training/frames` stores ~85.18 MiB of extracted images; archival candidates should go through the sweeper once they age out.
* Manual log exports under `output/manual_logs` occupy ~9.45 MiB and account for 11 of the 23 duplicate sets detected.
* 174 Python/shell scripts are not referenced by import heuristics (see table below); many look like standalone CLI utilities that can be retired or documented.
* Large single files include `bfg.jar` (13.81 MiB) and calibration assets; consider storing binaries in release buckets instead of the repo when practical.

## Top 30 directories by size

| Rank | Directory | Size (MB) |
| --- | --- | ---: |
| 1 | `.` | 240.46 |
| 2 | `./.git` | 108.82 |
| 3 | `./.git/objects` | 108.48 |
| 4 | `./.git/objects/pack` | 108.47 |
| 5 | `./training` | 87.03 |
| 6 | `./training/frames` | 85.18 |
| 7 | `./output` | 12.29 |
| 8 | `./output/manual_logs` | 9.45 |
| 9 | `./logs` | 2.95 |
| 10 | `./logs/pipeline` | 2.90 |
| 11 | `./output/soccer` | 2.33 |
| 12 | `./output/soccer/logs` | 2.28 |
| 13 | `./training/labels` | 1.32 |
| 14 | `./analysis` | 1.29 |
| 15 | `./training/logs` | 0.50 |
| 16 | `./output/summary` | 0.43 |
| 17 | `./scripts` | 0.38 |
| 18 | `./tools` | 0.37 |
| 19 | `./analysis/__pycache__` | 0.36 |
| 20 | `./export` | 0.23 |
| 21 | `./export/jenks_silver_20250913` | 0.23 |
| 22 | `./export/jenks_silver_20250913/data` | 0.21 |
| 23 | `./tests` | 0.18 |
| 24 | `./..bfg-report` | 0.17 |
| 25 | `./..bfg-report/2025-08-17` | 0.17 |
| 26 | `./..bfg-report/2025-08-17/11-01-53` | 0.16 |
| 27 | `./__pycache__` | 0.11 |
| 28 | `./tools/__pycache__` | 0.11 |
| 29 | `./.git/hooks` | 0.08 |
| 30 | `./soccer` | 0.07 |

## Top 200 largest files

| Rank | File | Size (MB) | Modified |
| --- | --- | ---: | --- |
| 1 | `./.git/objects/pack/pack-ccc21652c78cb38fc498519a683466b15e6d79aa.pack` | 108.39 | 2025-09-21 15:01:33.8893635800 |
| 2 | `./bfg.jar` | 13.81 | 2025-09-21 15:01:34.6653510900 |
| 3 | `./yolov8n.pt` | 6.25 | 2025-09-21 15:01:35.5173373760 |
| 4 | `./calib_frame.png` | 2.51 | 2025-09-21 15:01:34.6813508320 |
| 5 | `./output/soccer/logs/20250830_031313-U11_vs_Opponent_20250830.log` | 2.12 | 2025-09-21 15:01:34.8293484510 |
| 6 | `./out.flv` | 1.31 | 2025-09-21 15:01:34.7933490300 |
| 7 | `./logs/pipeline/run_2025-07-29_12-42-32.log` | 1.18 | 2025-09-21 15:01:34.6893507030 |
| 8 | `./logs/pipeline/run_2025-07-29_07-44-17.log` | 1.18 | 2025-09-21 15:01:34.6893507030 |
| 9 | `./firefox_listing.html` | 0.65 | 2025-09-21 15:01:34.6813508320 |
| 10 | `./GAMEDAY_AUDIT.json` | 0.56 | 2025-09-21 15:01:34.5053536720 |
| 11 | `./logs/pipeline/run_2025-08-09_22-14-21.log` | 0.34 | 2025-09-21 15:01:34.6893507030 |
| 12 | `./training/frames/play_71_00-02-21.jpg` | 0.31 | 2025-09-21 15:01:35.3213405310 |
| 13 | `./training/frames/play_71_00-02-20.jpg` | 0.30 | 2025-09-21 15:01:35.3173405960 |
| 14 | `./training/frames/play_89_00-02-56.jpg` | 0.29 | 2025-09-21 15:01:35.3893394360 |
| 15 | `./training/frames/play_90_00-02-59.jpg` | 0.29 | 2025-09-21 15:01:35.3973393070 |
| 16 | `./training/frames/play_122_00-04-03.jpg` | 0.28 | 2025-09-21 15:01:34.9413466480 |
| 17 | `./training/frames/play_54_00-01-47.jpg` | 0.28 | 2025-09-21 15:01:35.2493416900 |
| 18 | `./training/frames/play_121_00-04-01.jpg` | 0.28 | 2025-09-21 15:01:34.9373467120 |
| 19 | `./training/frames/play_55_00-01-48.jpg` | 0.28 | 2025-09-21 15:01:35.2533416250 |
| 20 | `./training/frames/play_123_00-04-04.jpg` | 0.28 | 2025-09-21 15:01:34.9413466480 |
| 21 | `./training/frames/play_55_00-01-49.jpg` | 0.28 | 2025-09-21 15:01:35.2533416250 |
| 22 | `./training/frames/play_84_00-02-47.jpg` | 0.28 | 2025-09-21 15:01:35.3733396940 |
| 23 | `./training/frames/play_122_00-04-02.jpg` | 0.28 | 2025-09-21 15:01:34.9373467120 |
| 24 | `./training/frames/play_96_00-03-11.jpg` | 0.28 | 2025-09-21 15:01:35.4173389860 |
| 25 | `./training/frames/play_95_00-03-09.jpg` | 0.28 | 2025-09-21 15:01:35.4133390500 |
| 26 | `./training/frames/play_97_00-03-12.jpg` | 0.28 | 2025-09-21 15:01:35.4213389210 |
| 27 | `./training/frames/play_75_00-02-29.jpg` | 0.28 | 2025-09-21 15:01:35.3333403380 |
| 28 | `./training/frames/play_29_00-00-57.jpg` | 0.28 | 2025-09-21 15:01:35.1533432350 |
| 29 | `./training/frames/play_73_00-02-24.jpg` | 0.28 | 2025-09-21 15:01:35.3253404670 |
| 30 | `./training/frames/play_72_00-02-23.jpg` | 0.28 | 2025-09-21 15:01:35.3253404670 |
| 31 | `./training/frames/play_89_00-02-57.jpg` | 0.28 | 2025-09-21 15:01:35.3893394360 |
| 32 | `./training/frames/play_61_00-02-01.jpg` | 0.28 | 2025-09-21 15:01:35.2813411750 |
| 33 | `./training/frames/play_56_00-01-50.jpg` | 0.28 | 2025-09-21 15:01:35.2573415620 |
| 34 | `./training/frames/play_112_00-03-42.jpg` | 0.28 | 2025-09-21 15:01:34.8973473560 |
| 35 | `./training/frames/play_54_00-01-46.jpg` | 0.28 | 2025-09-21 15:01:35.2493416900 |
| 36 | `./training/frames/play_90_00-02-58.jpg` | 0.28 | 2025-09-21 15:01:35.3933393720 |
| 37 | `./training/frames/play_73_00-02-25.jpg` | 0.28 | 2025-09-21 15:01:35.3293404020 |
| 38 | `./training/frames/play_61_00-02-00.jpg` | 0.28 | 2025-09-21 15:01:35.2773412390 |
| 39 | `./training/frames/play_60_00-01-59.jpg` | 0.28 | 2025-09-21 15:01:35.2773412390 |
| 40 | `./training/frames/play_41_00-01-21.jpg` | 0.28 | 2025-09-21 15:01:35.2013424620 |
| 41 | `./training/frames/play_120_00-03-58.jpg` | 0.28 | 2025-09-21 15:01:34.9293468410 |
| 42 | `./training/frames/play_82_00-02-42.jpg` | 0.28 | 2025-09-21 15:01:35.3613398870 |
| 43 | `./training/frames/play_31_00-01-01.jpg` | 0.27 | 2025-09-21 15:01:35.1613431070 |
| 44 | `./training/frames/play_96_00-03-10.jpg` | 0.27 | 2025-09-21 15:01:35.4173389860 |
| 45 | `./training/frames/play_33_00-01-05.jpg` | 0.27 | 2025-09-21 15:01:35.1693429780 |
| 46 | `./training/frames/play_47_00-01-33.jpg` | 0.27 | 2025-09-21 15:01:35.2213421410 |
| 47 | `./training/frames/play_56_00-01-51.jpg` | 0.27 | 2025-09-21 15:01:35.2573415620 |
| 48 | `./training/frames/play_51_00-01-41.jpg` | 0.27 | 2025-09-21 15:01:35.2413418180 |
| 49 | `./training/frames/play_49_00-01-37.jpg` | 0.27 | 2025-09-21 15:01:35.2293420120 |
| 50 | `./training/frames/play_60_00-01-58.jpg` | 0.27 | 2025-09-21 15:01:35.2733413040 |
| 51 | `./training/frames/play_114_00-03-46.jpg` | 0.27 | 2025-09-21 15:01:34.9053472270 |
| 52 | `./training/frames/play_95_00-03-08.jpg` | 0.27 | 2025-09-21 15:01:35.4133390500 |
| 53 | `./training/frames/play_119_00-03-57.jpg` | 0.27 | 2025-09-21 15:01:34.9253469050 |
| 54 | `./training/frames/play_62_00-02-02.jpg` | 0.27 | 2025-09-21 15:01:35.2813411750 |
| 55 | `./training/frames/play_48_00-01-35.jpg` | 0.27 | 2025-09-21 15:01:35.2253420760 |
| 56 | `./training/frames/play_50_00-01-38.jpg` | 0.27 | 2025-09-21 15:01:35.2333419470 |
| 57 | `./training/frames/play_50_00-01-39.jpg` | 0.27 | 2025-09-21 15:01:35.2373418830 |
| 58 | `./training/frames/play_80_00-02-39.jpg` | 0.27 | 2025-09-21 15:01:35.3573399520 |
| 59 | `./training/frames/play_91_00-03-00.jpg` | 0.27 | 2025-09-21 15:01:35.3973393070 |
| 60 | `./training/frames/play_46_00-01-31.jpg` | 0.27 | 2025-09-21 15:01:35.2173422050 |
| 61 | `./training/frames/play_59_00-01-56.jpg` | 0.27 | 2025-09-21 15:01:35.2693413680 |
| 62 | `./training/frames/play_48_00-01-34.jpg` | 0.27 | 2025-09-21 15:01:35.2213421410 |
| 63 | `./training/frames/play_76_00-02-30.jpg` | 0.27 | 2025-09-21 15:01:35.3373402730 |
| 64 | `./training/frames/play_13_00-00-25.jpg` | 0.27 | 2025-09-21 15:01:35.0053456170 |
| 65 | `./training/frames/play_121_00-04-00.jpg` | 0.27 | 2025-09-21 15:01:34.9333467760 |
| 66 | `./training/frames/play_53_00-01-45.jpg` | 0.27 | 2025-09-21 15:01:35.2453417540 |
| 67 | `./training/frames/play_64_00-02-06.jpg` | 0.27 | 2025-09-21 15:01:35.2893410460 |
| 68 | `./training/frames/play_112_00-03-43.jpg` | 0.27 | 2025-09-21 15:01:34.8973473560 |
| 69 | `./training/frames/play_76_00-02-31.jpg` | 0.27 | 2025-09-21 15:01:35.3373402730 |
| 70 | `./training/frames/play_91_00-03-01.jpg` | 0.27 | 2025-09-21 15:01:35.4013392440 |
| 71 | `./training/frames/play_52_00-01-43.jpg` | 0.27 | 2025-09-21 15:01:35.2413418180 |
| 72 | `./training/frames/play_45_00-01-29.jpg` | 0.27 | 2025-09-21 15:01:35.2133422700 |
| 73 | `./training/frames/play_15_00-00-28.jpg` | 0.27 | 2025-09-21 15:01:35.0773444590 |
| 74 | `./training/frames/play_94_00-03-07.jpg` | 0.27 | 2025-09-21 15:01:35.4133390500 |
| 75 | `./training/frames/play_80_00-02-38.jpg` | 0.27 | 2025-09-21 15:01:35.3533400160 |
| 76 | `./training/frames/play_44_00-01-27.jpg` | 0.27 | 2025-09-21 15:01:35.2093423330 |
| 77 | `./training/frames/play_51_00-01-40.jpg` | 0.27 | 2025-09-21 15:01:35.2373418830 |
| 78 | `./training/frames/play_68_00-02-14.jpg` | 0.27 | 2025-09-21 15:01:35.3053407880 |
| 79 | `./training/frames/play_46_00-01-30.jpg` | 0.27 | 2025-09-21 15:01:35.2173422050 |
| 80 | `./training/frames/play_40_00-01-19.jpg` | 0.27 | 2025-09-21 15:01:35.1973425270 |
| 81 | `./training/frames/play_14_00-00-26.jpg` | 0.27 | 2025-09-21 15:01:35.0413450380 |
| 82 | `./training/frames/play_68_00-02-15.jpg` | 0.27 | 2025-09-21 15:01:35.3053407880 |
| 83 | `./training/frames/play_21_00-00-40.jpg` | 0.27 | 2025-09-21 15:01:35.1213437510 |
| 84 | `./training/frames/play_47_00-01-32.jpg` | 0.27 | 2025-09-21 15:01:35.2173422050 |
| 85 | `./training/frames/play_92_00-03-03.jpg` | 0.27 | 2025-09-21 15:01:35.4053391790 |
| 86 | `./training/frames/play_72_00-02-22.jpg` | 0.27 | 2025-09-21 15:01:35.3213405310 |
| 87 | `./training/frames/play_94_00-03-06.jpg` | 0.27 | 2025-09-21 15:01:35.4093391150 |
| 88 | `./training/frames/play_28_00-00-54.jpg` | 0.27 | 2025-09-21 15:01:35.1453433640 |
| 89 | `./training/frames/play_28_00-00-55.jpg` | 0.27 | 2025-09-21 15:01:35.1493432990 |
| 90 | `./training/frames/play_13_00-00-24.jpg` | 0.27 | 2025-09-21 15:01:35.0053456170 |
| 91 | `./training/frames/play_63_00-02-04.jpg` | 0.27 | 2025-09-21 15:01:35.2853411100 |
| 92 | `./training/frames/play_77_00-02-32.jpg` | 0.27 | 2025-09-21 15:01:35.3413402090 |
| 93 | `./training/frames/play_19_00-00-36.jpg` | 0.27 | 2025-09-21 15:01:35.1133438790 |
| 94 | `./training/frames/play_17_00-00-33.jpg` | 0.27 | 2025-09-21 15:01:35.1093439440 |
| 95 | `./training/frames/play_81_00-02-40.jpg` | 0.27 | 2025-09-21 15:01:35.3573399520 |
| 96 | `./training/frames/play_59_00-01-57.jpg` | 0.27 | 2025-09-21 15:01:35.2693413680 |
| 97 | `./training/frames/play_140_00-04-38.jpg` | 0.27 | 2025-09-21 15:01:35.0053456170 |
| 98 | `./training/frames/play_49_00-01-36.jpg` | 0.27 | 2025-09-21 15:01:35.2253420760 |
| 99 | `./training/frames/play_39_00-01-16.jpg` | 0.27 | 2025-09-21 15:01:35.1893426560 |
| 100 | `./training/frames/play_53_00-01-44.jpg` | 0.27 | 2025-09-21 15:01:35.2453417540 |
| 101 | `./training/frames/play_93_00-03-04.jpg` | 0.27 | 2025-09-21 15:01:35.4053391790 |
| 102 | `./training/frames/play_41_00-01-20.jpg` | 0.27 | 2025-09-21 15:01:35.1973425270 |
| 103 | `./training/frames/play_64_00-02-07.jpg` | 0.27 | 2025-09-21 15:01:35.2893410460 |
| 104 | `./training/frames/play_35_00-01-09.jpg` | 0.27 | 2025-09-21 15:01:35.1773428490 |
| 105 | `./training/frames/play_119_00-03-56.jpg` | 0.27 | 2025-09-21 15:01:34.9213469700 |
| 106 | `./training/frames/play_30_00-00-59.jpg` | 0.27 | 2025-09-21 15:01:35.1573431700 |
| 107 | `./training/frames/play_97_00-03-13.jpg` | 0.27 | 2025-09-21 15:01:35.4213389210 |
| 108 | `./training/frames/play_79_00-02-37.jpg` | 0.27 | 2025-09-21 15:01:35.3493400810 |
| 109 | `./training/frames/play_44_00-01-26.jpg` | 0.27 | 2025-09-21 15:01:35.2093423330 |
| 110 | `./training/frames/play_93_00-03-05.jpg` | 0.27 | 2025-09-21 15:01:35.4093391150 |
| 111 | `./training/frames/play_69_00-02-16.jpg` | 0.27 | 2025-09-21 15:01:35.3093407250 |
| 112 | `./training/frames/play_113_00-03-44.jpg` | 0.27 | 2025-09-21 15:01:34.9013472910 |
| 113 | `./training/frames/play_92_00-03-02.jpg` | 0.27 | 2025-09-21 15:01:35.4013392440 |
| 114 | `./training/frames/play_38_00-01-14.jpg` | 0.27 | 2025-09-21 15:01:35.1853427200 |
| 115 | `./training/frames/play_17_00-00-32.jpg` | 0.26 | 2025-09-21 15:01:35.1053440070 |
| 116 | `./training/frames/play_62_00-02-03.jpg` | 0.26 | 2025-09-21 15:01:35.2853411100 |
| 117 | `./training/frames/play_63_00-02-05.jpg` | 0.26 | 2025-09-21 15:01:35.2893410460 |
| 118 | `./training/frames/play_79_00-02-36.jpg` | 0.26 | 2025-09-21 15:01:35.3493400810 |
| 119 | `./training/frames/play_139_00-04-37.jpg` | 0.26 | 2025-09-21 15:01:35.0013456820 |
| 120 | `./training/frames/play_43_00-01-24.jpg` | 0.26 | 2025-09-21 15:01:35.2053423990 |
| 121 | `./training/frames/play_34_00-01-06.jpg` | 0.26 | 2025-09-21 15:01:35.1693429780 |
| 122 | `./training/frames/play_57_00-01-52.jpg` | 0.26 | 2025-09-21 15:01:35.2613414970 |
| 123 | `./training/frames/play_40_00-01-18.jpg` | 0.26 | 2025-09-21 15:01:35.1933425910 |
| 124 | `./training/frames/play_20_00-00-39.jpg` | 0.26 | 2025-09-21 15:01:35.1213437510 |
| 125 | `./training/frames/play_58_00-01-54.jpg` | 0.26 | 2025-09-21 15:01:35.2653414330 |
| 126 | `./training/frames/play_107_00-03-32.jpg` | 0.26 | 2025-09-21 15:01:34.8733477410 |
| 127 | `./training/frames/play_27_00-00-52.jpg` | 0.26 | 2025-09-21 15:01:35.1413434280 |
| 128 | `./training/frames/play_67_00-02-13.jpg` | 0.26 | 2025-09-21 15:01:35.3013408520 |
| 129 | `./training/frames/play_58_00-01-55.jpg` | 0.26 | 2025-09-21 15:01:35.2653414330 |
| 130 | `./training/frames/play_38_00-01-15.jpg` | 0.26 | 2025-09-21 15:01:35.1853427200 |
| 131 | `./training/frames/play_139_00-04-36.jpg` | 0.26 | 2025-09-21 15:01:35.0013456820 |
| 132 | `./training/frames/play_125_00-04-09.jpg` | 0.26 | 2025-09-21 15:01:34.9493465190 |
| 133 | `./training/frames/play_67_00-02-12.jpg` | 0.26 | 2025-09-21 15:01:35.3013408520 |
| 134 | `./training/frames/play_30_00-00-58.jpg` | 0.26 | 2025-09-21 15:01:35.1573431700 |
| 135 | `./training/frames/play_39_00-01-17.jpg` | 0.26 | 2025-09-21 15:01:35.1893426560 |
| 136 | `./training/frames/play_33_00-01-04.jpg` | 0.26 | 2025-09-21 15:01:35.1653430420 |
| 137 | `./training/frames/play_118_00-03-55.jpg` | 0.26 | 2025-09-21 15:01:34.9213469700 |
| 138 | `./training/frames/play_98_00-03-14.jpg` | 0.26 | 2025-09-21 15:01:35.4253388570 |
| 139 | `./training/frames/play_74_00-02-26.jpg` | 0.26 | 2025-09-21 15:01:35.3293404020 |
| 140 | `./training/frames/play_81_00-02-41.jpg` | 0.26 | 2025-09-21 15:01:35.3613398870 |
| 141 | `./training/frames/play_101_00-03-21.jpg` | 0.26 | 2025-09-21 15:01:34.8573479990 |
| 142 | `./training/frames/play_117_00-03-53.jpg` | 0.26 | 2025-09-21 15:01:34.9173470330 |
| 143 | `./training/frames/play_8_00-00-14.jpg` | 0.26 | 2025-09-21 15:01:35.3933393720 |
| 144 | `./training/frames/play_106_00-03-30.jpg` | 0.26 | 2025-09-21 15:01:34.8733477410 |
| 145 | `./training/frames/play_18_00-00-35.jpg` | 0.26 | 2025-09-21 15:01:35.1133438790 |
| 146 | `./training/frames/play_57_00-01-53.jpg` | 0.26 | 2025-09-21 15:01:35.2613414970 |
| 147 | `./training/frames/play_83_00-02-45.jpg` | 0.26 | 2025-09-21 15:01:35.3693397590 |
| 148 | `./training/frames/play_19_00-00-37.jpg` | 0.26 | 2025-09-21 15:01:35.1133438790 |
| 149 | `./training/frames/play_84_00-02-46.jpg` | 0.26 | 2025-09-21 15:01:35.3693397590 |
| 150 | `./training/frames/play_75_00-02-28.jpg` | 0.26 | 2025-09-21 15:01:35.3333403380 |
| 151 | `./training/frames/play_20_00-00-38.jpg` | 0.26 | 2025-09-21 15:01:35.1213437510 |
| 152 | `./training/frames/play_120_00-03-59.jpg` | 0.26 | 2025-09-21 15:01:34.9333467760 |
| 153 | `./training/frames/play_43_00-01-25.jpg` | 0.26 | 2025-09-21 15:01:35.2053423990 |
| 154 | `./training/frames/play_113_00-03-45.jpg` | 0.26 | 2025-09-21 15:01:34.9013472910 |
| 155 | `./training/frames/play_109_00-03-37.jpg` | 0.26 | 2025-09-21 15:01:34.8853475490 |
| 156 | `./training/frames/play_32_00-01-03.jpg` | 0.26 | 2025-09-21 15:01:35.1653430420 |
| 157 | `./training/frames/play_66_00-02-11.jpg` | 0.26 | 2025-09-21 15:01:35.2973409180 |
| 158 | `./training/frames/play_74_00-02-27.jpg` | 0.26 | 2025-09-21 15:01:35.3293404020 |
| 159 | `./training/frames/play_15_00-00-29.jpg` | 0.26 | 2025-09-21 15:01:35.0813443940 |
| 160 | `./training/frames/play_22_00-00-43.jpg` | 0.26 | 2025-09-21 15:01:35.1293436220 |
| 161 | `./training/frames/play_11_00-00-20.jpg` | 0.26 | 2025-09-21 15:01:34.9253469050 |
| 162 | `./training/frames/play_99_00-03-17.jpg` | 0.26 | 2025-09-21 15:01:35.4293387930 |
| 163 | `./training/frames/play_70_00-02-19.jpg` | 0.26 | 2025-09-21 15:01:35.3173405960 |
| 164 | `./training/frames/play_34_00-01-07.jpg` | 0.26 | 2025-09-21 15:01:35.1733429130 |
| 165 | `./training/frames/play_14_00-00-27.jpg` | 0.26 | 2025-09-21 15:01:35.0413450380 |
| 166 | `./training/frames/play_31_00-01-00.jpg` | 0.26 | 2025-09-21 15:01:35.1613431070 |
| 167 | `./training/frames/play_101_00-03-20.jpg` | 0.26 | 2025-09-21 15:01:34.8533480640 |
| 168 | `./training/frames/play_153_00-05-04.jpg` | 0.26 | 2025-09-21 15:01:35.0533448440 |
| 169 | `./training/frames/play_18_00-00-34.jpg` | 0.26 | 2025-09-21 15:01:35.1093439440 |
| 170 | `./training/frames/play_27_00-00-53.jpg` | 0.26 | 2025-09-21 15:01:35.1453433640 |
| 171 | `./training/frames/play_35_00-01-08.jpg` | 0.26 | 2025-09-21 15:01:35.1733429130 |
| 172 | `./training/frames/play_105_00-03-28.jpg` | 0.26 | 2025-09-21 15:01:34.8693478060 |
| 173 | `./training/frames/play_11_00-00-21.jpg` | 0.26 | 2025-09-21 15:01:34.9293468410 |
| 174 | `./training/frames/play_131_00-04-20.jpg` | 0.26 | 2025-09-21 15:01:34.9733461330 |
| 175 | `./training/frames/play_128_00-04-14.jpg` | 0.26 | 2025-09-21 15:01:34.9573463900 |
| 176 | `./training/frames/play_107_00-03-33.jpg` | 0.26 | 2025-09-21 15:01:34.8773476770 |
| 177 | `./training/frames/play_152_00-05-03.jpg` | 0.26 | 2025-09-21 15:01:35.0493449090 |
| 178 | `./training/frames/play_153_00-05-05.jpg` | 0.26 | 2025-09-21 15:01:35.0533448440 |
| 179 | `./training/frames/play_138_00-04-35.jpg` | 0.26 | 2025-09-21 15:01:34.9973457460 |
| 180 | `./training/frames/play_37_00-01-12.jpg` | 0.26 | 2025-09-21 15:01:35.1813427840 |
| 181 | `./training/frames/play_29_00-00-56.jpg` | 0.26 | 2025-09-21 15:01:35.1493432990 |
| 182 | `./training/frames/play_10_00-00-18.jpg` | 0.26 | 2025-09-21 15:01:34.8853475490 |
| 183 | `./training/frames/play_106_00-03-31.jpg` | 0.26 | 2025-09-21 15:01:34.8733477410 |
| 184 | `./training/frames/play_140_00-04-39.jpg` | 0.26 | 2025-09-21 15:01:35.0093455530 |
| 185 | `./training/frames/play_70_00-02-18.jpg` | 0.26 | 2025-09-21 15:01:35.3133406600 |
| 186 | `./training/frames/play_109_00-03-36.jpg` | 0.26 | 2025-09-21 15:01:34.8813476140 |
| 187 | `./training/frames/play_22_00-00-42.jpg` | 0.26 | 2025-09-21 15:01:35.1253436860 |
| 188 | `./training/frames/play_37_00-01-13.jpg` | 0.26 | 2025-09-21 15:01:35.1813427840 |
| 189 | `./training/frames/play_124_00-04-07.jpg` | 0.26 | 2025-09-21 15:01:34.9453465830 |
| 190 | `./training/frames/play_116_00-03-51.jpg` | 0.26 | 2025-09-21 15:01:34.9133470980 |
| 191 | `./training/frames/play_132_00-04-22.jpg` | 0.26 | 2025-09-21 15:01:34.9773460680 |
| 192 | `./training/frames/play_125_00-04-08.jpg` | 0.26 | 2025-09-21 15:01:34.9493465190 |
| 193 | `./training/frames/play_77_00-02-33.jpg` | 0.26 | 2025-09-21 15:01:35.3413402090 |
| 194 | `./training/frames/play_124_00-04-06.jpg` | 0.26 | 2025-09-21 15:01:34.9453465830 |
| 195 | `./training/frames/play_99_00-03-16.jpg` | 0.25 | 2025-09-21 15:01:35.4293387930 |
| 196 | `./training/frames/play_88_00-02-54.jpg` | 0.25 | 2025-09-21 15:01:35.3853395010 |
| 197 | `./training/frames/play_104_00-03-27.jpg` | 0.25 | 2025-09-21 15:01:34.8653478700 |
| 198 | `./training/frames/play_105_00-03-29.jpg` | 0.25 | 2025-09-21 15:01:34.8693478060 |
| 199 | `./training/frames/play_83_00-02-44.jpg` | 0.25 | 2025-09-21 15:01:35.3653398230 |
| 200 | `./training/frames/play_86_00-02-50.jpg` | 0.25 | 2025-09-21 15:01:35.3773396300 |

## Duplicate file sets (first 50)

| Set | Size (MB) | Files |
| --- | ---: | --- |
| 1 | 1.18 | `./logs/pipeline/run_2025-07-29_07-44-17.log`<br>`./logs/pipeline/run_2025-07-29_12-42-32.log` |
| 2 | 0.13 | `./output/manual_logs/IMG_7632_log.json`<br>`./output/manual_logs/IMG_7654_log.json` |
| 3 | 0.12 | `./output/manual_logs/IMG_7660_log.json`<br>`./output/manual_logs/IMG_7637_log.json` |
| 4 | 0.11 | `./output/manual_logs/IMG_7592_log.json`<br>`./output/manual_logs/IMG_7580_log.json`<br>`./output/manual_logs/IMG_7615_log.json` |
| 5 | 0.10 | `./output/manual_logs/IMG_7670_log.json`<br>`./output/manual_logs/IMG_7588_log.json` |
| 6 | 0.10 | `./output/manual_logs/IMG_7617_log.json`<br>`./output/manual_logs/IMG_7665_log.json` |
| 7 | 0.09 | `./output/manual_logs/IMG_7590_log.json`<br>`./output/manual_logs/IMG_7582_log.json` |
| 8 | 0.08 | `./output/manual_logs/IMG_7649_log.json`<br>`./output/manual_logs/IMG_7669_log.json` |
| 9 | 0.06 | `./output/manual_logs/IMG_7603_log.json`<br>`./output/manual_logs/IMG_7583_log.json` |
| 10 | 0.06 | `./output/manual_logs/IMG_7599_log.json`<br>`./output/manual_logs/IMG_7589_log.json` |
| 11 | 0.05 | `./output/manual_logs/IMG_7579_log.json`<br>`./output/manual_logs/IMG_7598_log.json` |
| 12 | 0.05 | `./output/manual_logs/IMG_7635_log.json`<br>`./output/manual_logs/IMG_7606_log.json` |
| 13 | 0.01 | `./export/jenks_silver_20250913/data/audit_template.csv`<br>`./output/opponent_jenks_silver_20250913/audit/audit_template.csv` |
| 14 | 0.01 | `./gameday.bak.1757120734`<br>`./gameday.bak.1757121442`<br>`./gameday.bak.1757118262`<br>`./gameday.bak.1757117556`<br>`./gameday.bak.1757121420`<br>`./gameday.bak.1757115366`<br>`./gameday.bak.1757120571`<br>`./gameday.bak.1757121433`<br>`./gameday.bak.1757117729`<br>`./gameday.bak.1757116437`<br>`./gameday.bak.1757120943`<br>`./gameday.orig`<br>`./gameday.bak.1757115646`<br>`./gameday.bak.1757121622`<br>`./gameday.bak.1757121045`<br>`./gameday.bak.1757117550`<br>`./gameday.bak.1757117896`<br>`./gameday.bak.1757121228`<br>`./gameday.bak.1757120824`<br>`./gameday` |
| 15 | 0.01 | `./gameday.bak.1756845824`<br>`./gameday.bak.1756849895` |
| 16 | 0.00 | `./output/opponent_jenks_silver_20250913/quick_tendencies.csv`<br>`./output/opponent_jenks_silver_20250913/audit/audit_summary.csv` |
| 17 | 0.00 | `./tendencies_defense_conf40_nophase.md`<br>`./tendencies_offense_conf40_nophase.md` |
| 18 | 0.00 | `./firefox-115.10.0esr.tar.bz2`<br>`./firefox-115.10.0esr.tar.gz` |
| 19 | 0.00 | `./export/jenks_silver_20250913/data/audit_disagreements.csv`<br>`./output/opponent_jenks_silver_20250913/audit/audit_disagreements.csv` |
| 20 | 0.00 | `./output/soccer/logs/20250830_031920-U11_vs_Opponent_20250830.log`<br>`./output/soccer/logs/20250830_033740-smoketest.log`<br>`./output/soccer/logs/20250830_025725-smoketest.log`<br>`./output/soccer/logs/20250830_032039-U11_vs_Opponent_20250830.log`<br>`./output/soccer/logs/20250830_030657-smoketest.log` |
| 21 | 0.00 | `./tendencies_defense_conf40_nophase.csv`<br>`./tendencies_offense_conf40_nophase.csv` |
| 22 | 0.00 | `./output/summaries/player_play_counts.json`<br>`./output/soccer/meta/thumb_map.json` |
| 23 | 0.00 | `./run-gameday`<br>`./1.0`<br>`./firefox-140.1.0esr.tar.bz2`<br>`./segment/__init__.py`<br>`./tools/__init__.py`<br>`./training/dataset/labels/.gitkeep`<br>`./training/dataset/videos/.gitkeep`<br>`./training/dataset/frames/.gitkeep`<br>`./logs/pipeline_resume_20250916_1139.log`<br>`./logs/stream_status.log`<br>`./analysis/match/__init__.py`<br>`./output/tracking.jsonl`<br>`./output/grades.jsonl`<br>`./output/report.pdf`<br>`./output/play_predictions.jsonl`<br>`./output/plays.jsonl`<br>`./output/.gitkeep`<br>`./models/player_detector/best.onnx`<br>`./models/play_classifier/.upload_nfshrjc9`<br>`./models_bundle/SHA256SUMS` |

## Candidate dead code (heuristic)

The following files were not referenced by simple import/name scans. Review before removing; many are standalone scripts guarded by `if __name__ == "__main__"`.

| File | Reason |
| --- | --- |
| `.install/install_firefox_esr.sh` | shell script name not referenced elsewhere |
| `ai_performance_dashboard.py` | not imported elsewhere; contains __main__ guard (standalone script?) |
| `ai_trainer.py` | not imported elsewhere; contains __main__ guard (standalone script?) |
| `analysis/clipper.py` | not imported elsewhere |
| `analysis/core/ml_utils.py` | not imported elsewhere |
| `analysis/core/vision_utils.py` | not imported elsewhere |
| `analysis/defense_grader.py` | not imported elsewhere |
| `analysis/formation_classifier.py` | not imported elsewhere |
| `analysis/match/formation_matcher.py` | not imported elsewhere |
| `analysis/match/play_matcher.py` | not imported elsewhere |
| `analysis/opponent_report.py` | not imported elsewhere; contains __main__ guard (standalone script?) |
| `analysis/outcomes.py` | not imported elsewhere |
| `analysis/phase_classify.py` | not imported elsewhere; contains __main__ guard (standalone script?) |
| `analysis/play_matcher.py` | not imported elsewhere |
| `analysis/play_segment.py` | not imported elsewhere |
| `analysis/player_identity.py` | not imported elsewhere |
| `analysis/quick_tag.py` | not imported elsewhere; contains __main__ guard (standalone script?) |
| `analysis/report_builder.py` | not imported elsewhere |
| `analysis/reporter.py` | not imported elsewhere |
| `analysis/review_ranker.py` | not imported elsewhere |
| `analysis/role_guess.py` | not imported elsewhere |
| `analysis/validate_jsonl.py` | not imported elsewhere |
| `analysis/video_reader.py` | not imported elsewhere |
| `annotate_clip.py` | not imported elsewhere; contains __main__ guard (standalone script?) |
| `assignment_analyzer.py` | not imported elsewhere; contains __main__ guard (standalone script?) |
| `bin/launch_ffmpeg_shared.sh` | shell script name not referenced elsewhere |
| `build_highlight_dataset.py` | not imported elsewhere; contains __main__ guard (standalone script?) |
| `camera_test.py` | not imported elsewhere |
| `clean_labels.py` | not imported elsewhere; contains __main__ guard (standalone script?) |
| `coach_assistant.py` | not imported elsewhere; contains __main__ guard (standalone script?) |
| `coach_review_app.py` | not imported elsewhere; contains __main__ guard (standalone script?) |
| `drive_service_uploader.py` | not imported elsewhere; contains __main__ guard (standalone script?) |
| `game_uploader.py` | not imported elsewhere; contains __main__ guard (standalone script?) |
| `generate_coaches_cut_and_summary.py` | not imported elsewhere; contains __main__ guard (standalone script?) |
| `generate_coaching_report.py` | not imported elsewhere; contains __main__ guard (standalone script?) |
| `highlight_recorder.py` | not imported elsewhere; contains __main__ guard (standalone script?) |
| `install_firefox_esr.py` | not imported elsewhere; contains __main__ guard (standalone script?) |
| `install_firefox_esr.sh` | shell script name not referenced elsewhere |
| `jersey_detector.py` | not imported elsewhere; contains __main__ guard (standalone script?) |
| `list_hw_encoders.py` | not imported elsewhere; contains __main__ guard (standalone script?) |
| `live_dashboard.py` | not imported elsewhere; contains __main__ guard (standalone script?) |
| `motion_detector.py` | not imported elsewhere; contains __main__ guard (standalone script?) |
| `play_count_tracker.py` | not imported elsewhere; contains __main__ guard (standalone script?) |
| `play_tracker.py` | not imported elsewhere; contains __main__ guard (standalone script?) |
| `player_id/calibration_ui.py` | not imported elsewhere |
| `postgame_review.py` | not imported elsewhere; contains __main__ guard (standalone script?) |
| `practice_trainer.py` | not imported elsewhere; contains __main__ guard (standalone script?) |
| `process_all_uploaded_videos.py` | not imported elsewhere; contains __main__ guard (standalone script?) |
| `record_video.py` | not imported elsewhere; contains __main__ guard (standalone script?) |
| `reporting/debug_summary.py` | not imported elsewhere |

## Notes

* Duplicate detection excludes files under the optional `--keep` paths (none specified here).
* Dead code analysis used a regex-based import scan because `vulture` is unavailable in this environment; expect false positives.
* Disk totals derive from `du -k` and `find` snapshots taken just before generating this report.
