# 移植檢查與搜尋效能量測

[English](migration-validation.md)

## 環境與方法

Windows 桌上型電腦，配備多核心 CPU 和支援 CUDA 的 GPU；PyTorch 2.11.0+cu128；已安裝 MSVC x64 工具。為保護隱私，不公開確切硬體規格，因此下列時間僅供參考，無法依完整硬體條件重現。

配對搜尋使用相同權重、合法開局棋盤、棋規、批次、模擬數，Python／C++ 都使用 **FP32**。下列為三次暖機後執行的中位數，以 `time.perf_counter` 量測。原生推論包含必要傳輸及等待；建置、模型載入及暖機不計入穩態搜尋中位數，但計入 GPU 測試程序完整時間預算。

這是配對**搜尋**時間，不是已完成棋局數比例或訓練棋力的量測。啟動、學習及檔案 I/O 各有獨立成本。

## CUDA：預設大小的批次與搜尋

64 個根節點、64 次模擬，C++ 使用預設六個工作執行緒。3×3 模型為 8 通道／1 區塊，其他棋盤為 64／6。

| 遊戲 | Python 參考 | C++／LibTorch | 比例：Python／原生 |
| --- | ---: | ---: | ---: |
| 3×3 井字棋 | 378.60 ms | 30.69 ms | 12.34× |
| 15×15 五子棋 | 582.44 ms | 224.82 ms | 2.59× |
| 19×19 六子棋 | 664.77 ms | 249.23 ms | 2.67× |

模型參數數量依序為 3,084、477,764、495,172。上述數字較大不代表棋力較好。

## CPU 與工作執行緒取捨

八個根節點、16 次模擬、相同模型、三次暖機後執行：

| 遊戲 | Python CPU | 原生 CPU，1 工作執行緒 | 原生 CPU，6 工作執行緒 |
| --- | ---: | ---: | ---: |
| 3×3 | 23.97 ms | 5.00 ms | 10.74 ms |
| 15×15 | 89.83 ms | 59.82 ms | 58.23 ms |
| 19×19 | 89.19 ms | 70.64 ms | 81.21 ms |

小型 CPU 井字棋及六子棋批次中，六個工作執行緒慢於一個。更多執行緒不一定更快；較大的 CUDA 六子棋批次則是六個快於一個（249.23 vs 267.43 ms）。建立及協調執行緒有成本。保留預設六個工作執行緒 preset，但須依裝置及批次調整 `--workers`。

早期原生 profiling 發現，掃描無威脅連線時產生不必要配置。現在遇到過多空格或對手棋子即停止該連線掃描；修改後重新執行完整棋規／特徵／證明檢查。

## 正確性與框架檢查

- **全部 5,478 個合法井字棋局面**的原生／參考棋規、動作及特徵窮舉比對。
- 完整原生棋局依序完成 3×3、15×15、19×19。Smoke 棋局最多使用四次模擬，只檢查運作，不檢查棋力。
- CPU 搜尋策略／回傳與參考實作一致，包含六子棋同一玩家連續落子。
- CPU／CUDA 模型輸出一致。CPU 涵蓋 32/2、64/6、128/10 五子棋模型；真實更新後，CPU／CUDA 再比對刷新權重及 running buffers。
- 原生多回合 TSS 證明亦通過 Python 驗證器。缺漏回應被拒絕，未滿回合及已保存防守分支可正確續跑，上限結果仍為未知。
- 暫存 CPU／CUDA 學習：32 次更新後停止，從八盤續訓至九盤。CUDA 涵蓋 15×15／19×19 預設 64/6 模型及學習 batch 128，保留未完成棋局。
- Checkpoint 保存有界 replay；只有摘要統計，不封存自我對弈棋譜或永久逐局 replay。摘要寫入失敗可由持久 checkpoint 復原。
- 原子保存／磁碟上限／run lock，以及凍結 100 盤 gate、升級、程式碼／checksum 變更與重啟封存檢查。
- Chrome 檢查：15×15／13×13／19×19 棋盤、完整 bot 回合、圖表、鍵盤、行動版、本機 token 保護、無瀏覽器訓練控制、無 JavaScript 錯誤。

**68 個單元測試全部通過**（27.516 秒），僅使用 CPU。瀏覽器檢查亦以暫存 CPU run 通過，未啟動持久訓練實驗。測試通過不證明沒有 bug。

CUDA 學習檢查中，PyTorch 回報配置峰值約為五子棋 139.2 MiB、六子棋 205.6 MiB。這是程序峰值，不是隔離的單模型記憶體或總 VRAM；不含 CUDA context／driver 及 PyTorch allocator 外部記憶體。不能據此認定所有允許的模型／batch 都能放入記憶體。

## 測試時間預算

暫存 GPU 測試程序累計**約 104 秒**，包含啟動及暖機，低於核准的五分鐘上限。CPU 檢查、原生建置及 CPU 瀏覽器檢查不占 GPU 額度；總數已包含所有重複檢查；上述最終搜尋量測與單元／瀏覽器檢查分開執行。暫存 run 目錄已移除。

暖機後搜尋時間不包含建置。原生建置使用現有編譯器成功；一次建置後首次載入曾短暫被 Windows application control 拒絕，後續載入及全部檢查皆通過。沒有停用安全設定。

## 執行檢查

```sh
uv run python -m engine.native
uv run python -m unittest discover -s tests -v
uv run --group browser python tests/ui_smoke.py
uv run python -m experiments.benchmark_native --device cpu --output .native-cache/checks.json
uv run python -m experiments.benchmark_native --device cuda --batch 64 --simulations 64 --wall-seconds 90 --output .native-cache/cuda-checks.json
uv run python -m experiments.benchmark_native --device cuda --repeats 1 --simulations 8 --training-presets tictactoe,gomoku,connect6 --wall-seconds 90 --output .native-cache/training-checks.json
```

Benchmark controller 限制子程序實際時間，使用暫存訓練目錄。CUDA 指令由使用者啟動，不代表另授權 agent GPU 工作。原始本機 JSON 保留在被忽略的 `.native-cache/`。

專案目標仍是提高每訓練小時的棋力，需要使用者執行長期控制比較。本報告不宣稱棋力提升、完整訓練吞吐量或最佳硬體利用率。
