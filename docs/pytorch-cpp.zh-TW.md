# PyTorch C++：研究與已選定路徑

[English](pytorch-cpp.md)

**選定路徑：C++ 直接使用 LibTorch tensor 操作，Python／PyTorch 保留模型學習。已驗證 Windows CPU／CUDA。** 見[原生指南](native-engine.zh-TW.md)及[本機結果](migration-validation.zh-TW.md)。

## C++ 能讓 PyTorch 更快嗎？

不會自動加速。PyTorch Python 介面已呼叫編譯後的 C++／CUDA 運算；官方 C++ 指南明確提醒 Python 不一定比較慢。當操作之間的 Python 工作或低延遲整合成本重要時，C++ 才可能有幫助。[1]

本引擎的完整對弈及推論現在都在 C++，移除了搜尋迴圈內的 Python 棋盤／葉節點生成及模型呼叫。CPU／GPU 傳輸及 GPU 啟動成本仍存在。

本機配對搜尋測試在指定條件下有加速，但不能將模型呼叫語言當作唯一原因，也不能證明每訓練小時的棋力提升。

## 術語與依據

- **推論（inference）：** 預測落子分數及局面價值，不改變權重。
- **LibTorch：** PyTorch C++ 標頭及程式庫，提供 tensor、模型、梯度、CPU 及 GPU 運算。[1][2]
- **派發成本（dispatch cost）：** 數值運算前，CPU 提出模型操作請求的工作。
- **CUDA graph：** 以一次啟動請求重播已保存的 GPU 操作，可省去 Python、C++ 及驅動派發工作，但需要固定記憶體與相容形狀／控制流程。[3]
- **AOTInductor：** 將匯出的 PyTorch 模型編譯供 Python／C++ 使用的工具。2.11 指南示範 `torch.export` 與 C++ package loader。[5]

## 選擇此路徑的原因

`engine/inference.cpp` 以 LibTorch tensor 操作，實作與 `engine/network.py` 相同的殘差、池化及注意力架構。Python 在安全點傳入更新權重及 running buffers；兩端共用架構／寬度／深度／棋盤設定，並以相同輸入比對。

建置使用已安裝 PyTorch wheel。本機 Windows 已驗證 MSVC Release 風格連結及 CPU／CUDA 推論，不需要 TorchScript／export loader、CMake 或另裝 CUDA SDK。這不代表已確認 Windows CUDA 的 AOTInductor 或 `torch.compile` 支援。

2.11 文件標示 TorchScript 已棄用。[6] Export／AOTInductor 仍是未來部署選項；目前 Python 保留學習且可直接傳遞權重，因此不需要它。

## 本版未納入的其他選項

- Python／PyTorch 推論能縮小移植範圍，但不是使用者選定需求。
- CUDA graphs 可降低兩種語言的啟動工作；動態自我對弈批次須另設計符合 graph 限制的路徑。[3]
- `torch.compile(mode="reduce-overhead")` 在可行時使用 CUDA graphs，可能增加 workspace，不保證加速。[4]
- Windows 編譯器教學確認 CPU／XPU 設定，不是本機 Windows／NVIDIA CUDA 編譯路徑。[7]

量測剩餘需求後才考慮加入，不以這些尚未測試的路徑宣稱效能。

## 檢查與限制

CPU／CUDA 檢查比對模型及 learner 輸出；真實更新後再載入權重／running buffers 並比對。完整棋局及暫存訓練／續訓依序涵蓋 3×3、15×15、19×19。

配對搜尋量測使用相同權重、棋規、批次、模擬數及 FP32 模式；這是測試控制條件，不是精確重現承諾。暖機後搜尋中位數不含建置、模型載入及暖機；GPU 程序完整時間預算包含啟動及暖機。原生推論時間包含輸入／輸出傳輸及等待，不是純 GPU kernel 時間。

Agent 授權仍為累計五分鐘暫存 GPU 程序時間，不建立持久訓練。長期棋力比較由使用者執行。檔案 I/O 另行量測，不能靠 C++ 推論消除。

## 官方來源

1. [PyTorch C++ 介面：範圍與效能提醒](https://docs.pytorch.org/cppdocs/frontend.html)
2. [LibTorch 安裝與連結](https://docs.pytorch.org/cppdocs/installing.html)
3. [PyTorch：CUDA graphs 與限制](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/)
4. [PyTorch 2.11：`torch.compile`](https://docs.pytorch.org/docs/2.11/generated/torch.compile.html)
5. [PyTorch 2.11：AOTInductor 與 C++ 推論](https://docs.pytorch.org/docs/2.11/user_guide/torch_compiler/torch.compiler_aot_inductor.html)
6. [PyTorch 2.11：TorchScript 棄用說明](https://docs.pytorch.org/docs/2.11/notes/cpu_threading_torchscript_inference.html)
7. [PyTorch：Windows CPU／XPU 編譯器設定](https://docs.pytorch.org/tutorials/unstable/inductor_windows.html)
