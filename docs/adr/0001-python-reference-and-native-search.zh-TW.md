---
status: superseded
---

# 原生對弈之前先建立 Python 正確性參考

[English](0001-python-reference-and-native-search.md)

原決議先實作有界 Python 威脅空間搜尋及獨立證明驗證器，再將量測確認的 CPU 瓶頸移入原生程式。保留易讀的參考實作，才能在不削弱合法落子、對手先獲勝及完整防守檢查的前提下驗證原生實作。目標是每訓練小時的棋力增益，不只是吞吐量。

[已核准移植共識](../migration-requirements.zh-TW.md)取代了部分移植計畫：正式對弈及推論現在使用 C++，Python／PyTorch 負責學習及實驗管理。Python 棋規、搜尋、戰術及證明參考實作只供測試，不是正式對弈 fallback。已移除過時半原生 forest 及其效能測試。

現行設計見[原生引擎](../native-engine.zh-TW.md)及[共用模型實驗](0002-shared-board-model-experiments.zh-TW.md)。原決議全文及移植前報告仍可由 Git 歷史查閱。
