# ZeroFG Android

Android 無需 Root 的即時 Vulkan 補幀與空間超解析應用程式，基於 [LSFG-Android](https://github.com/FrankBarretta/LSFG-Android) 架構並將底層補幀引擎完整替換為開源的 [ZeroFG 1.2.1](https://github.com/hy300leosquizz-ctrl/ZeroFG)（免 `Lossless.dll`）。

An open-source, rootless Android Vulkan frame generation & spatial super-resolution app built on the [LSFG-Android](https://github.com/FrankBarretta/LSFG-Android) capture/overlay host and powered by [ZeroFG 1.2.1](https://github.com/hy300leosquizz-ctrl/ZeroFG) (no `Lossless.dll` required).

---

## 主要功能 / Features

- **ZeroFG 1.2.1 補幀核心：**
  - 新增 **Advanced VTF Filter（Vector Trail Fill）** 與色彩導向光流邊緣，大幅改善角色／物體移動遮擋背景時的邊緣光暈與果凍假影。
  - 支援 **2× / 3× / 4×** 倍率即時熱切換（依螢幕更新率自動限速，支援同對多相位光流重用 MFG Flow Reuse）。
  - 內建全域鏡頭平移搜尋（`df_cam`）、6 幀靜止 HUD／字幕鎖定（HUD Hold）與切鏡／淡入淡出保護。
  - 自動啟用 `Backend::kAuto`、`FP16`、`Subgroup`、`VK_KHR_global_priority` 高優先權 GPU Queue 與 `VK_GOOGLE_DISPLAY_TIMING`（不支援時安全降級）。
- **動漫模式（Anime Mode）：**
  - 自動略過動畫重複畫格，支援拍一（24fps）、拍二（12fps）、拍三（8fps）連續等距輸出網格。
  - 選 `2×` 鎖定省電穩定的 45/60fps；選 `3× / 4×` 解鎖滿刷 90/120Hz，消除 2:3 抖動。
- **GPU 空間超解析（Spatial SR）：**
  - 內建 **Qualcomm SGSR v1（Edge-Direction 12-tap，跨 GPU 支援 Mali 與 Adreno）**、**AMD FSR 1（EASU + RCAS）**、**Anime4K**，與 ZeroFG 1.2 並行即時熱切換。
- **高效能非同步管線與直通疊加層：**
  - 擷取、ZeroFG 計算與呈現排程完全解耦（獨立 `presentThread` + 專屬 `VkCommandPool`），支援直屏與橫屏全螢幕自動偵測解析度與硬體旋轉。
  - 採用直通 `SurfaceView` + Vulkan WSI Swapchain。
- **雙語介面：** 內建首頁一鍵即時切換 **繁體中文 / English / 跟隨系統**。

---

## 系統需求 / Requirements

- Android 10+（API 29+）；強烈建議 **Android 14+**，以便在螢幕錄製選擇器中使用「單一應用程式」擷取避免錄到疊加層。
- GPU 驅動需支援 **Vulkan 1.3**、`synchronization2` 與 `shaderStorageImageExtendedFormats`（支援高通 Adreno 與聯發科天璣 Mali/Immortalis）。

---

## 建置方式 / Build

使用 JDK 17 或 21（例如 Android Studio 內建 JBR）、Android SDK Platform 35、NDK `29.0.14206865` 與 CMake `3.22.1`（Windows 上請使用純 ASCII 路徑或 `subst` 磁碟機代號避免 AIDL 非 ASCII 路徑編碼錯誤）：

```powershell
.\gradlew.bat :app:assembleDebug
```

產出的 APK 位於 `app/build/outputs/apk/debug/`：
- `arm64-v8a`：一般 Android 手機／平板
- `x86_64`：模擬器

---

## 特別感謝與上游專案 / Acknowledgements & Upstream Sources

本專案建立於以下優秀開源專案的基礎之上，特此致謝：

- **[LSFG-Android](https://github.com/FrankBarretta/LSFG-Android)** by **Francesco Barretta** — 感謝提供免 Root Android 螢幕擷取、無障礙觸控穿透疊加層、懸浮控制抽屜與 Vulkan 宿主框架（保留原始 [LICENSE](LICENSE)）。
- **[ZeroFG (v1.2.1)](https://github.com/hy300leosquizz-ctrl/ZeroFG)**（commit `c5e9b64`）— 感謝開源高效能行動端 Vulkan 補幀引擎，支援 Advanced VTF Filter、多相位光流重用、全域平移搜尋與 HUD 固定（原始碼與授權位於 `third_party/zerofg/`，採 Apache-2.0 / GPL-2.0+ 雙授權）。
- **[Snapdragon Game Super Resolution (SGSR v1)](https://github.com/SnapdragonGameStudios/snapdragon-gsr)** by **Qualcomm Innovation Center / Snapdragon Game Studios** — 感謝開源單 Pass 空間超解析演算法（BSD-3-Clause）。
- **[volk](https://github.com/zeux/volk)**（commit `a8da8ef`）— Vulkan meta-loader（見 `third_party/volk/LICENSE.md`）。
