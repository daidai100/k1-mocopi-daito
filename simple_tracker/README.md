# simple_tracker — 軽量な汎用モーション追従（K1）

目的：mocopi からのリアルタイム入力を想定し、「ある程度汎用」な全身追従ポリシーを
**単段 PPO・7 報酬項・3.5 時間のデータ**で、1 台の RTX 5090 で 15 時間以内に学習させる。
既存の `src/k1_motion`（teacher/student・18k クリップ・複雑な報酬）には手を入れず、別ディレクトリに追加した。

## 変更点（このブランチ `daidai/simple-tracker` で追加したもの）

| パス | 内容 |
| --- | --- |
| `retarget/bandai_to_k1.py` | Bandai Namco BVH → K1（GMR）。回転オフセットの自動キャリブレーション、脚長スケール、接地、品質指標。GMR 本体は無改変 |
| `retarget/bvh_bandai_to_k1.json` | 上記で生成した GMR IK 設定 |
| `retarget/render_check.py` | リターゲット結果のオフスクリーン確認動画（GUI 不要） |
| `isaaclab/build_library.py` | 全クリップを 1 つのライブラリ npz に（Isaac の K1 で FK を一括計算、URDF 基準で再接地） |
| `isaaclab/k1_simple_tracker/` | Isaac Lab タスク `K1-Simple-Tracker-v0`（コマンド・観測・報酬・終了条件・PPO 設定） |
| `isaaclab/run_rsl_rl.py` | booster_train の train.py / play.py をそのまま使うラッパー |
| `isaaclab/export_policy.py` | チェックポイント → TorchScript / ONNX（Isaac 不要、学習中に実行可） |
| `deploy/simple_tracker_policy.py` | booster_deploy 用ポリシー（観測契約は下記） |
| `deploy/deploy.py`, `deploy/sim2sim_eval.py` | 実機 / MuJoCo 実行、ヘッドレス sim2sim 評価（成功率・誤差・動画） |
| `live/mocopi_retarget.py` | mocopi → K1 を 1 フレームずつ（因果的に）GMR でリターゲット。`udp`（ライブ）/ `bvh`（録画を同じ処理に通す・`--stream` で実時間送信） |
| `live/causal_reference.py` | K1 qpos 列 → 参照（後退差分＋EMA の速度、URDF 用の高さ補正）。ライブとオフライン検証で共通 |

## 学習方法の変更点

既存 k1-mocopi（`docs/training-campaign.md` 系）および booster_train の単一クリップ BeyondMimic との差分：

1. **データを小さく・きれいに**：Bandai Namco 1+2 の 3,077 本から、走行・ダッシュ（K1 換算で約 2 m/s）、
   極端な低姿勢・関節速度・接地貫通のクリップを除外し、**2,724 本・3.54 時間**にした。
   GMR の回転オフセットは手書きの表をコピーせず、直立フレームから自動で求める。
   リターゲット用 MJCF と学習用 URDF では脚の付け根が 1.5 cm ずれているため、URDF 側で再接地した。
2. **単段 PPO＋非対称 critic**：teacher/student の蒸留はしない。方策の入力は実機で得られる量だけにし、
   critic は特権情報（ボディ位置、並進速度）を使う。
3. **観測はワールド座標に依存しない**：BeyondMimic の `motion_anchor_pos/ori_b`（参照とロボットのワールド姿勢差）は外した。
   参照側の入力は「関節角・関節速度・参照胴体の重力方向・heading 座標系での並進速度・角速度・胴体高さ」だけ。
   mocopi の位置やヨーがドリフトしても方策の入力は変わらない。
4. **報酬 7 項・σ 固定**：BeyondMimic の追従 4 項（ボディ相対位置・姿勢・速度・角速度）＋正則化 3 項
   （action rate、関節リミット、不要接触）。ワールドのアンカー位置・姿勢の 2 項は削除した。
   速度も参照をロボットの heading に揃えて比較する。適応 σ は使わない。
5. **クリップ横断の適応サンプリング**：1 秒ビン（クリップをまたがない）ごとに失敗率の EMA を持ち、
   50% を失敗重み、50% を時間一様で開始点を引く（BeyondMimic / GMT 方式）。
   クリップの終端は打ち切り（truncation）として扱い、テレポートはしない。
6. **PPO の安定化**：`noise_std_type="log"`。以前の学習は `normal expects all elements of std >= 0.0` で落ちていた。
   あわせて MLP を 1024-512-256 に広げ、250 イテレーションごとに保存する。
7. **sim2real 周りは変更なし**：アクチュエータモデル（2〜8 物理ステップの遅延）、ドメインランダム化、
   80% ほぼ平坦な地形、PD ゲインは booster_train の K1 設定のまま。booster_deploy もそのまま使える。

参考にした前例：BeyondMimic（報酬・適応サンプリング）、GMT（多クリップでの失敗重み付きサンプリング）、
ExBody2 / OmniH2O / HumanPlus（テレオペでは現在フレームの参照だけを使い、ルートは速度で条件付け）、
TWIST（teacher/student は精度が上がるが重いため今回は不採用）。

## 観測（方策 126 次元・この順序）

```
ref_joint_pos(22) ref_joint_vel(22)   # sim(Isaac) 関節順
ref_root_gravity_b(3) ref_root_lin_vel_heading(3) ref_root_ang_vel_b(3) ref_root_height(1)
base_ang_vel_b(3) projected_gravity_b(3) joint_pos-default(22) joint_vel(22) last_action(22)
```
行動：`target = default + action * 0.25 * effort_limit / stiffness`（booster_train と同じ）

## 使い方

```bash
# 1) リターゲット（GMR venv、20 並列で約 3 分）
~/ws/mocopi2beyondmimic-ref/GMR/.venv/bin/python simple_tracker/retarget/bandai_to_k1.py \
  --bandai ~/ws/k1-mocopi-data/bandai/dataset --out ~/ws/k1-mocopi-data/k1_bandai --workers 20
# 2) ライブラリ（Isaac、約 8 分）
cd ~/ws/beyondmimic && OMNI_KIT_ACCEPT_EULA=Y .venv-isaaclab/bin/python ~/ws/k1-mocopi/simple_tracker/isaaclab/build_library.py \
  --src ~/ws/k1-mocopi-data/k1_bandai --out ~/ws/k1-mocopi-data/k1_bandai_library.npz --headless
# 3) 学習（~/ws/k1-mocopi で実行 → logs/rsl_rl/k1_simple_tracker/）
OMNI_KIT_ACCEPT_EULA=Y ~/ws/beyondmimic/.venv-isaaclab/bin/python simple_tracker/isaaclab/run_rsl_rl.py \
  train --task K1-Simple-Tracker-v0 --headless --run_name bandai_v1
# 4) 書き出し（学習中でも可）
~/ws/beyondmimic/.venv-isaaclab/bin/python simple_tracker/isaaclab/export_policy.py logs/rsl_rl/k1_simple_tracker/<run>/model_<N>.pt
# 5) sim2sim（ヘッドレス、MuJoCo、動画＋成功率）
cd ~/ws/beyondmimic/booster_deploy && MUJOCO_GL=egl .venv/bin/python ~/ws/k1-mocopi/simple_tracker/deploy/sim2sim_eval.py \
  --checkpoint <exported/policy_N.pt> --clips '_normal_00[1-2]$' --video s2s.mp4 --json s2s.json
#    まとめて：最新チェックポイントの書き出し＋sim2sim（Bandai 23 本＋mocopi 録画 2 本、動画は run ディレクトリへ）
simple_tracker/eval_latest.sh            # 引数でチェックポイントやクリップの正規表現を指定可
# 6) 実機（booster_deploy と同じ手順。--mujoco を付けると GUI ビューア版）
cd ~/ws/beyondmimic/booster_deploy && .venv/bin/python ~/ws/k1-mocopi/simple_tracker/deploy/deploy.py \
  --checkpoint <exported/policy_N.pt> --clips 'dataset-2_wave-right-hand_normal_001$'
```

## 途中経過（学習 run `2026-09-26_20-58-22_bandai_v1`）

- 20:58 開始、`timeout 15h` で 9/27 11:58 ごろ自動停止。約 2.2 s/iter（約 24k iter の見込み）。
- iter 500：終了要因の 64% がクリップ完走、33% が手足の高さずれ（失敗）。行動 std 0.36。崩壊なし。
- **iter 1000（約 40 分）の MuJoCo sim2sim**：Bandai `*_normal_001` の 23 本は全て転倒なし（関節誤差 平均 0.16 rad）。
  mocopi 録画 2 本（ライブと同じ因果処理）も転倒なし。ただし深いランジ（胴体 0.42 m）は浅くしか再現しない。
  Bandai データには深くしゃがむ動きがほとんどない（胴体はほぼ 0.47 m 以上）ため、学習が進んでも限界がありうる。

- iter 6000（3 時間 38 分）：Bandai 23 本は全て転倒なし、関節誤差 0.129 rad。mocopi 録画は 0.138 / 0.097 rad（iter 1000 では 0.187 / 0.138）。

- **最終（iter 14000、11:58 停止）**：Bandai 23 本すべて転倒なし、関節誤差 0.122 rad。mocopi 録画は 0.122 / 0.110 rad。
- 注意：`timeout` の SIGTERM を受けると Isaac が終了処理で固まり、GPU を保持したまま残る（停止後に PID 指定で kill した）。
  チェックポイントは 250 iter ごとに保存されるため、最後の 250 iter 未満は失われる。

## LAFAN1 版（並行 run `*_lafan1_v1`、9/27 00:50 開始）

- `bandai_to_k1.py --format lafan1 --calib_frame auto --segment`：長い収録を区間に分け、床での動作・傾いた胴体・
  1 秒平均で 1.3 m/s 超の移動・両足が浮いた区間・激しい関節速度を ±0.5 秒の余白付きで除外する。
  77 本 → 577 区間・2.68 時間（walk 43 分、dance 22 分、aiming 19 分、obstacles 20 分ほか）。
- 学習は同じタスクで、ライブラリだけ `K1_TRACKER_LIBRARY=~/ws/k1-mocopi-data/k1_lafan1_library.npz` で差し替える。
- 2 本の同時実行では、どちらも約 5.2 s/iter（単独時は 2.25 s/iter）。合計スループットは単独時より約 13% 低い。
  結果として、どちらの run も停止時点で 14k iter（最終チェックポイント `model_14000.pt`）だった。
- iter 10750（13:50）：LAFAN1 subject1 の 34 区間すべて転倒なし、関節誤差 0.113 rad（265 秒の歩行を含む）。mocopi 録画は 0.129 / 0.099 rad。
- `~/ws/k1-mocopi-data/auto_eval.sh` が 07:30 に両 run の最新チェックポイントを sim2sim にかけ、
  各 run の終了後にも最終チェックポイントを評価する予定だったが、相対パスの不具合で失敗した（修正済み）。
  LAFAN1 の最終評価は `auto_eval_lafan.sh` が 15:55 に固まったプロセスを kill してから実行した（結果：`~/ws/k1-mocopi-data/auto_eval.log`）。
- **最終（iter 14000、15:50 停止）**：LAFAN1 subject1 の 34 区間すべて転倒なし、関節誤差 0.110 rad
  （苦手なのは fightAndSports の 0.18 rad、それ以外は 0.09〜0.13 rad）。mocopi 録画は 0.136 / 0.089 rad。

### Bandai 版と LAFAN1 版の比較（どちらも iter 14000、MuJoCo sim2sim、関節誤差 rad）

| 評価対象 | Bandai | LAFAN1 |
| --- | ---: | ---: |
| それぞれの学習データのクリップ | 23/23 完走、0.122 | 34/34 完走、0.110 |
| mocopi 録画 MCPM_20260922_163917（ランジを含む） | **0.122** | 0.136 |
| mocopi 録画 MCPM_20260922_164010 | 0.110 | **0.089** |

- 学習データにない mocopi 録画では優劣がはっきりしない。どちらも転倒はしていない。
- 評価クリップはどれも学習データに含まれるため、未知の動作への汎化はまだ測れていない。
- 書き出したポリシー：`logs/rsl_rl/k1_simple_tracker/<run>/exported/policy_14000.{pt,onnx}`（ログは git に含めていない）。
- LAFAN1 は CC BY-NC-ND 4.0。リターゲット後のデータは再配布しないこと（ローカルの研究利用のみ）。

## 既知の制限・次の手順

- **ライブ mocopi**：`live/mocopi_retarget.py udp`（GMR venv）が mocopi アプリの UDP（12351）を受けて
  K1 の qpos を 127.0.0.1:12400 に送り、`deploy.py --source udp` が受け取る。最初の 1 秒は静止して立つ
  （スケールと接地のキャリブレーション）。BVH で同じ処理を再現でき、`bvh --stream` で UDP 経路を
  mocopi なしでテストできる（受信値がオフライン処理と一致することを確認済み）。
  **UDP のボーン座標系が BVH と同じという前提は実機の mocopi では未確認。** 最初は `--mujoco` で確認すること。
- 歩行の参照速度は p95 で約 1 m/s あり、K1 には速い可能性がある。追従しきれない場合はクリップを
  時間方向に伸ばす（例：1.25 倍）か、速度でフィルタする。
- K1 の腕は短い（約 0.16 m）ため、人の肘を K1 の手先に対応させている（GMR の smplx_to_k1 と同じ）。
- 低い姿勢（ランジ・スクワット）はデータが少ない。必要なら mocopi で自分の動きを録ってライブラリに追加する
  （`live/mocopi_retarget.py bvh` で作る npz と同じ形式で `build_library.py` に入れる経路を用意するとよい）。
- 参照に未来フレームは使わない（ライブでは得られないため）。
- ライセンス：Bandai Namco データは CC BY-NC 4.0。派生データ（`~/ws/k1-mocopi-data`）はリポジトリに含めない。
