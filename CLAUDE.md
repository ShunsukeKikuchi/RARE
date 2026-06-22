# Competition Workspace

このリポジトリはデータ分析コンペ用の実験管理テンプレートです。
Kaggle だけでなく、grand-challenge.org / CodaBench / 独自プラットフォームなど **Kaggle 以外のコンペにも対応** する想定で運用します。
**判断に迷ったら `competition_DIRECTION.md` の設計意図を確認すること。**

## コンペ開始時の把握フェーズ（学習コードを書く前に必ず実施）

新しいコンペに取り組むときは、いきなり学習コードに入らず、まず以下を **`survey/competition/` 配下に整理**してから実装に着手する:

1. **プラットフォーム特定**: Kaggle か / grand-challenge.org か / CodaBench か / 独自サイトか
2. **タスク定義**: 入出力、クラス数、評価単位（画像単位/ピクセル単位/患者単位 など）
3. **データの所在**: ダウンロード元 URL、サイズ、フォーマット、ライセンス、配置先のローカルパス
4. **評価指標**: 正確な定義（per-class/macro/micro、背景クラスの扱い、など実装上の曖昧さを潰す）
5. **提出形式**: CSV か / 予測ファイル（画像・JSON）の zip か / Docker コンテナか
6. **タイムライン**: Validation phase / Test phase / 最終締切、提出回数制限
7. **ルール**: チーム人数、外部データ可否、事前学習モデル可否、商用ライセンス

この 7 項目を埋めないまま実装を始めない。

## アイデア提案の原則（堅実＋爆発）

**アプローチやアイデアを提案するときは、必ず「堅実案」と「爆発案」の両方を出すこと。**

- **堅実案**: 既知の手法、定石、段階的改善。確実にスコアが上がる見込みがあるもの
- **爆発案**: 常識外れ、異分野からの転用、誰もやらなそうなアプローチ。失敗リスクは高いが当たれば大きいもの

例:
```
堅実: encoder を efficientnet_b0 → b4 に変更（+0.5%程度の改善見込み）
爆発: セグメンテーションを捨てて物体検出で解く / 全く別のモダリティの事前学習済みモデルを転用
```

局所解に陥らないために、爆発案は「それは普通やらないだろう」くらいがちょうどいい。

## 基本ルール

- 実験は `workspace/` 以下で行う
- Claude用: `expA00_baseline` (アルファベット+数字2桁)、人間用: `exp200_name` (数字3桁)
- 各実験フォルダには必ず `SESSION_NOTES.md` を作成する
- 大きく方針が変わる時だけ新しいexp番号にする。微調整は同じフォルダ内で
- 結果・知見・戦略はすべて日報 `daily_reports/YYYYMMDD.md` に集約する（個別planファイルは作らない）

## 学習コードの鉄則

- **AMP (Mixed Precision) は常にON** (`precision: 16-mixed`)
- **チェックポイント再開は必須** (`save_last=True` + `ckpt_path`)
- **シード固定** (`pl.seed_everything(seed, workers=True)`)
- ハイパーパラメータはすべてconfigで管理（ハードコーディング禁止）
- **ログはPythonの `logging` モジュールで出力する**（`print`禁止）
  - コンソール（INFO）とファイル（DEBUG）の両方に出力する
  - ログファイルは `results/{experiment_name}/foldN/` にタイムスタンプ付きで保存する
  - フォーマット: `%(asctime)s | %(levelname)s | %(message)s`
- **全出力は `results/{experiment_name}/foldN/` に集約する**
  - best_model, checkpoint, log, training_log.json, **config.yaml** を全て同一ディレクトリに保存する
  - **config.yamlは学習開始時に自動コピーされる**（再現性のため）
  - `experiment.name` をconfigに定義し、パラメータ変更時は名前を変える
  - 同名の実験ディレクトリが存在する場合は `_001`, `_002` とナンバリングして上書きしない
- 実行は `run.sh` 経由で行う（各実験フォルダに `run.sh` を作成し、学習・推論はすべてこのスクリプト経由）
- **日報 `daily_reports/YYYYMMDD.md` がすべての記録の中心**
  - 1日1ファイル。戦略・ロードマップ・知見・実験結果をすべてここに書く
  - **知見が出たら都度追記する**（学習完了、エラー発見、スコア変化など、待たずに即記録）
  - 実験結果（CVスコア等）は数値で明記する
  - 過去分は編集せず履歴として保持
  - **セッション開始時に最新の日報を読んで状況を把握する**
  - テンプレート:
    ```markdown
    # 日報 YYYY-MM-DD

    ## コンペ情報
    - **コンペ**: コンペ名
    - **締切**: YYYY-MM-DD（残りN日）
    - **評価指標**:
    - **LB状況**:

    ## 今日やったこと
    ### 1. ...

    ## 数値まとめ
    | 実験 | データ量 | CV | LB | 状態 |
    |------|---------|-----|-----|------|

    ## 戦略・ロードマップ

    ## 判断・知見
    <!-- 今日得られた重要な判断や知見 -->

    ## データ在庫
    <!-- 利用可能なデータソースの整理 -->

    ## 次にやること
    - [ ] TODO
    ```

## Fold設計（最重要）

**安易にランダムKFoldにしない。データの性質を先に確認する。**

- 時系列 → TimeSeriesSplit
- グループ構造あり → GroupKFold
- クラス不均衡 → StratifiedKFold
- グループ+不均衡 → StratifiedGroupKFold
- fold設計の理由と各foldの分布はSESSION_NOTES.mdに記録する
- CV/LBの相関を確認し、相関が弱ければfold設計を見直す

**fold割り当ての永続化（`workspace/fold/`）:**
- fold割り当ては `workspace/fold/{version}/folds.csv` に保存し、全実験で共有する
- `generate_folds.py` で生成。バージョン管理付き
- 前処理やデータを変更した場合は新バージョンを作る。古いバージョンは削除しない
- config.yamlの `cv.folds_csv` で使用バージョンを指定する
- 設計意図・切り方の詳細は `workspace/fold/README.md` に記載

## 提出パイプライン（`submit/`）

提出用コードとモデルは `submit/` 以下で管理する。提出形式はプラットフォームによって異なるため、以下のいずれかに沿って構成する。

### 共通ルール（プラットフォーム非依存）

- **命名**: `v001_説明`, `v002_説明`, ... の連番形式
- **自己完結**: 提出用スクリプトは前処理・後処理を内部に持ち、`workspace/` の学習コードに依存しない
- **環境判別**: 実行環境（Kaggle / grand-challenge / ローカル）を自動判別してパスを切り替える
- **推論パラメータ**: ファイル冒頭の定数で管理（ハードコーディング散在を避ける）
- **出典記録**: スクリプト冒頭の docstring に `Source: workspace/expXXX/.../best_model/` とコピー元パス・CV スコアを明記する
- **モデル取り扱い**: `workspace/` から `best_model/` をコピー。`model/` は git 管理外（`.gitignore`）。サイズが大きいため
- **ローカルテスト必須**: 提出物（CSV / 予測ファイル / Docker イメージ）をローカルで生成まで確認してから提出する
- **提出物の検証**: プラットフォームの要件（ファイル数・命名規則・値の範囲・欠損）を提出前に必ずチェック
- **アップロードは手動**: Kaggle Dataset / grand-challenge Submission ページ / その他への実アップロードはユーザーが手動で行う（Claude は実行しない）
- **提出履歴**: `submit/SUBMISSIONS.md` に全提出を記録する。実験フォルダ、モデル元パス、fold 定義、学習データ、学習/推論パラメータ、前処理、CV/LB スコアを必ず記載

### Kaggle の場合（`submission.csv` 提出型）

```
submit/v001_baseline/
├── notebook.py          # Kaggle Notebook にそのまま貼れる推論スクリプト
└── model/               # 学習済みモデル一式
```

- エントリポイントは `notebook.py`
- 出力は `submission.csv`。**行数・カラム名・欠損・値の範囲**を必ず検証してから提出
- git には `notebook.py` のみコミット

### Kaggle 以外の場合（grand-challenge.org / CodaBench / 独自サイト など）

提出形式は主に次の 2 タイプ。コンペ仕様に合わせて選ぶ。

**(A) 予測ファイル提出型**（画像・マスク・JSON を zip してアップロード）

```
submit/v001_baseline/
├── predict.py           # 入力ファイル群を読んで予測を出力するスクリプト
├── run.sh               # predict.py → 出力検証 → zip 化を一発で行う
├── requirements.txt     # 再現性のため固定
├── model/               # 学習済みモデル一式（.gitignore）
└── output/              # 生成された提出物（.gitignore）
```

- 入出力ディレクトリは `predict.py` 冒頭の定数で指定（`INPUT_DIR` / `OUTPUT_DIR`）
- 出力は **ファイル名・サイズ・dtype・値域** まで厳密に公式仕様へ一致させる
- 提出前に「入力ファイル数 == 出力ファイル数」「stem 名の一致」を assert

**(B) Docker コンテナ提出型**（algorithm container として提出）

```
submit/v001_baseline/
├── Dockerfile
├── process.py           # grand-challenge の algorithm interface を実装
├── requirements.txt
├── test/                # ローカル検証用のサンプル入出力
│   ├── input/
│   └── expected_output/
├── build.sh             # docker build
├── test.sh              # ローカルでコンテナを回して出力を検証
├── export.sh            # docker save → tar.gz（提出物）
└── model/               # .gitignore
```

- プラットフォーム側の I/O 契約（入力パス / 出力パス / ファイル形式）を最優先で守る
- ビルド後に必ず `test.sh` でローカル回帰テスト。本番と同じ入出力パスで動くことを確認
- イメージサイズ・GPU 要件・推論時間制限を事前に把握し、必要ならモデル軽量化や ONNX 化を検討
- gitには `Dockerfile` / `process.py` / スクリプト類のみコミット（`model/` と `test/` の大容量データは除外）

## エラー分析の原則（スコアの前に出力を見ろ）

**スコアを上げようとする前に、まず出力を観察して「何が悪いか」を特定する。**

- 実験後は必ず prediction vs ground truth を目視確認する（最低20件）
- エラーの種類を分類し、パターンを把握する（分類はタスク依存。事前にリスト化せず、出力から帰納的に発見する）
- エラーの種類に応じた対策を打つ（前処理・後処理・推論戦略・学習データ・モデル変更）
- スコアという数字だけ見てパラメータを変える盲目的な探索をしない

順序: 出力を読む → 何が悪いか特定 → 原因に対処する → スコアで検証

## 前処理・評価

- 正規化はtrainデータの統計量で計算する（testの情報を使わない）
- コンペの評価指標を正確に再現する（既存実装のパラメータも確認）
- Augmentationはまず弱めで、過学習が確認されてから強める
- single modelのCV/LBを記録してからアンサンブルする
- 提出前の検証はプラットフォームに応じて行う:
  - Kaggle (CSV): 行数・カラム名・欠損値・値の範囲
  - 予測ファイル (画像/JSON): ファイル数・命名規則・dtype・値域・サイズ
  - Docker: ローカル回帰テスト、入出力パス、GPU/時間制限

## リファレンスコード

- `reference/` に2.5Dセグメンテーションのテンプレートコード（PyTorch Lightning + timm + smp）がある
- 新しい実験のベースとして活用すること
- 詳細は `reference/README.md` を参照

## 利用可能なSkills

- `/survey-papers [キーワード]` - 論文・解法調査（メインコンテキストを汚さず別コンテキストで実行）
- `retrospective-codify` - 先に知っておくべきだった知見をast-grep ルール / skill / CLAUDE.mdなどに言語化する
- `emprical-prompt-tuning` - sub agentが上手く動くようなagent 向けテキスト指示を最適化する

## Custom Agents

状況に応じて自動的にサブエージェントに委譲される。並列実行も可能。

- **kaggle-researcher** (sonnet) - 論文・類似コンペ解法・ディスカッション調査。Kaggle に限らず grand-challenge.org や CodaBench など他プラットフォームのコンペ調査にも使う
- **data-analyst** (sonnet) - EDA・可視化・特徴量分析。データの全体像把握に
- **code-reviewer** (sonnet) - ML/DLコード品質レビュー。読み取り専用で安全