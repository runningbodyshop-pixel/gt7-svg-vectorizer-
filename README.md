# イラスト SVG ベクター化ツール

Streamlitで動く、画像イラスト向けSVGベクター化アプリです。

## ファイル

- `app.py`
- `requirements.txt`

## 使い方

1. GitHubで新規リポジトリを作成
2. `app.py` と `requirements.txt` を追加
3. Streamlit Community Cloudでデプロイ
4. 画像をアップロードして「SVGに変換する」を押す
5. ZIPをダウンロード

## 変換の考え方

大きな色面から順にSVGパスを作り、上に小さなパーツと線画を重ねます。
完全なAI意味分解ではなく、色・面積・輪郭ベースの自動分割です。
