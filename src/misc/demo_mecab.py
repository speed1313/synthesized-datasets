import MeCab

# MeCabのTaggerを初期化
tagger = MeCab.Tagger("-d /home/share/usr/lib/mecab/dic/mecab-ipadic-neologd")


# 判定するための関数
def contains_pronoun(text):
    # MeCabで形態素解析を実行
    parsed_text = tagger.parse(text)

    # 解析結果を行ごとに分割
    for line in parsed_text.splitlines():
        print(line)
        if line == "EOS":
            break
        # タブで区切られた形態素とその情報を取得
        word_info = line.split("\t")
        if len(word_info) > 1:
            # 詳細情報をコンマで分割（品詞など）
            features = word_info[1].split(",")
            # 代名詞かどうかを判定（品詞が "代名詞" であるかを確認）
            print(features)
            if features[0] == "代名詞":
                return True
    return False


# テスト用の文字列
# text = "この本は面白いです。"
# text = "1942年9月、日本軍はガダルカナル島ヘンダーソン飛行場に対して大規模な総攻撃を行った。これは失敗に終わり、日本軍は何を実施した？"
text = "この文章で歌われている曲のタイトルは何ですか？"

# 代名詞が含まれているか確認
if contains_pronoun(text):
    print("代名詞が含まれています。")
else:
    print("代名詞は含まれていません。")
