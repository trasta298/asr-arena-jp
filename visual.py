import sqlite3
import pandas as pd


def visualize_results():
    # データベースに接続
    conn = sqlite3.connect('db/experiment.db')

    # 投票結果を取得
    query = """
    SELECT
        user_id,
        audio_id,
        original_text,
        model_a,
        model_b,
        model_a_output,
        model_b_output,
        winner,
        timestamp
    FROM votes
    ORDER BY user_id, timestamp
    """

    # pandasデータフレームとして読み込み
    df = pd.read_sql_query(query, conn)

    # 全データを表示
    print("\n=== 全投票データ ===")
    pd.set_option('display.max_rows', None)  # すべての行を表示
    pd.set_option('display.max_columns', None)  # すべての列を表示
    pd.set_option('display.width', None)  # 表示幅の制限を解除
    pd.set_option('display.max_colwidth', None)  # 列の幅の制限を解除
    print(df)

    # 基本的な統計情報を表示
    print("\n=== 投票結果の概要 ===")
    print(f"総投票数: {len(df)}")
    print(f"ユニークユーザー数: {df['user_id'].nunique()}")

    # モデルごとの勝利数を集計
    model_wins = df.apply(lambda row: row['winner'], axis=1).value_counts()
    print("\n=== モデルごとの勝利数 ===")
    print(model_wins)

    # ユーザーごとの投票数を表示
    print("\n=== ユーザーごとの投票数 ===")
    print(df['user_id'].value_counts().sort_index())

    # データフレームを保存（オプション）
    df.to_csv('results.csv', index=False)

    conn.close()
    return df


if __name__ == "__main__":
    df = visualize_results()
