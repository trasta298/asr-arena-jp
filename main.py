import json
import random
import sqlite3
import threading
from datetime import datetime

import gradio as gr

local = threading.local()


def get_db():
    if not hasattr(local, "db"):
        local.db = sqlite3.connect('db/experiment.db', check_same_thread=False)
    return local.db


def get_cursor():
    return get_db().cursor()


def init_db():
    with get_db() as conn:
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS votes
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      user_id TEXT,
                      audio_id TEXT,
                      original_text TEXT,
                      model_a TEXT,
                      model_b TEXT,
                      model_a_output TEXT,
                      model_b_output TEXT,
                      winner TEXT,
                      timestamp DATETIME)''')
        conn.commit()


def load_user_data(user_id):
    try:
        with open(f'data/user_{user_id:03d}.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def get_next_sample(data, current_index):
    if current_index >= len(data):
        return None, None, None, current_index

    sample = data[current_index]
    models = list(sample['model_outputs'].keys())
    model_a, model_b = random.sample(models, 2)
    return sample, model_a, model_b, current_index + 1


def vote_and_record(user_id, audio_id, original_text, model_a, model_b,
                    model_a_output, model_b_output, winner):
    user_id_int = int(user_id)
    with get_db() as conn:
        c = conn.cursor()
        c.execute("""
            INSERT INTO votes
            (user_id, audio_id, original_text, model_a, model_b,
             model_a_output, model_b_output, winner, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (user_id_int, audio_id, original_text, model_a, model_b,
              model_a_output, model_b_output, winner,
              datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        conn.commit()


def launch_experiment():
    with gr.Blocks(title="音声文字起こし評価実験") as demo:
        # ログインコンテナ
        with gr.Column(variant="panel") as login_container:
            gr.Markdown("# 🎙️ 音声文字起こし評価実験")
            with gr.Row():
                with gr.Column():
                    user_id = gr.Number(label="ユーザーID（1-25）",
                                        minimum=1, maximum=25, step=1,
                                        value=1)
                    password = gr.Textbox(label="パスワード", type="password")
                    login_btn = gr.Button("ログイン", variant="primary")
                    error_msg = gr.Markdown(visible=False)

        # 実験コンテナ（最初は非表示）
        experiment_container = gr.Column(visible=False)

        def login(user_id, password):
            correct_password = "experiment2024"

            if not user_id or user_id < 1 or user_id > 25:
                return {
                    experiment_container: gr.update(visible=False),
                    login_container: gr.update(visible=True),
                    error_msg: gr.update(visible=True, value="❌ ユーザーIDは1から25の間で入力してください。")
                }

            if password != correct_password:
                return {
                    experiment_container: gr.update(visible=False),
                    login_container: gr.update(visible=True),
                    error_msg: gr.update(visible=True, value="❌ パスワードが正しくありません。")
                }

            # ログイン成功
            return {
                experiment_container: gr.update(visible=True),
                login_container: gr.update(visible=False),
                error_msg: gr.update(visible=False)
            }

        login_btn.click(
            login,
            inputs=[user_id, password],
            outputs=[experiment_container, login_container, error_msg]
        )

        # 実験インターフェースを実験コンテナ内に配置
        with experiment_container:
            user_data = None
            sample = None
            model_a = None
            model_b = None
            current_index = 0

            def init_experiment(user_id_value):
                nonlocal user_data, sample, model_a, model_b, current_index
                user_id_int = int(user_id_value)
                user_data = load_user_data(user_id_int)
                current_index = 0
                if user_data:
                    sample, model_a, model_b, current_index = get_next_sample(user_data, current_index)
                    if sample is None:
                        return None
                    total_samples = len(user_data)
                    return {
                        audio: gr.update(value=sample['audio_path']),
                        model_a_output: sample['model_outputs'][model_a],
                        model_b_output: sample['model_outputs'][model_b],
                        progress_text: f"進捗状況: {current_index}/{total_samples}"
                    }
                return None

            gr.Markdown(lambda: f"# 🎙️ 音声文字起こし評価実験（ユーザーID: {int(user_id.value):03d}）")
            progress_text = gr.Markdown("進捗状況: 0/0")
            gr.Markdown("### 👂 以下の音声を聞いて、より正確な文字起こし結果を選んでください")

            with gr.Row():
                audio = gr.Audio(label="音声")

            with gr.Row():
                with gr.Column(scale=1):
                    model_a_output = gr.Textbox(label="文字起こし結果 A", lines=3)
                    model_a_btn = gr.Button("👈️ Aの方が良い", size="lg", variant="primary")

                with gr.Column(scale=1):
                    model_b_output = gr.Textbox(label="文字起こし結果 B", lines=3)
                    model_b_btn = gr.Button("👉 Bの方が良い", size="lg", variant="primary")

            next_btn = gr.Button("➡️ 次の音声へ", size="lg", variant="secondary", visible=False)
            result_text = gr.Markdown(visible=False)

            def vote_and_show_next(choice):
                nonlocal sample, model_a, model_b
                if not sample or not model_a or not model_b:
                    return None
                winner = model_a if choice == "A" else model_b
                vote_and_record(
                    int(user_id.value), sample['id'], sample['original_text'],
                    model_a, model_b,
                    sample['model_outputs'][model_a],
                    sample['model_outputs'][model_b],
                    winner
                )
                return {
                    model_a_btn: gr.update(visible=False),
                    model_b_btn: gr.update(visible=False),
                    next_btn: gr.update(visible=True),
                    result_text: gr.update(visible=True,
                                           value="ご評価ありがとうございます。次の音声に進むには「次の音声へ」を押してください。")
                }

            def load_next_sample():
                nonlocal sample, model_a, model_b, current_index
                sample, model_a, model_b, current_index = get_next_sample(user_data, current_index)
                if user_data is None:
                    return None
                total_samples = len(user_data)

                if sample is None:
                    return {
                        audio: gr.update(value=None),
                        model_a_output: "",
                        model_b_output: "",
                        model_a_btn: gr.update(visible=False),
                        model_b_btn: gr.update(visible=False),
                        next_btn: gr.update(visible=False),
                        result_text: gr.update(visible=True, value="🎉 すべての評価が完了しました！ご協力ありがとうございました。"),
                        progress_text: f"進捗状況: {total_samples}/{total_samples}"
                    }

                return {
                    audio: gr.update(value=sample['audio_path']),
                    model_a_output: sample['model_outputs'][model_a],
                    model_b_output: sample['model_outputs'][model_b],
                    model_a_btn: gr.update(visible=True),
                    model_b_btn: gr.update(visible=True),
                    next_btn: gr.update(visible=False),
                    result_text: gr.update(visible=False),
                    progress_text: f"進捗状況: {current_index}/{total_samples}"
                }

            # イベントハンドラの設定
            login_btn.click(fn=init_experiment, inputs=[user_id],
                            outputs=[audio, model_a_output, model_b_output, progress_text])
            model_a_btn.click(fn=vote_and_show_next, inputs=gr.State("A"),
                              outputs=[model_a_btn, model_b_btn, next_btn, result_text])
            model_b_btn.click(fn=vote_and_show_next, inputs=gr.State("B"),
                              outputs=[model_a_btn, model_b_btn, next_btn, result_text])
            next_btn.click(fn=load_next_sample,
                           outputs=[audio, model_a_output, model_b_output,
                                    model_a_btn, model_b_btn, next_btn, result_text,
                                    progress_text])

    demo.launch(server_port=5730, server_name="0.0.0.0")


if __name__ == "__main__":
    init_db()
    launch_experiment()
