import json
import random
import sqlite3
import threading

import gradio as gr
import numpy as np
import plotly.graph_objs as go

local = threading.local()


def get_db():
    if not hasattr(local, "db"):
        local.db = sqlite3.connect('db/asr_arena.db', check_same_thread=False)
    return local.db


def get_cursor():
    return get_db().cursor()


def init_db():
    with get_db() as conn:
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS votes_asr
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      audio_id TEXT,
                      winner TEXT,
                      loser TEXT)''')
        c.execute('''CREATE TABLE IF NOT EXISTS votes_translation
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      audio_id TEXT,
                      winner TEXT,
                      loser TEXT)''')
        c.execute('''CREATE TABLE IF NOT EXISTS elo_scores_asr
                     (model TEXT PRIMARY KEY,
                      score REAL)''')
        c.execute('''CREATE TABLE IF NOT EXISTS elo_scores_translation
                     (model TEXT PRIMARY KEY,
                      score REAL)''')
        conn.commit()


init_db()

with open('data/asr.json', 'r', encoding='utf-8') as f:
    asr_data = json.load(f)


def get_random_sample(arena_type):
    if arena_type == "asr":
        data = asr_data

    sample = random.choice(data)
    models = list(sample['model_outputs'].keys())
    model_a, model_b = random.sample(models, 2)
    return sample, model_a, model_b


def update_elo(model_a, model_b, result, table_name):
    K = 32
    with get_db() as conn:
        c = conn.cursor()
        c.execute(f"SELECT score FROM {table_name} WHERE model = ?", (model_a,))
        score_a = c.fetchone()
        if score_a is None:
            c.execute(f"INSERT INTO {table_name} (model, score) VALUES (?, 1500)", (model_a,))
            score_a = 1500
        else:
            score_a = score_a[0]

        c.execute(f"SELECT score FROM {table_name} WHERE model = ?", (model_b,))
        score_b = c.fetchone()
        if score_b is None:
            c.execute(f"INSERT INTO {table_name} (model, score) VALUES (?, 1500)", (model_b,))
            score_b = 1500
        else:
            score_b = score_b[0]

        expected_a = 1 / (1 + 10 ** ((score_b - score_a) / 400))
        expected_b = 1 / (1 + 10 ** ((score_a - score_b) / 400))

        if result == "Skip":
            new_score_a = score_a + K * (0.5 - expected_a)
            new_score_b = score_b + K * (0.5 - expected_b)
        elif result == "Model A":
            new_score_a = score_a + K * (1 - expected_a)
            new_score_b = score_b + K * (0 - expected_b)
        else:  # Model B
            new_score_a = score_a + K * (0 - expected_a)
            new_score_b = score_b + K * (1 - expected_b)

        c.execute(f"UPDATE {table_name} SET score = ? WHERE model = ?", (new_score_a, model_a))
        c.execute(f"UPDATE {table_name} SET score = ? WHERE model = ?", (new_score_b, model_b))
        conn.commit()


def vote(choice, audio_id, model_a, model_b, table_name):
    if choice != "Skip":
        with get_db() as conn:
            c = conn.cursor()
            winner = model_a if choice == "Model A" else model_b
            loser = model_b if choice == "Model A" else model_a
            c.execute(f"INSERT INTO {table_name} (audio_id, winner, loser) VALUES (?, ?, ?)",
                      (audio_id, winner, loser))
            conn.commit()

        update_elo(model_a, model_b, choice, f"elo_scores_{table_name.split('_')[1]}")


def get_leaderboard(table_name):
    arena_type = table_name.split('_')[2]
    with get_db() as conn:
        c = conn.cursor()
        c.execute(f"""
            SELECT e.model, e.score,
                   COUNT(CASE WHEN v.winner = e.model THEN 1 END) as wins,
                   COUNT(CASE WHEN v.loser = e.model THEN 1 END) as losses
            FROM {table_name} e
            LEFT JOIN votes_{arena_type} v ON e.model = v.winner OR e.model = v.loser
            GROUP BY e.model
            ORDER BY e.score DESC
        """)
        return c.fetchall()


def get_total_votes(table_name):
    with get_db() as conn:
        c = conn.cursor()
        c.execute(f"SELECT COUNT(*) FROM {table_name}")
        return c.fetchone()[0]


def bootstrap_elo(votes, n_iterations=1000):
    models = set([vote[1] for vote in votes] + [vote[2] for vote in votes])
    ratings = {model: 1500 for model in models}

    bootstrapped_ratings = {model: [] for model in models}

    for _ in range(n_iterations):
        sample_votes = random.choices(votes, k=len(votes))
        sample_ratings = ratings.copy()

        for _, winner, loser in sample_votes:
            expected_winner = 1 / (1 + 10 ** ((sample_ratings[loser] - sample_ratings[winner]) / 400))
            sample_ratings[winner] += 32 * (1 - expected_winner)  # type: ignore
            sample_ratings[loser] += 32 * (0 - (1 - expected_winner))  # type: ignore

        for model in models:
            bootstrapped_ratings[model].append(sample_ratings[model])

    return bootstrapped_ratings


def calculate_confidence_interval(ratings, confidence=0.95):
    lower = (1 - confidence) / 2
    upper = 1 - lower
    return {
        model: (np.percentile(scores, lower * 100), np.mean(scores), np.percentile(scores, upper * 100))
        for model, scores in ratings.items()
    }


def get_confidence_intervals(table_name):
    arena_type = table_name.split('_')[2]
    with get_db() as conn:
        c = conn.cursor()
        c.execute(f"SELECT audio_id, winner, loser FROM votes_{arena_type}")
        votes = c.fetchall()

    bootstrapped_ratings = bootstrap_elo(votes)
    confidence_intervals = calculate_confidence_interval(bootstrapped_ratings)

    return sorted(confidence_intervals.items(), key=lambda x: x[1][1], reverse=True)


def create_confidence_interval_plot(confidence_intervals):
    models = [ci[0] for ci in confidence_intervals]
    means = [ci[1][1] for ci in confidence_intervals]
    lowers = [ci[1][0] for ci in confidence_intervals]
    uppers = [ci[1][2] for ci in confidence_intervals]

    error_y = [upper - mean for upper, mean in zip(uppers, means)]
    error_y_minus = [mean - lower for lower, mean in zip(lowers, means)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=models,
        y=means,
        error_y=dict(
            type='data',
            symmetric=False,
            array=error_y,
            arrayminus=error_y_minus
        ),
        mode='markers',
        marker=dict(size=10, color='slateblue'),
        name='ELO Rating'
    ))

    fig.update_layout(
        title='📊 Model ELO Ratings with 95% Confidence Intervals',
        xaxis_title='Model',
        yaxis_title='ELO Rating',
        height=600,
        width=714,  # Keep this width to accommodate all models
        margin=dict(l=50, r=50, t=100, b=150),
        xaxis=dict(
            tickangle=45,
            tickmode='array',
            tickvals=list(range(len(models))),
            ticktext=models
        )
    )

    return fig


def get_win_rates(table_name):
    arena_type = table_name.split('_')[2]
    with get_db() as conn:
        c = conn.cursor()
        c.execute(f"""
            SELECT winner, loser, COUNT(*) as battles
            FROM votes_{arena_type}
            GROUP BY winner, loser
        """)
        battles = c.fetchall()

    models = set([b[0] for b in battles] + [b[1] for b in battles])
    win_rates = {model: {opponent: -1 for opponent in models} for model in models}
    battle_counts = {(winner, loser): count for winner, loser, count in battles}

    for model in models:
        for opponent in models:
            if model != opponent:
                win_count = battle_counts.get((model, opponent), 0)
                lose_count = battle_counts.get((opponent, model), 0)
                total = win_count + lose_count
                if total > 0:
                    win_rates[model][opponent] = win_count / total
                    win_rates[opponent][model] = lose_count / total

    return win_rates


def create_win_rate_heatmap(win_rates):
    average_win_rates = {model: np.mean([rate for rate in win_rates[model].values() if rate != -1]) for model in win_rates}
    models = sorted(average_win_rates.keys(), key=lambda x: float(average_win_rates[x]), reverse=True)

    # ヒートマップのデータを作成
    z = [[win_rates[row][col] if win_rates[row][col] != -1 else np.nan for col in models] for row in models]

    colorscale = [
        [0, "red"],
        [0.5, "white"],
        [1, "blue"]
    ]

    hovertemplate = "Model A: %{y}<br>Model B: %{x}<br>Win rate: %{z:.2f}<extra></extra>"

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=models,
        y=models,
        colorscale=colorscale,
        showscale=True,
        zmin=0,
        zmax=1,
        hovertemplate=hovertemplate,
        text=z,
        texttemplate="%{text:.2f}"
    ))

    fig.update_layout(
        title='🏆 Model Win Rates',
        xaxis_title='Model B',
        yaxis_title='Model A',
        height=600,
        width=714,
        xaxis=dict(side='top', tickangle=-45),
        yaxis=dict(autorange='reversed')
    )

    fig.update_layout(margin=dict(l=150, r=100, t=100, b=50))

    return fig


def create_arena(arena_type):
    sample, model_a, model_b = get_random_sample(arena_type)

    with gr.Blocks() as demo:
        gr.Markdown(f"# 🎙️ ASR Arena - {arena_type.capitalize()}")

        audio = gr.Audio(sample['audio_path'])

        gr.Markdown("## Original Text")
        original = gr.Textbox(value=sample['original_text'], label="Original Transcription", lines=3)

        gr.Markdown("## Model Outputs")
        with gr.Row():
            with gr.Column():
                model_a_output = gr.Textbox(value=sample['model_outputs'][model_a], label="Model A Output", lines=5)
                model_a_btn = gr.Button("👈️ A is better", size="lg")
            with gr.Column():
                model_b_output = gr.Textbox(value=sample['model_outputs'][model_b], label="Model B Output", lines=5)
                model_b_btn = gr.Button("👉️ B is better", size="lg")

        skip_btn = gr.Button("🚫 Skip", size="lg")
        next_battle_btn = gr.Button("Next Battle", size="lg", visible=False)

        result_text = gr.Markdown(visible=False)

        gr.Markdown("## Leaderboard")
        leaderboard = gr.Dataframe(get_leaderboard(f"elo_scores_{arena_type}"), headers=["Model", "ELO Score", "Wins", "Losses"])
        total_votes = gr.Textbox(f"Total votes: {get_total_votes(f'votes_{arena_type}')}", label="Overall")
        refresh_leaderboard_btn = gr.Button("Refresh Leaderboard")

        gr.Markdown("## Model Performance Graphs")
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Confidence Intervals")
                confidence_interval_plot = gr.Plot()
            with gr.Column():
                gr.Markdown("### Win Rates")
                win_rate_plot = gr.Plot()

        refresh_graphs_btn = gr.Button("Refresh Graphs")

        def vote_and_show_next(choice):
            nonlocal sample, model_a, model_b
            vote(choice, sample['id'], model_a, model_b, f"votes_{arena_type}")
            if choice == "Skip":
                result = f"You skipped this comparison. The models were:\nModel A: `{model_a}`\nModel B: `{model_b}`"
            else:
                winner = model_a if choice == "Model A" else model_b
                result = f"You voted for `{winner}`. The models were:\nModel A: `{model_a}`\nModel B: `{model_b}`"
            return {
                model_a_btn: gr.update(visible=False),
                model_b_btn: gr.update(visible=False),
                skip_btn: gr.update(visible=False),
                next_battle_btn: gr.update(visible=True),
                result_text: gr.update(visible=True, value=result)
            }

        def load_next_battle():
            nonlocal sample, model_a, model_b
            sample, model_a, model_b = get_random_sample(arena_type)
            return {
                audio: gr.update(value=sample['audio_path']),
                original: sample['original_text'],
                model_a_output: gr.update(value=sample['model_outputs'][model_a], label="Model A Output"),
                model_b_output: gr.update(value=sample['model_outputs'][model_b], label="Model B Output"),
                model_a_btn: gr.update(visible=True, value="👈️ A is better"),
                model_b_btn: gr.update(visible=True, value="👉️ B is better"),
                skip_btn: gr.update(visible=True),
                next_battle_btn: gr.update(visible=False),
                result_text: gr.update(visible=False, value="")
            }

        def refresh_leaderboard():
            return {
                leaderboard: get_leaderboard(f"elo_scores_{arena_type}"),
                total_votes: f"Total votes: {get_total_votes(f'votes_{arena_type}')}"
            }

        def refresh_graphs():
            confidence_intervals = get_confidence_intervals(f"elo_scores_{arena_type}")
            ci_fig = create_confidence_interval_plot(confidence_intervals)

            win_rates = get_win_rates(f"elo_scores_{arena_type}")
            wr_fig = create_win_rate_heatmap(win_rates)

            return {
                confidence_interval_plot: ci_fig,
                win_rate_plot: wr_fig
            }

        model_a_btn.click(vote_and_show_next, inputs=gr.State("Model A"), outputs=[model_a_btn, model_b_btn, skip_btn, next_battle_btn, result_text])
        model_b_btn.click(vote_and_show_next, inputs=gr.State("Model B"), outputs=[model_a_btn, model_b_btn, skip_btn, next_battle_btn, result_text])
        skip_btn.click(vote_and_show_next, inputs=gr.State("Skip"), outputs=[model_a_btn, model_b_btn, skip_btn, next_battle_btn, result_text])
        next_battle_btn.click(load_next_battle, outputs=[audio, original, model_a_output, model_b_output, model_a_btn, model_b_btn, skip_btn, next_battle_btn, result_text])

        refresh_leaderboard_btn.click(refresh_leaderboard, outputs=[leaderboard, total_votes])
        refresh_graphs_btn.click(refresh_graphs, outputs=[confidence_interval_plot, win_rate_plot])

    return demo


def asr_arena():
    with gr.Blocks(title="ASR Arena") as demo:
        gr.Markdown("# ASR Arena")

        with gr.Tabs():
            with gr.TabItem("Japanese ASR"):
                create_arena("asr")

    demo.launch(
        share=False,
        server_name="0.0.0.0",
        auth=("tokyotech", "siencetokyo"),
    )


if __name__ == "__main__":
    asr_arena()
