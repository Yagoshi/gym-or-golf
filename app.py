import streamlit as st
import random

st.set_page_config(page_title="今日の予定", page_icon="🎯")

# セッション状態の初期化
if 'page' not in st.session_state:
    st.session_state.page = 'select'
if 'lazy_attempts' not in st.session_state:
    st.session_state.lazy_attempts = 0

def go_to_result():
    st.session_state.page = 'result'

def try_lazy():
    st.session_state.lazy_attempts += 1

# 選択ページ
if st.session_state.page == 'select':
    st.title("今日は何する？🤔")
    
    # 逃げようとした回数に応じてメッセージを表示
    if st.session_state.lazy_attempts > 0:
        messages = [
            "ダメだよ〜😏",
            "まだ諦めないの？🤣",
            "ゴロゴロはナシ！💪",
            "運動しよう！🏃",
            "もう{}回も試したね...😅".format(st.session_state.lazy_attempts)
        ]
        idx = min(st.session_state.lazy_attempts - 1, len(messages) - 1)
        st.warning(messages[idx])
    
    st.write("### あなたの選択は...")
    
    # ランダムな順序でボタンを配置
    positions = list(range(3))
    random.seed(st.session_state.lazy_attempts)
    random.shuffle(positions)
    
    cols = st.columns(3)
    
    buttons = [
        ("⛳ ゴルフ行く", "golf", go_to_result),
        ("💪 ジム行く", "gym", go_to_result),
        ("🏠 家でゴロゴロ", "home", try_lazy)
    ]
    
    for i, pos in enumerate(positions):
        with cols[i]:
            label, key, callback = buttons[pos]
            if st.button(label, key=f"{key}_{st.session_state.lazy_attempts}", 
                        use_container_width=True, on_click=callback):
                if pos < 2:  # ゴルフかジム
                    st.rerun()

# 結果ページ
elif st.session_state.page == 'result':
    st.balloons()
    st.title("🎉 そうだと思ったよ！")
    st.write("### やっぱり動く方を選んだね！")
    st.write("健康的な選択、素晴らしい！👍")
    
    st.write(f"※ 家でゴロゴロを選ぼうとした回数: **{st.session_state.lazy_attempts}回** 😄")
    
    if st.button("もう一度選ぶ"):
        st.session_state.page = 'select'
        st.session_state.lazy_attempts = 0
        st.rerun()
