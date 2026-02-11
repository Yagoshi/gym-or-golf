import streamlit as st
import random

# ページの設定
st.set_page_config(page_title="今日の予定", page_icon="🎯")

# セッション状態の初期化
if 'page' not in st.session_state:
    st.session_state.page = 'select'
if 'button_position' not in st.session_state:
    st.session_state.button_position = 50

def go_to_result():
    st.session_state.page = 'result'

def move_button():
    # ボタンの位置をランダムに変更
    st.session_state.button_position = random.randint(0, 100)

# 選択ページ
if st.session_state.page == 'select':
    st.title("今日は何する？🤔")
    st.write("あなたの選択は...")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("⛳ ゴルフ行く", key="golf", use_container_width=True):
            go_to_result()
            st.rerun()
    
    with col2:
        if st.button("💪 ジム行く", key="gym", use_container_width=True):
            go_to_result()
            st.rerun()
    
    with col3:
        # 逃げ回るボタンのスタイル
        st.markdown(f"""
        <style>
        div[data-testid="column"]:nth-child(3) button {{
            position: relative;
            transition: all 0.3s ease;
        }}
        </style>
        """, unsafe_allow_html=True)
        
        if st.button("🏠 家でゴロゴロ", key="home", on_click=move_button, use_container_width=True):
            pass
    
    # マウスオーバーで逃げる効果をJavaScriptで実装
    st.markdown("""
    <script>
    const buttons = window.parent.document.querySelectorAll('button');
    buttons.forEach(button => {
        if (button.textContent.includes('家でゴロゴロ')) {
            button.addEventListener('mouseenter', function() {
                const x = Math.random() * 200 - 100;
                const y = Math.random() * 200 - 100;
                this.style.transform = `translate(${x}px, ${y}px)`;
            });
        }
    });
    </script>
    """, unsafe_allow_html=True)

# 結果ページ
elif st.session_state.page == 'result':
    st.balloons()
    st.title("🎉 そうだと思ったよ！")
    st.write("### やっぱり動く方を選んだね！")
    st.write("健康的な選択、素晴らしい！👍")
    
    if st.button("もう一度選ぶ"):
        st.session_state.page = 'select'
        st.rerun()
