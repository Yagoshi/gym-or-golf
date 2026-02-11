import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="今日の予定", page_icon="🎯", layout="wide")

# セッション状態の初期化
if 'page' not in st.session_state:
    st.session_state.page = 'select'

# 選択ページ
if st.session_state.page == 'select':
    st.title("今日は何する？🤔")
    st.write("### あなたの選択は...")
    
    # カスタムHTMLとJavaScriptで逃げるボタンを実装
    html_code = """
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {
                margin: 0;
                padding: 20px;
                font-family: "Source Sans Pro", sans-serif;
            }
            .container {
                display: flex;
                justify-content: center;
                align-items: center;
                gap: 30px;
                min-height: 400px;
                position: relative;
            }
            .button {
                padding: 20px 40px;
                font-size: 20px;
                font-weight: 600;
                border: none;
                border-radius: 8px;
                cursor: pointer;
                transition: all 0.2s ease;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }
            .button:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 12px rgba(0,0,0,0.15);
            }
            .golf {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
            }
            .gym {
                background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                color: white;
            }
            .lazy {
                background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                color: white;
                position: absolute;
                transition: all 0.1s ease-out;
            }
            .message {
                position: fixed;
                top: 20px;
                left: 50%;
                transform: translateX(-50%);
                background: #ff6b6b;
                color: white;
                padding: 15px 30px;
                border-radius: 8px;
                font-size: 18px;
                font-weight: 600;
                display: none;
                z-index: 1000;
                box-shadow: 0 4px 12px rgba(0,0,0,0.2);
            }
        </style>
    </head>
    <body>
        <div id="message" class="message"></div>
        <div class="container" id="container">
            <button class="button golf" onclick="selectOption('golf')">⛳ ゴルフ行く</button>
            <button class="button gym" onclick="selectOption('gym')">💪 ジム行く</button>
            <button class="button lazy" id="lazyBtn">🏠 家でゴロゴロ</button>
        </div>

        <script>
            const lazyBtn = document.getElementById('lazyBtn');
            const container = document.getElementById('container');
            const messageDiv = document.getElementById('message');
            let attempts = 0;
            
            const messages = [
                "逃げちゃった😏",
                "捕まえられないよ〜🤣",
                "もう諦めたら？💪",
                "運動しよう！🏃",
                "しつこいなぁ😅"
            ];

            // 初期位置を設定
            function setInitialPosition() {
                const containerRect = container.getBoundingClientRect();
                lazyBtn.style.left = (containerRect.width / 2 - lazyBtn.offsetWidth / 2) + 'px';
                lazyBtn.style.top = '50%';
                lazyBtn.style.transform = 'translateY(-50%)';
            }

            // ページ読み込み時に初期位置を設定
            window.addEventListener('load', setInitialPosition);
            window.addEventListener('resize', setInitialPosition);

            // カーソルが近づいたときの処理
            document.addEventListener('mousemove', function(e) {
                const btnRect = lazyBtn.getBoundingClientRect();
                const btnCenterX = btnRect.left + btnRect.width / 2;
                const btnCenterY = btnRect.top + btnRect.height / 2;
                
                const distance = Math.sqrt(
                    Math.pow(e.clientX - btnCenterX, 2) + 
                    Math.pow(e.clientY - btnCenterY, 2)
                );
                
                // 150px以内に近づいたら逃げる
                if (distance < 150) {
                    attempts++;
                    showMessage();
                    moveButton(e.clientX, e.clientY);
                }
            });

            function moveButton(mouseX, mouseY) {
                const containerRect = container.getBoundingClientRect();
                const btnRect = lazyBtn.getBoundingClientRect();
                
                // マウスから離れる方向を計算
                const btnCenterX = btnRect.left + btnRect.width / 2;
                const btnCenterY = btnRect.top + btnRect.height / 2;
                
                const angle = Math.atan2(btnCenterY - mouseY, btnCenterX - mouseX);
                
                // 移動距離をランダムに
                const moveDistance = 150 + Math.random() * 100;
                
                let newX = btnCenterX + Math.cos(angle) * moveDistance - containerRect.left;
                let newY = btnCenterY + Math.sin(angle) * moveDistance - containerRect.top;
                
                // 画面内に収まるように調整
                const margin = 20;
                newX = Math.max(margin, Math.min(containerRect.width - btnRect.width - margin, newX));
                newY = Math.max(margin, Math.min(containerRect.height - btnRect.height - margin, newY));
                
                lazyBtn.style.left = newX + 'px';
                lazyBtn.style.top = newY + 'px';
                lazyBtn.style.transform = 'none';
            }

            function showMessage() {
                const msgIndex = Math.min(attempts - 1, messages.length - 1);
                messageDiv.textContent = messages[msgIndex];
                messageDiv.style.display = 'block';
                
                setTimeout(() => {
                    messageDiv.style.display = 'none';
                }, 1500);
            }

            function selectOption(choice) {
                // Streamlitに結果を送信
                window.parent.postMessage({
                    type: 'streamlit:setComponentValue',
                    value: choice
                }, '*');
            }

            // 逃げるボタンがクリックされたとき（万が一捕まえた場合）
            lazyBtn.addEventListener('click', function() {
                attempts++;
                showMessage();
                moveButton(event.clientX, event.clientY);
            });
        </script>
    </body>
    </html>
    """
    
    # コンポーネントを表示
    selected = components.html(html_code, height=500)
    
    # 選択があった場合
    if selected:
        if selected in ['golf', 'gym']:
            st.session_state.page = 'result'
            st.rerun()

# 結果ページ
elif st.session_state.page == 'result':
    st.balloons()
    st.title("🎉 そうだと思ったよ！")
    st.write("### やっぱり動く方を選んだね！")
    st.write("健康的な選択、素晴らしい！👍")
    
    if st.button("もう一度選ぶ"):
        st.session_state.page = 'select'
        st.rerun()
