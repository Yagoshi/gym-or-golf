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
        <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
                -webkit-tap-highlight-color: transparent;
            }
            body {
                padding: 20px;
                font-family: "Source Sans Pro", sans-serif;
                overflow: hidden;
                touch-action: none;
            }
            .container {
                display: flex;
                justify-content: center;
                align-items: center;
                gap: 20px;
                min-height: 500px;
                position: relative;
                width: 100%;
            }
            @media (max-width: 768px) {
                .container {
                    flex-direction: column;
                    gap: 30px;
                    min-height: 600px;
                }
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
                user-select: none;
                -webkit-user-select: none;
            }
            .button:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 12px rgba(0,0,0,0.15);
            }
            .button:active {
                transform: translateY(0px);
            }
            .golf {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                z-index: 1;
            }
            .gym {
                background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                color: white;
                z-index: 1;
            }
            .lazy {
                background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                color: white;
                position: absolute;
                transition: all 0.15s cubic-bezier(0.68, -0.55, 0.265, 1.55);
                z-index: 2;
                pointer-events: none;
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
                animation: shake 0.5s;
            }
            @keyframes shake {
                0%, 100% { transform: translateX(-50%) rotate(0deg); }
                25% { transform: translateX(-50%) rotate(-5deg); }
                75% { transform: translateX(-50%) rotate(5deg); }
            }
            @media (max-width: 768px) {
                .button {
                    padding: 18px 35px;
                    font-size: 18px;
                    min-width: 200px;
                }
                .message {
                    font-size: 16px;
                    padding: 12px 24px;
                }
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
            let isMoving = false;
            
            const messages = [
                "逃げちゃった😏",
                "捕まえられないよ〜🤣",
                "もう諦めたら？💪",
                "運動しよう！🏃",
                "しつこいなぁ😅",
                "まだやるの？🙄",
                "無理無理！😎",
                "諦めが悪いね😂",
                "ゴロゴロは許さない！💢",
                "健康第一！🌟"
            ];

            // 初期位置を設定
            function setInitialPosition() {
                const containerRect = container.getBoundingClientRect();
                lazyBtn.style.left = (containerRect.width / 2 - lazyBtn.offsetWidth / 2) + 'px';
                lazyBtn.style.top = (containerRect.height / 2 - lazyBtn.offsetHeight / 2) + 'px';
            }

            window.addEventListener('load', setInitialPosition);
            window.addEventListener('resize', setInitialPosition);

            // マウスとタッチの両方に対応
            function handleMove(clientX, clientY) {
                if (isMoving) return;
                
                const btnRect = lazyBtn.getBoundingClientRect();
                const btnCenterX = btnRect.left + btnRect.width / 2;
                const btnCenterY = btnRect.top + btnRect.height / 2;
                
                const distance = Math.sqrt(
                    Math.pow(clientX - btnCenterX, 2) + 
                    Math.pow(clientY - btnCenterY, 2)
                );
                
                // 200px以内に近づいたら逃げる（範囲拡大）
                if (distance < 200) {
                    isMoving = true;
                    attempts++;
                    showMessage();
                    moveButton(clientX, clientY);
                    setTimeout(() => { isMoving = false; }, 150);
                }
            }

            // マウス移動
            document.addEventListener('mousemove', function(e) {
                handleMove(e.clientX, e.clientY);
            });

            // タッチ移動（スマホ対応）
            document.addEventListener('touchmove', function(e) {
                e.preventDefault();
                if (e.touches.length > 0) {
                    handleMove(e.touches[0].clientX, e.touches[0].clientY);
                }
            }, { passive: false });

            // タッチ開始（スマホ対応）
            document.addEventListener('touchstart', function(e) {
                if (e.touches.length > 0) {
                    handleMove(e.touches[0].clientX, e.touches[0].clientY);
                }
            });

            function moveButton(inputX, inputY) {
                const containerRect = container.getBoundingClientRect();
                const btnRect = lazyBtn.getBoundingClientRect();
                
                // 現在のボタン中心位置
                const btnCenterX = btnRect.left + btnRect.width / 2 - containerRect.left;
                const btnCenterY = btnRect.top + btnRect.height / 2 - containerRect.top;
                
                // 入力位置から逃げる角度を計算
                const inputRelativeX = inputX - containerRect.left;
                const inputRelativeY = inputY - containerRect.top;
                
                const angle = Math.atan2(btnCenterY - inputRelativeY, btnCenterX - inputRelativeX);
                
                // 移動距離（ランダム性を追加）
                const moveDistance = 200 + Math.random() * 150;
                
                let newX = btnCenterX + Math.cos(angle) * moveDistance - btnRect.width / 2;
                let newY = btnCenterY + Math.sin(angle) * moveDistance - btnRect.height / 2;
                
                // 画面内に収まるように調整
                const margin = 30;
                const maxX = containerRect.width - btnRect.width - margin;
                const maxY = containerRect.height - btnRect.height - margin;
                
                newX = Math.max(margin, Math.min(maxX, newX));
                newY = Math.max(margin, Math.min(maxY, newY));
                
                // 端に追い詰められたら反対側にワープ
                if (newX <= margin || newX >= maxX || newY <= margin || newY >= maxY) {
                    newX = containerRect.width / 2 - btnRect.width / 2;
                    newY = containerRect.height / 2 - btnRect.height / 2;
                    
                    // さらにランダムにずらす
                    newX += (Math.random() - 0.5) * 150;
                    newY += (Math.random() - 0.5) * 150;
                    
                    newX = Math.max(margin, Math.min(maxX, newX));
                    newY = Math.max(margin, Math.min(maxY, newY));
                }
                
                lazyBtn.style.left = newX + 'px';
                lazyBtn.style.top = newY + 'px';
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
                window.parent.postMessage({
                    type: 'streamlit:setComponentValue',
                    value: choice
                }, '*');
            }

            // ボタンへの直接クリック/タップを完全に無効化
            lazyBtn.addEventListener('click', function(e) {
                e.preventDefault();
                e.stopPropagation();
                attempts++;
                showMessage();
                const rect = lazyBtn.getBoundingClientRect();
                moveButton(rect.left + rect.width / 2, rect.top + rect.height / 2);
                return false;
            });

            lazyBtn.addEventListener('touchend', function(e) {
                e.preventDefault();
                e.stopPropagation();
                return false;
            });

            // マウスオーバーでも逃げる
            lazyBtn.addEventListener('mouseenter', function(e) {
                attempts++;
                showMessage();
                moveButton(e.clientX, e.clientY);
            });
        </script>
    </body>
    </html>
    """
    
    # コンポーネントを表示
    selected = components.html(html_code, height=600)
    
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
    st.write("")
    st.write("家でゴロゴロなんてダメだよ〜💪")
    
    if st.button("🔄 もう一度チャレンジ", use_container_width=True):
        st.session_state.page = 'select'
        st.rerun()
