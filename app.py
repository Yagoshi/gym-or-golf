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
                position: relative;
                width: 100%;
                height: 600px;
                border: 2px solid #e0e0e0;
                border-radius: 12px;
                background: linear-gradient(135deg, #fdfbfb 0%, #ebedee 100%);
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
                position: absolute;
            }
            .button:hover {
                transform: translateY(-2px) scale(1.05);
                box-shadow: 0 6px 12px rgba(0,0,0,0.15);
            }
            .button:active {
                transform: translateY(0px) scale(1);
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
                transition: all 0.3s cubic-bezier(0.68, -0.55, 0.265, 1.55);
                z-index: 2;
                pointer-events: none;
                animation: pulse 1s infinite alternate;
            }
            @keyframes pulse {
                0% { box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
                100% { box-shadow: 0 8px 16px rgba(79, 172, 254, 0.4); }
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
                }
                .message {
                    font-size: 16px;
                    padding: 12px 24px;
                }
                .container {
                    height: 500px;
                }
            }
        </style>
    </head>
    <body>
        <div id="message" class="message"></div>
        <div class="container" id="container">
            <button class="button golf" id="golfBtn" onclick="selectOption('golf')">⛳ ゴルフ行く</button>
            <button class="button gym" id="gymBtn" onclick="selectOption('gym')">💪 ジム行く</button>
            <button class="button lazy" id="lazyBtn">🏠 家でゴロゴロ</button>
        </div>

        <script>
            const lazyBtn = document.getElementById('lazyBtn');
            const golfBtn = document.getElementById('golfBtn');
            const gymBtn = document.getElementById('gymBtn');
            const container = document.getElementById('container');
            const messageDiv = document.getElementById('message');
            let attempts = 0;
            let autoMoveInterval;
            let lastMoveTime = 0;
            
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

            // 固定ボタンの位置を設定
            function setFixedButtonPositions() {
                const containerRect = container.getBoundingClientRect();
                const buttonWidth = 180;
                const buttonHeight = 60;
                
                // ゴルフボタン（左下）
                golfBtn.style.left = '50px';
                golfBtn.style.top = (containerRect.height - buttonHeight - 50) + 'px';
                
                // ジムボタン（右下）
                gymBtn.style.left = (containerRect.width - buttonWidth - 50) + 'px';
                gymBtn.style.top = (containerRect.height - buttonHeight - 50) + 'px';
            }

            // ボタンが重なっているかチェック
            function isOverlapping(rect1, rect2, margin = 30) {
                return !(rect1.right + margin < rect2.left || 
                        rect1.left - margin > rect2.right || 
                        rect1.bottom + margin < rect2.top || 
                        rect1.top - margin > rect2.bottom);
            }

            // 重ならない位置を見つける
            function findNonOverlappingPosition() {
                const containerRect = container.getBoundingClientRect();
                const btnWidth = lazyBtn.offsetWidth || 200;
                const btnHeight = lazyBtn.offsetHeight || 60;
                const margin = 40;
                
                let attempts = 0;
                let newX, newY;
                
                do {
                    newX = margin + Math.random() * (containerRect.width - btnWidth - margin * 2);
                    newY = margin + Math.random() * (containerRect.height - btnHeight - margin * 2);
                    
                    const lazyRect = {
                        left: newX,
                        right: newX + btnWidth,
                        top: newY,
                        bottom: newY + btnHeight
                    };
                    
                    const golfRect = golfBtn.getBoundingClientRect();
                    const gymRect = gymBtn.getBoundingClientRect();
                    
                    const golfRelative = {
                        left: golfRect.left - containerRect.left,
                        right: golfRect.right - containerRect.left,
                        top: golfRect.top - containerRect.top,
                        bottom: golfRect.bottom - containerRect.top
                    };
                    
                    const gymRelative = {
                        left: gymRect.left - containerRect.left,
                        right: gymRect.right - containerRect.left,
                        top: gymRect.top - containerRect.top,
                        bottom: gymRect.bottom - containerRect.top
                    };
                    
                    if (!isOverlapping(lazyRect, golfRelative, 50) && 
                        !isOverlapping(lazyRect, gymRelative, 50)) {
                        return { x: newX, y: newY };
                    }
                    
                    attempts++;
                } while (attempts < 50);
                
                // 50回試して見つからなければ中央上部に配置
                return {
                    x: containerRect.width / 2 - btnWidth / 2,
                    y: margin
                };
            }

            // 常時自動で動く
            function autoMove() {
                const now = Date.now();
                if (now - lastMoveTime < 300) return; // 300ms以内の連続移動を防ぐ
                
                lastMoveTime = now;
                const pos = findNonOverlappingPosition();
                lazyBtn.style.left = pos.x + 'px';
                lazyBtn.style.top = pos.y + 'px';
            }

            // ユーザーの入力に反応して動く
            function handleMove(clientX, clientY) {
                const btnRect = lazyBtn.getBoundingClientRect();
                const btnCenterX = btnRect.left + btnRect.width / 2;
                const btnCenterY = btnRect.top + btnRect.height / 2;
                
                const distance = Math.sqrt(
                    Math.pow(clientX - btnCenterX, 2) + 
                    Math.pow(clientY - btnCenterY, 2)
                );
                
                // 250px以内に近づいたら即座に逃げる
                if (distance < 250) {
                    attempts++;
                    showMessage();
                    moveAwayFrom(clientX, clientY);
                }
            }

            // 特定の位置から逃げる
            function moveAwayFrom(inputX, inputY) {
                const containerRect = container.getBoundingClientRect();
                const btnRect = lazyBtn.getBoundingClientRect();
                
                const btnCenterX = btnRect.left + btnRect.width / 2 - containerRect.left;
                const btnCenterY = btnRect.top + btnRect.height / 2 - containerRect.top;
                
                const inputRelativeX = inputX - containerRect.left;
                const inputRelativeY = inputY - containerRect.top;
                
                const angle = Math.atan2(btnCenterY - inputRelativeY, btnCenterX - inputRelativeX);
                
                const moveDistance = 250 + Math.random() * 100;
                
                let newX = btnCenterX + Math.cos(angle) * moveDistance - btnRect.width / 2;
                let newY = btnCenterY + Math.sin(angle) * moveDistance - btnRect.height / 2;
                
                const margin = 40;
                const maxX = containerRect.width - btnRect.width - margin;
                const maxY = containerRect.height - btnRect.height - margin;
                
                newX = Math.max(margin, Math.min(maxX, newX));
                newY = Math.max(margin, Math.min(maxY, newY));
                
                // 重なりチェック
                const testRect = {
                    left: newX,
                    right: newX + btnRect.width,
                    top: newY,
                    bottom: newY + btnRect.height
                };
                
                const golfRect = golfBtn.getBoundingClientRect();
                const gymRect = gymBtn.getBoundingClientRect();
                
                const golfRelative = {
                    left: golfRect.left - containerRect.left,
                    right: golfRect.right - containerRect.left,
                    top: golfRect.top - containerRect.top,
                    bottom: golfRect.bottom - containerRect.top
                };
                
                const gymRelative = {
                    left: gymRect.left - containerRect.left,
                    right: gymRect.right - containerRect.left,
                    top: gymRect.top - containerRect.top,
                    bottom: gymRect.bottom - containerRect.top
                };
                
                // 重なる場合は別の位置を探す
                if (isOverlapping(testRect, golfRelative, 50) || 
                    isOverlapping(testRect, gymRelative, 50)) {
                    const pos = findNonOverlappingPosition();
                    newX = pos.x;
                    newY = pos.y;
                }
                
                lazyBtn.style.left = newX + 'px';
                lazyBtn.style.top = newY + 'px';
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

            document.addEventListener('touchstart', function(e) {
                if (e.touches.length > 0) {
                    handleMove(e.touches[0].clientX, e.touches[0].clientY);
                }
            });

            function showMessage() {
                const msgIndex = Math.min(attempts - 1, messages.length - 1);
                messageDiv.textContent = messages[msgIndex];
                messageDiv.style.display = 'block';
                
                setTimeout(() => {
                    messageDiv.style.display = 'none';
                }, 1500);
            }

            function selectOption(choice) {
                clearInterval(autoMoveInterval);
                window.parent.postMessage({
                    type: 'streamlit:setComponentValue',
                    value: choice
                }, '*');
            }

            // イベントリスナー
            lazyBtn.addEventListener('click', function(e) {
                e.preventDefault();
                e.stopPropagation();
                attempts++;
                showMessage();
                autoMove();
                return false;
            });

            lazyBtn.addEventListener('touchend', function(e) {
                e.preventDefault();
                e.stopPropagation();
                return false;
            });

            lazyBtn.addEventListener('mouseenter', function(e) {
                attempts++;
                showMessage();
                moveAwayFrom(e.clientX, e.clientY);
            });

            // 初期化
            window.addEventListener('load', function() {
                setFixedButtonPositions();
                const pos = findNonOverlappingPosition();
                lazyBtn.style.left = pos.x + 'px';
                lazyBtn.style.top = pos.y + 'px';
                
                // 0.8秒ごとに自動で動く（素早く）
                autoMoveInterval = setInterval(autoMove, 800);
            });

            window.addEventListener('resize', function() {
                setFixedButtonPositions();
                autoMove();
            });
        </script>
    </body>
    </html>
    """
    
    # コンポーネントを表示
    selected = components.html(html_code, height=650)
    
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
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔄 もう一度チャレンジ", use_container_width=True):
            st.session_state.page = 'select'
            st.rerun()
