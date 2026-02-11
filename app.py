import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="今日の予定", page_icon="🎯", layout="wide")

# セッション状態の初期化
if 'page' not in st.session_state:
    st.session_state.page = 'select'
if 'choice' not in st.session_state:
    st.session_state.choice = None

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
                padding: 10px;
                font-family: "Source Sans Pro", sans-serif;
                overflow: hidden;
                touch-action: none;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            }
            .container {
                position: relative;
                width: 100%;
                height: 100vh;
                max-height: 700px;
                border-radius: 20px;
                background: white;
                box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            }
            .button {
                padding: 22px 45px;
                font-size: 22px;
                font-weight: 700;
                border: none;
                border-radius: 16px;
                cursor: pointer;
                box-shadow: 0 6px 12px rgba(0,0,0,0.15);
                user-select: none;
                -webkit-user-select: none;
                position: absolute;
                transition: all 0.3s cubic-bezier(0.68, -0.55, 0.265, 1.55);
                min-width: 180px;
                text-align: center;
            }
            .button:active {
                transform: scale(0.95);
            }
            .golf {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                z-index: 10;
            }
            .gym {
                background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                color: white;
                z-index: 10;
            }
            .lazy {
                background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                color: white;
                z-index: 5;
                pointer-events: none;
                animation: float 3s ease-in-out infinite;
                box-shadow: 0 8px 20px rgba(79, 172, 254, 0.5);
            }
            @keyframes float {
                0%, 100% { transform: translateY(0px); }
                50% { transform: translateY(-10px); }
            }
            .message {
                position: fixed;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
                color: white;
                padding: 25px 40px;
                border-radius: 20px;
                font-size: 24px;
                font-weight: 700;
                display: none;
                z-index: 1000;
                box-shadow: 0 10px 40px rgba(0,0,0,0.3);
                animation: popIn 0.3s cubic-bezier(0.68, -0.55, 0.265, 1.55);
                text-align: center;
                max-width: 90%;
            }
            @keyframes popIn {
                0% { transform: translate(-50%, -50%) scale(0); }
                100% { transform: translate(-50%, -50%) scale(1); }
            }
            .attempt-counter {
                position: absolute;
                top: 20px;
                left: 50%;
                transform: translateX(-50%);
                background: rgba(255,255,255,0.95);
                padding: 12px 24px;
                border-radius: 30px;
                font-size: 16px;
                font-weight: 600;
                color: #333;
                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                z-index: 100;
            }
            .hint {
                position: absolute;
                bottom: 20px;
                left: 50%;
                transform: translateX(-50%);
                font-size: 14px;
                color: #999;
                text-align: center;
                animation: blink 2s infinite;
            }
            @keyframes blink {
                0%, 100% { opacity: 0.3; }
                50% { opacity: 1; }
            }
            @media (max-width: 768px) {
                .button {
                    padding: 20px 40px;
                    font-size: 20px;
                    min-width: 160px;
                }
                .message {
                    font-size: 20px;
                    padding: 20px 30px;
                }
            }
        </style>
    </head>
    <body>
        <div class="attempt-counter" id="counter">逃げられた回数: 0回</div>
        <div id="message" class="message"></div>
        <div class="container" id="container">
            <button class="button golf" id="golfBtn" onclick="selectOption('golf')">⛳ ゴルフ行く</button>
            <button class="button gym" id="gymBtn" onclick="selectOption('gym')">💪 ジム行く</button>
            <button class="button lazy" id="lazyBtn">🏠 家でゴロゴロ</button>
            <div class="hint">家でゴロゴロを選んでみて...😏</div>
        </div>

        <script>
            const lazyBtn = document.getElementById('lazyBtn');
            const golfBtn = document.getElementById('golfBtn');
            const gymBtn = document.getElementById('gymBtn');
            const container = document.getElementById('container');
            const messageDiv = document.getElementById('message');
            const counter = document.getElementById('counter');
            let attempts = 0;
            let autoMoveInterval;
            let lastMoveTime = 0;
            let touchStartTime = 0;
            
            const escapeMessages = [
                "逃げちゃった😏",
                "遅い遅い！🏃💨",
                "もっと速く！⚡",
                "無理だって〜😂",
                "諦めたら？🤷",
                "まだやるの？😅",
                "しつこい！😤",
                "もう" + (attempts + 1) + "回目だよ？🙄",
                "運動しなよ！💪",
                "ダメダメ〜！✋"
            ];

            // 固定ボタンの位置を設定
            function setFixedButtonPositions() {
                const containerRect = container.getBoundingClientRect();
                const isMobile = window.innerWidth <= 768;
                
                if (isMobile) {
                    // スマホ: 縦に配置
                    golfBtn.style.left = '50%';
                    golfBtn.style.transform = 'translateX(-50%)';
                    golfBtn.style.bottom = '120px';
                    
                    gymBtn.style.left = '50%';
                    gymBtn.style.transform = 'translateX(-50%)';
                    gymBtn.style.bottom = '50px';
                } else {
                    // PC: 横に配置
                    const buttonWidth = 180;
                    const spacing = 30;
                    
                    golfBtn.style.left = spacing + 'px';
                    golfBtn.style.bottom = '50px';
                    golfBtn.style.transform = 'none';
                    
                    gymBtn.style.right = spacing + 'px';
                    gymBtn.style.bottom = '50px';
                    gymBtn.style.left = 'auto';
                    gymBtn.style.transform = 'none';
                }
            }

            // ボタンが重なっているかチェック（余裕を持って）
            function isOverlapping(rect1, rect2, margin = 60) {
                return !(rect1.right + margin < rect2.left || 
                        rect1.left - margin > rect2.right || 
                        rect1.bottom + margin < rect2.top || 
                        rect1.top - margin > rect2.bottom);
            }

            // 重ならない位置を見つける
            function findNonOverlappingPosition() {
                const containerRect = container.getBoundingClientRect();
                const btnWidth = lazyBtn.offsetWidth || 180;
                const btnHeight = lazyBtn.offsetHeight || 70;
                const margin = 50;
                
                let tryCount = 0;
                let newX, newY;
                
                do {
                    // より広い範囲でランダム配置
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
                    
                    if (!isOverlapping(lazyRect, golfRelative, 80) && 
                        !isOverlapping(lazyRect, gymRelative, 80)) {
                        return { x: newX, y: newY };
                    }
                    
                    tryCount++;
                } while (tryCount < 100);
                
                // 100回試して見つからなければ中央上部に配置
                return {
                    x: containerRect.width / 2 - btnWidth / 2,
                    y: margin + 50
                };
            }

            // 常時自動で動く（速く）
            function autoMove() {
                const now = Date.now();
                if (now - lastMoveTime < 200) return;
                
                lastMoveTime = now;
                const pos = findNonOverlappingPosition();
                lazyBtn.style.left = pos.x + 'px';
                lazyBtn.style.top = pos.y + 'px';
            }

            // ゴルフ・ジムボタンへのマウス/タッチを検知して除外
            function isOverFixedButton(clientX, clientY) {
                const golfRect = golfBtn.getBoundingClientRect();
                const gymRect = gymBtn.getBoundingClientRect();
                
                // 固定ボタンの範囲内かチェック（余裕を持たせる）
                const isOverGolf = clientX >= golfRect.left - 20 && 
                                   clientX <= golfRect.right + 20 &&
                                   clientY >= golfRect.top - 20 && 
                                   clientY <= golfRect.bottom + 20;
                
                const isOverGym = clientX >= gymRect.left - 20 && 
                                  clientX <= gymRect.right + 20 &&
                                  clientY >= gymRect.top - 20 && 
                                  clientY <= gymRect.bottom + 20;
                
                return isOverGolf || isOverGym;
            }

            // タッチが近づいたら逃げる（スマホメイン）
            function handleTouch(clientX, clientY, isTouchStart = false) {
                // 固定ボタンの上にいる場合は逃げない
                if (isOverFixedButton(clientX, clientY)) {
                    return;
                }
                
                const btnRect = lazyBtn.getBoundingClientRect();
                const btnCenterX = btnRect.left + btnRect.width / 2;
                const btnCenterY = btnRect.top + btnRect.height / 2;
                
                const distance = Math.sqrt(
                    Math.pow(clientX - btnCenterX, 2) + 
                    Math.pow(clientY - btnCenterY, 2)
                );
                
                // スマホ用: 200px圏内で逃げる（距離を短縮）
                const escapeDistance = window.innerWidth <= 768 ? 200 : 150;
                
                if (distance < escapeDistance) {
                    attempts++;
                    updateCounter();
                    showMessage();
                    moveAwayFrom(clientX, clientY, true);
                }
            }

            // 特定の位置から素早く逃げる
            function moveAwayFrom(inputX, inputY, isTouch = false) {
                const containerRect = container.getBoundingClientRect();
                const btnRect = lazyBtn.getBoundingClientRect();
                
                const btnCenterX = btnRect.left + btnRect.width / 2 - containerRect.left;
                const btnCenterY = btnRect.top + btnRect.height / 2 - containerRect.top;
                
                const inputRelativeX = inputX - containerRect.left;
                const inputRelativeY = inputY - containerRect.top;
                
                const angle = Math.atan2(btnCenterY - inputRelativeY, btnCenterX - inputRelativeX);
                
                // スマホでは超速く逃げる
                const moveDistance = isTouch ? 350 + Math.random() * 150 : 300 + Math.random() * 100;
                
                let newX = btnCenterX + Math.cos(angle) * moveDistance - btnRect.width / 2;
                let newY = btnCenterY + Math.sin(angle) * moveDistance - btnRect.height / 2;
                
                const margin = 50;
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
                
                if (isOverlapping(testRect, golfRelative, 80) || 
                    isOverlapping(testRect, gymRelative, 80)) {
                    const pos = findNonOverlappingPosition();
                    newX = pos.x;
                    newY = pos.y;
                }
                
                lazyBtn.style.left = newX + 'px';
                lazyBtn.style.top = newY + 'px';
            }

            function updateCounter() {
                counter.textContent = `逃げられた回数: ${attempts}回`;
                counter.style.animation = 'none';
                setTimeout(() => {
                    counter.style.animation = 'popIn 0.3s cubic-bezier(0.68, -0.55, 0.265, 1.55)';
                }, 10);
            }

            function showMessage() {
                const msgIndex = Math.min(attempts - 1, escapeMessages.length - 1);
                messageDiv.textContent = escapeMessages[msgIndex];
                messageDiv.style.display = 'block';
                
                setTimeout(() => {
                    messageDiv.style.display = 'none';
                }, 1200);
            }

            function selectOption(choice) {
                clearInterval(autoMoveInterval);
                // 少し待ってからStreamlitに送信（確実に送信するため）
                setTimeout(() => {
                    window.parent.postMessage({
                        type: 'streamlit:setComponentValue',
                        value: choice
                    }, '*');
                }, 100);
            }

            // タッチイベント（スマホメイン）
            document.addEventListener('touchstart', function(e) {
                touchStartTime = Date.now();
                if (e.touches.length > 0) {
                    handleTouch(e.touches[0].clientX, e.touches[0].clientY, true);
                }
            });

            document.addEventListener('touchmove', function(e) {
                e.preventDefault();
                if (e.touches.length > 0) {
                    handleTouch(e.touches[0].clientX, e.touches[0].clientY, true);
                }
            }, { passive: false });

            // マウスイベント（PC）
            document.addEventListener('mousemove', function(e) {
                handleTouch(e.clientX, e.clientY, false);
            });

            // 逃げるボタンのイベント
            lazyBtn.addEventListener('click', function(e) {
                e.preventDefault();
                e.stopPropagation();
                attempts++;
                updateCounter();
                showMessage();
                autoMove();
                return false;
            });

            lazyBtn.addEventListener('touchend', function(e) {
                e.preventDefault();
                e.stopPropagation();
                return false;
            });

            // 初期化
            window.addEventListener('load', function() {
                setFixedButtonPositions();
                const pos = findNonOverlappingPosition();
                lazyBtn.style.left = pos.x + 'px';
                lazyBtn.style.top = pos.y + 'px';
                
                // 0.6秒ごとに自動で動く
                autoMoveInterval = setInterval(autoMove, 600);
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
    selected = components.html(html_code, height=750, scrolling=False)
    
    # 選択があった場合（デバッグ用ログ追加）
    if selected:
        st.session_state.choice = selected
        st.session_state.page = 'result'
        st.rerun()

# 結果ページ
elif st.session_state.page == 'result':
    st.balloons()
    
    # 選択に応じた褒め言葉
    if st.session_state.choice == 'golf':
        st.title("⛳ 素晴らしい！ゴルフだね！")
        praise = [
            "### 🎉 やったね！最高の選択だよ！",
            "ゴルフは健康にも良いし、気分転換にもなるよね！",
            "青空の下で思いっきりスイングしてきて！🌤️",
            "ナイスショット間違いなし！⛳✨"
        ]
    else:  # gym
        st.title("💪 最高！ジムに行くんだね！")
        praise = [
            "### 🎉 完璧な選択！カッコいい！",
            "体を動かすって本当に気持ちいいよね！",
            "今日もしっかりトレーニングだ！🏋️",
            "理想の体に一歩近づくぞ！💯"
        ]
    
    for text in praise:
        st.write(text)
    
    st.success("家でゴロゴロなんて選ばなくて本当に良かった！健康的な生活、応援してるよ！🌟")
    
    st.write("")
    st.write("")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔄 もう一度チャレンジする", use_container_width=True, type="primary"):
            st.session_state.page = 'select'
            st.session_state.choice = None
            st.rerun()
