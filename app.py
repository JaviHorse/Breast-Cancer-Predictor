import gradio as gr
import pandas as pd
import joblib

model = joblib.load("breast_cancer_svm_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")


def predict_breast_cancer(*inputs):
    input_df = pd.DataFrame([inputs], columns=feature_names)
    scaled_input = scaler.transform(input_df)

    prediction = model.predict(scaled_input)[0]

    if prediction == 1:
        return """
        <div class="prediction-card malignant-card">
            <div class="prediction-label">MODEL OUTPUT</div>
            <h2>Malignant</h2>
            <p>
                The model classified this tumor profile as <strong>malignant</strong>
                based on the provided diagnostic measurements.
            </p>
            <div class="risk-pill malignant-pill">Higher-risk classification</div>
            <p class="medical-note">
                This SHOULD NOT replace medical diagnosis.
            </p>
        </div>
        """
    else:
        return """
        <div class="prediction-card benign-card">
            <div class="prediction-label">MODEL OUTPUT</div>
            <h2>Benign</h2>
            <p>
                The model classified this tumor profile as <strong>benign</strong>
                based on the provided diagnostic measurements.
            </p>
            <div class="risk-pill benign-pill">Lower-risk classification</div>
            <p class="medical-note">
                This SHOULD NOT replace medical diagnosis.
            </p>
        </div>
        """


custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --bg-main: #25282d;
    --bg-card: #2b2f36;
    --bg-dark: #202329;
    --bg-input: #30343b;
    --text-main: #f7f7f8;
    --text-muted: #a8adb7;
    --border-soft: rgba(255, 255, 255, 0.08);
    --glow-blue: #1f7aff;
    --glow-purple: #c947ff;
    --glow-pink: #ef4cff;
    --good-green: #22c55e;
    --bad-red: #ef4444;
}

/* Global responsive safety */
* {
    box-sizing: border-box !important;
}

html,
body,
#root,
.gradio-container,
.main,
.app {
    width: 100% !important;
    min-width: 0 !important;
    overflow-x: hidden !important;
    background: #25282d !important;
    color: var(--text-main) !important;
    font-family: 'Inter', sans-serif !important;
}

.gradio-container {
    width: 100% !important;
    max-width: 1280px !important;
    margin: auto !important;
    padding: clamp(16px, 4vw, 32px) !important;
    background:
        radial-gradient(circle at top left, rgba(201, 71, 255, 0.13), transparent 28%),
        radial-gradient(circle at top right, rgba(31, 122, 255, 0.14), transparent 30%),
        #25282d !important;
}

/* Remove default Gradio blocks */
.contain,
.block,
.form,
.wrap,
.group,
.tabs,
.tabitem,
.prose,
.markdown {
    background: transparent !important;
    color: var(--text-main) !important;
    border-color: transparent !important;
    box-shadow: none !important;
    min-width: 0 !important;
}

/* Hero */
.hero {
    position: relative;
    overflow: hidden;
    border-radius: clamp(22px, 4vw, 34px);
    background: #2b2f36;
    padding: clamp(30px, 7vw, 64px) clamp(22px, 5vw, 46px);
    box-shadow:
        16px 16px 34px rgba(0, 0, 0, 0.38),
        -12px -12px 28px rgba(255, 255, 255, 0.035);
    margin-bottom: clamp(20px, 4vw, 28px);
}

.hero::before {
    content: "";
    position: absolute;
    inset: 0;
    background:
        radial-gradient(circle at 20% 20%, rgba(201, 71, 255, 0.20), transparent 28%),
        radial-gradient(circle at 70% 18%, rgba(31, 122, 255, 0.20), transparent 30%);
    pointer-events: none;
}

.hero-content {
    position: relative;
    z-index: 2;
    max-width: 860px;
}

.kicker {
    display: inline-flex;
    align-items: center;
    color: #d8c7ff;
    background: #202329;
    padding: 8px 12px;
    border-radius: 999px;
    font-size: clamp(11px, 2.6vw, 13px);
    margin-bottom: 18px;
    box-shadow:
        inset 3px 3px 8px rgba(0, 0, 0, 0.42),
        inset -3px -3px 8px rgba(255, 255, 255, 0.04);
}

.hero h1 {
    margin: 0;
    font-size: clamp(34px, 8vw, 72px);
    line-height: 1.02;
    letter-spacing: -0.06em;
    color: var(--text-main);
    max-width: 100%;
    overflow-wrap: break-word;
}

.hero h1 span {
    background: linear-gradient(135deg, #1f7aff, #c947ff, #ef4cff);
    -webkit-background-clip: text;
    color: transparent;
}

.hero p {
    margin-top: 20px;
    color: var(--text-muted);
    font-size: clamp(14px, 3.4vw, 17px);
    line-height: 1.7;
    max-width: 760px;
}

.hero-actions {
    display: flex;
    gap: 12px;
    margin-top: 24px;
    flex-wrap: wrap;
}

.action-button {
    padding: 12px 16px;
    border-radius: 16px;
    font-weight: 700;
    font-size: clamp(12px, 3vw, 14px);
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-height: 42px;
}

.primary-action {
    background: linear-gradient(135deg, #1f7aff, #c947ff);
    color: white;
    box-shadow:
        0 0 22px rgba(31, 122, 255, 0.5),
        0 0 28px rgba(201, 71, 255, 0.35);
}

.secondary-action {
    background: #202329;
    color: #d6d9df;
    box-shadow:
        inset 3px 3px 8px rgba(0, 0, 0, 0.42),
        inset -3px -3px 8px rgba(255, 255, 255, 0.04);
}

/* Metrics */
.metrics-row {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: clamp(14px, 3vw, 18px);
    margin-bottom: clamp(20px, 4vw, 28px);
}

.metric-card {
    background: #2b2f36;
    border-radius: 24px;
    padding: clamp(18px, 4vw, 24px);
    min-width: 0;
    box-shadow:
        10px 10px 22px rgba(0, 0, 0, 0.35),
        -8px -8px 20px rgba(255, 255, 255, 0.035);
}

.metric-card p {
    color: var(--text-muted);
    margin: 0 0 10px 0;
    font-size: clamp(10px, 2.5vw, 12px);
    text-transform: uppercase;
    letter-spacing: 0.12em;
}

.metric-card h3 {
    color: var(--text-main);
    margin: 0;
    font-size: clamp(24px, 6vw, 30px);
}

/* Main panel row */
#main-panels {
    width: 100% !important;
    display: flex !important;
    gap: clamp(18px, 3vw, 24px) !important;
    align-items: flex-start !important;
}

#main-panels > div {
    min-width: 0 !important;
}

/* Main panels */
.dark-panel {
    background: #2b2f36 !important;
    border-radius: clamp(22px, 4vw, 30px) !important;
    padding: clamp(20px, 4vw, 30px) !important;
    width: 100% !important;
    min-width: 0 !important;
    box-shadow:
        14px 14px 30px rgba(0, 0, 0, 0.38),
        -10px -10px 24px rgba(255, 255, 255, 0.035) !important;
    color: var(--text-main) !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
}

.prediction-panel {
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    justify-content: flex-start !important;
}

.prediction-panel-inner {
    width: 100% !important;
    max-width: 390px !important;
    margin: 0 auto !important;
}

/* Headers */
.section-header {
    background: transparent !important;
    padding: 0 0 18px 0 !important;
    margin: 0 !important;
    width: 100% !important;
}

.section-header h2 {
    margin: 0 0 8px 0 !important;
    padding: 0 !important;
    color: #f7f7f8 !important;
    font-size: clamp(22px, 5vw, 26px) !important;
    font-weight: 800 !important;
    line-height: 1.2 !important;
    letter-spacing: -0.02em !important;
    overflow-wrap: break-word !important;
}

.section-header p {
    margin: 0 !important;
    padding: 0 !important;
    color: #a8adb7 !important;
    font-size: clamp(13px, 3vw, 15px) !important;
    line-height: 1.6 !important;
    max-width: 760px !important;
    overflow-wrap: break-word !important;
}

/* Remove inner wrappers */
.dark-panel > .block,
.dark-panel .block,
.dark-panel .form,
.dark-panel .wrap,
.dark-panel .tabs,
.dark-panel .tabitem,
.dark-panel .gap,
.dark-panel .form > div,
.dark-panel .block > div,
.dark-panel [data-testid="markdown"],
.dark-panel [data-testid="markdown"] *,
.dark-panel .prose,
.dark-panel .prose *,
.dark-panel .markdown,
.dark-panel .markdown * {
    background: transparent !important;
    color: var(--text-main) !important;
    border: none !important;
    box-shadow: none !important;
    min-width: 0 !important;
}

/* Inputs */
label,
.label-wrap span {
    color: #d6d9df !important;
    font-weight: 600 !important;
    font-size: clamp(12px, 3vw, 14px) !important;
    overflow-wrap: break-word !important;
}

input,
textarea {
    width: 100% !important;
    background: #30343b !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 18px !important;
    padding: 14px !important;
    box-shadow:
        inset 5px 5px 10px rgba(0, 0, 0, 0.34),
        inset -5px -5px 10px rgba(255, 255, 255, 0.035) !important;
}

input:focus,
textarea:focus {
    outline: none !important;
    color: white !important;
    box-shadow:
        0 0 0 2px rgba(201, 71, 255, 0.75),
        0 0 18px rgba(201, 71, 255, 0.35),
        inset 5px 5px 10px rgba(0, 0, 0, 0.34) !important;
}

/* Tabs */
.tab-nav {
    background: transparent !important;
    border-radius: 18px !important;
    padding: 4px 0 !important;
    box-shadow: none !important;
    border-bottom: 1px solid rgba(255, 255, 255, 0.12) !important;
    display: flex !important;
    flex-wrap: wrap !important;
    gap: 6px !important;
}

.tab-nav button {
    color: #d6d9df !important;
    background: transparent !important;
    border-radius: 14px !important;
    border: none !important;
    box-shadow: none !important;
    transition: all 0.2s ease !important;
    white-space: normal !important;
    min-width: fit-content !important;
    font-size: clamp(12px, 3vw, 14px) !important;
}

.tab-nav button:hover {
    color: #ffffff !important;
    background: rgba(201, 71, 255, 0.18) !important;
    box-shadow: none !important;
}

.tab-nav button.selected {
    color: white !important;
    background: linear-gradient(135deg, rgba(31, 122, 255, 0.95), rgba(201, 71, 255, 0.95)) !important;
    box-shadow:
        0 0 16px rgba(31, 122, 255, 0.32),
        0 0 18px rgba(201, 71, 255, 0.24) !important;
}

/* Buttons */
button {
    border-radius: 18px !important;
    border: none !important;
    transition: all 0.2s ease !important;
}

button:hover,
button.primary:hover,
button.secondary:hover {
    background: linear-gradient(135deg, #1f7aff, #c947ff) !important;
    color: white !important;
    opacity: 0.95 !important;
    box-shadow:
        0 0 28px rgba(31, 122, 255, 0.55),
        0 0 34px rgba(201, 71, 255, 0.45) !important;
}

button.primary {
    background: linear-gradient(135deg, #1f7aff, #c947ff) !important;
    color: white !important;
    font-weight: 800 !important;
    box-shadow:
        0 0 22px rgba(31, 122, 255, 0.45),
        0 0 28px rgba(201, 71, 255, 0.35) !important;
}

/* Generate prediction button */
.predict-button-wrap {
    width: 100% !important;
    display: flex !important;
    justify-content: center !important;
    margin: 18px 0 24px 0 !important;
}

.predict-button-wrap > div {
    width: 100% !important;
    display: flex !important;
    justify-content: center !important;
}

#predict-button {
    width: 72% !important;
    max-width: 270px !important;
    min-width: 200px !important;
    height: 52px !important;
    font-size: clamp(13px, 3.2vw, 15px) !important;
    font-weight: 800 !important;
    margin: 0 auto !important;
}

/* Prediction card */
.prediction-card {
    border-radius: 26px;
    padding: clamp(20px, 4vw, 26px);
    margin: 18px auto 0 auto;
    min-height: 180px;
    max-width: 390px;
    width: 100%;
}

.prediction-label {
    font-size: 12px;
    letter-spacing: 0.12em;
    font-weight: 800;
    opacity: 0.75;
    margin-bottom: 10px;
}

.prediction-card h2 {
    font-size: clamp(32px, 7vw, 42px);
    margin: 0 0 10px 0;
    letter-spacing: -0.04em;
}

.prediction-card p {
    line-height: 1.6;
    font-size: clamp(13px, 3vw, 15px);
}

.benign-card {
    background: #24382f;
    border: 1px solid rgba(34, 197, 94, 0.32);
    box-shadow: 0 0 35px rgba(34, 197, 94, 0.16);
    color: #dcfce7;
}

.malignant-card {
    background: #3a252c;
    border: 1px solid rgba(239, 68, 68, 0.34);
    box-shadow: 0 0 35px rgba(239, 68, 68, 0.16);
    color: #fee2e2;
}

.risk-pill {
    display: inline-flex;
    padding: 9px 13px;
    border-radius: 999px;
    font-size: 13px;
    font-weight: 800;
    margin-top: 12px;
}

.benign-pill {
    background: rgba(34, 197, 94, 0.16);
    color: #bbf7d0;
}

.malignant-pill {
    background: rgba(239, 68, 68, 0.16);
    color: #fecaca;
}

.medical-note {
    color: rgba(255,255,255,0.7) !important;
    font-size: 13px !important;
    margin-top: 16px;
}

/* Interpretation */
.info-box {
    width: 100%;
    max-width: 390px;
    background: #30343b;
    border-radius: 24px;
    padding: clamp(16px, 4vw, 20px);
    margin: 24px auto 0 auto;
    box-shadow:
        inset 5px 5px 12px rgba(0, 0, 0, 0.32),
        inset -5px -5px 12px rgba(255, 255, 255, 0.035);
}

.info-box h3 {
    margin: 0 0 16px 0;
    color: var(--text-main);
    font-size: clamp(18px, 4vw, 20px);
    font-weight: 800;
    letter-spacing: -0.02em;
}

.interpret-list {
    display: flex;
    flex-direction: column;
    gap: 12px;
}

.interpret-item {
    display: grid;
    grid-template-columns: 34px minmax(0, 1fr);
    gap: 12px;
    align-items: start;
    padding: 13px;
    border-radius: 18px;
    background: rgba(32, 35, 41, 0.72);
}

.interpret-icon {
    width: 34px;
    height: 34px;
    border-radius: 999px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 900;
    font-size: 16px;
    line-height: 1;
}

.good-icon {
    background: rgba(34, 197, 94, 0.16);
    color: #86efac;
    box-shadow: 0 0 18px rgba(34, 197, 94, 0.18);
}

.bad-icon {
    background: rgba(239, 68, 68, 0.16);
    color: #fca5a5;
    box-shadow: 0 0 18px rgba(239, 68, 68, 0.18);
}

.neutral-icon {
    background: rgba(201, 71, 255, 0.16);
    color: #e9d5ff;
    box-shadow: 0 0 18px rgba(201, 71, 255, 0.18);
}

.interpret-title {
    font-size: clamp(13px, 3.2vw, 14px);
    font-weight: 800;
    color: #ffffff;
    line-height: 1.35;
    margin-bottom: 4px;
}

.good-text {
    color: #86efac !important;
}

.bad-text {
    color: #fca5a5 !important;
}

.interpret-desc {
    font-size: clamp(12px, 3vw, 13px);
    line-height: 1.45;
    color: var(--text-muted);
}

.footer {
    text-align: center;
    color: var(--text-muted);
    margin: 32px 0 22px 0;
    font-size: clamp(11px, 2.7vw, 13px);
    line-height: 1.5;
}

/* Tablet */
@media (max-width: 900px) {
    .metrics-row {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
}

/* Stack panels instead of squeezing */
@media (max-width: 760px) {
    #main-panels {
        flex-direction: column !important;
    }

    #main-panels > div {
        width: 100% !important;
        max-width: 100% !important;
        flex: 1 1 100% !important;
    }

    .prediction-panel-inner,
    .info-box,
    .prediction-card {
        max-width: 100% !important;
    }

    #predict-button {
        width: 72% !important;
        max-width: 280px !important;
        min-width: 200px !important;
    }
}

/* Phone */
@media (max-width: 640px) {
    .gradio-container {
        padding: 14px !important;
    }

    .hero {
        padding: 30px 20px;
        border-radius: 24px;
    }

    .hero-actions {
        flex-direction: column;
        align-items: stretch;
    }

    .action-button {
        width: 100%;
        text-align: center;
    }

    .metrics-row {
        grid-template-columns: 1fr;
    }

    .dark-panel {
        padding: 20px !important;
        border-radius: 24px !important;
    }

    #predict-button {
        width: 100% !important;
        max-width: 100% !important;
        min-width: 0 !important;
    }

    .tab-nav {
        flex-direction: column !important;
        align-items: stretch !important;
    }

    .tab-nav button {
        width: 100% !important;
        text-align: left !important;
    }

    .interpret-item {
        grid-template-columns: 30px minmax(0, 1fr);
        gap: 10px;
        padding: 12px;
    }

    .interpret-icon {
        width: 30px;
        height: 30px;
        font-size: 14px;
    }
}

/* Very small phones */
@media (max-width: 420px) {
    .gradio-container {
        padding: 10px !important;
    }

    .hero h1 {
        font-size: 31px;
    }

    .hero p {
        font-size: 14px;
    }

    .metric-card h3 {
        font-size: 24px;
    }

    .section-header h2 {
        font-size: 22px !important;
    }
}
"""


mean_features = [f for f in feature_names if f.endswith("_mean")]
se_features = [f for f in feature_names if f.endswith("_se")]
worst_features = [f for f in feature_names if f.endswith("_worst")]


def create_number_inputs(features):
    return [
        gr.Number(
            label=feature.replace("_", " ").title(),
            value=0,
            precision=6
        )
        for feature in features
    ]


with gr.Blocks(
    css=custom_css,
    title="Breast Cancer AI Classifier",
    theme=gr.themes.Base(
        primary_hue="purple",
        secondary_hue="blue",
        neutral_hue="slate"
    ).set(
        body_background_fill="#25282d",
        body_text_color="#f7f7f8",
        block_background_fill="#2b2f36",
        block_border_color="rgba(255,255,255,0.08)",
        input_background_fill="#30343b",
        input_border_color="rgba(255,255,255,0.08)",
        button_primary_background_fill="linear-gradient(135deg, #1f7aff, #c947ff)",
        button_primary_text_color="#ffffff"
    )
) as demo:

    gr.HTML("""
    <section class="hero">
        <div class="hero-content">
            <div class="kicker">Machine Learning Diagnostic Demo</div>
            <h1>Predict Breast cancer tumor class with a <span>trained ML Model.</span></h1>
            <p>
                This app uses a trained Support Vector Machine model to classify breast tumor profiles
                as benign or malignant using 30 diagnostic cell nuclei measurements from the
                Wisconsin Diagnostic Breast Cancer dataset.
            </p>
            <div class="hero-actions">
                <span class="action-button primary-action">Run Prediction Below</span>
                <span class="action-button secondary-action">Trained Demo</span>
            </div>
        </div>
    </section>
    """)

    gr.HTML("""
    <div class="metrics-row">
        <div class="metric-card">
            <p>Dataset Samples</p>
            <h3>569</h3>
        </div>
        <div class="metric-card">
            <p>Input Features</p>
            <h3>30</h3>
        </div>
        <div class="metric-card">
            <p>Final Model</p>
            <h3>SVM</h3>
        </div>
        <div class="metric-card">
            <p>Test Accuracy</p>
            <h3>97.37%</h3>
        </div>
    </div>
    """)

    with gr.Row(elem_id="main-panels"):
        with gr.Column(scale=2, min_width=280):
            with gr.Group(elem_classes="dark-panel"):
                gr.HTML("""
                <div class="section-header">
                    <h2>Input Tumor Measurements</h2>
                    <p>
                        Provide the diagnostic feature values below. The inputs are grouped into
                        mean features, standard error features, and worst-case feature measurements.
                    </p>
                </div>
                """)

                all_inputs = []

                with gr.Tabs():
                    with gr.Tab("Mean Features"):
                        mean_inputs = create_number_inputs(mean_features)
                        all_inputs.extend(mean_inputs)

                    with gr.Tab("Standard Error Features"):
                        se_inputs = create_number_inputs(se_features)
                        all_inputs.extend(se_inputs)

                    with gr.Tab("Worst Features"):
                        worst_inputs = create_number_inputs(worst_features)
                        all_inputs.extend(worst_inputs)

        with gr.Column(scale=1, min_width=280):
            with gr.Group(elem_classes=["dark-panel", "prediction-panel"]):
                with gr.Column(elem_classes="prediction-panel-inner"):
                    gr.HTML("""
                    <div class="section-header">
                        <h2>Prediction</h2>
                        <p>Click the button after entering the tumor measurements.</p>
                    </div>
                    """)

                    with gr.Column(elem_classes="predict-button-wrap"):
                        predict_button = gr.Button(
                            "Generate Prediction",
                            variant="primary",
                            elem_id="predict-button"
                        )

                    output = gr.HTML()

                    gr.HTML("""
                    <div class="info-box">
                        <h3>How to interpret</h3>

                        <div class="interpret-list">
                            <div class="interpret-item">
                                <div class="interpret-icon good-icon">✓</div>
                                <div class="interpret-text">
                                    <div class="interpret-title good-text">Benign = Lower-risk result</div>
                                    <div class="interpret-desc">
                                        The model predicts a non-cancerous tumor profile.
                                    </div>
                                </div>
                            </div>

                            <div class="interpret-item">
                                <div class="interpret-icon bad-icon">!</div>
                                <div class="interpret-text">
                                    <div class="interpret-title bad-text">Malignant = Higher-risk result</div>
                                    <div class="interpret-desc">
                                        The model predicts a cancerous tumor profile.
                                    </div>
                                </div>
                            </div>

                            <div class="interpret-item">
                                <div class="interpret-icon neutral-icon">i</div>
                                <div class="interpret-text">
                                    <div class="interpret-title">Limited dataset for Demo only</div>
                                    <div class="interpret-desc">
                                        This does not replace medical diagnosis.
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    """)

    predict_button.click(
        fn=predict_breast_cancer,
        inputs=all_inputs,
        outputs=output
    )

    gr.HTML("""
    <div class="footer">
        Built with Python, scikit-learn, and Gradio · Wisconsin Diagnostic Breast Cancer Dataset · Educational Demo Only
    </div>
    """)


demo.launch()