import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

warnings.filterwarnings("ignore", category=FutureWarning)

DATASET_ID = 891
TARGET = "Diabetes_binary"
UCI_URL = "https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators"
UCI_DOI = "https://doi.org/10.24432/C53919"

st.set_page_config(
    page_title="당뇨병 데이터와 AI 공정성 탐구",
    page_icon="🧬",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1250px;}
    div[data-testid="stMetric"] {
        background: #f5f7fb;
        border: 1px solid #e4e8f0;
        padding: 13px 15px;
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🧬 당뇨병 건강지표 데이터 기반 머신러닝 분석")
st.caption("탐구 주제: 의료 AI의 전체 정확도는 성별·연령대별 신뢰성을 보장하는가?")
st.info(
    "교육·탐구용 프로젝트입니다. 이 모델은 설문 데이터의 분류 성능을 분석하기 위한 것이며, "
    "개인의 당뇨병 여부를 진단하거나 미래 발병을 예측하는 의료 도구가 아닙니다."
)


@st.cache_data(show_spinner="UCI 공개 데이터셋을 불러오는 중입니다...")
def load_data():
    """UCI Machine Learning Repository에서 CDC Diabetes Health Indicators를 불러온다."""
    from ucimlrepo import fetch_ucirepo

    dataset = fetch_ucirepo(id=DATASET_ID)
    features = dataset.data.features.copy()
    targets = dataset.data.targets.copy()
    features.columns = [str(c).strip() for c in features.columns]
    targets.columns = [str(c).strip() for c in targets.columns]

    target_feature = next((c for c in features.columns if c.lower() == TARGET.lower()), None)
    target_column = next((c for c in targets.columns if c.lower() == TARGET.lower()), None)

    if target_feature is not None:
        target_values = features.pop(target_feature).reset_index(drop=True)
    elif target_column is not None:
        target_values = targets[target_column].reset_index(drop=True)
    elif targets.shape[1] == 1:
        target_values = targets.iloc[:, 0].reset_index(drop=True)
    else:
        target_like = [c for c in targets.columns if "diabetes" in c.lower()]
        if len(target_like) == 1:
            target_values = targets[target_like[0]].reset_index(drop=True)
        else:
            raise ValueError(
                "데이터에서 당뇨병 결과 변수(타깃)를 찾지 못했습니다. "
                "UCI 데이터셋의 변수 구성을 확인해 주세요."
            )

    if len(features) != len(target_values):
        raise ValueError("입력 데이터와 결과 데이터의 행 수가 일치하지 않습니다.")

    df = features.reset_index(drop=True).copy()
    df[TARGET] = pd.to_numeric(target_values, errors="coerce").to_numpy()
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET]).copy()
    df = df[df[TARGET].isin([0, 1])].copy()
    df[TARGET] = df[TARGET].astype(int)

    for column in df.columns:
        if column != TARGET:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    if "Sex" not in df.columns or "Age" not in df.columns:
        raise ValueError(
            "성별(Sex) 또는 연령(Age) 변수를 찾지 못했습니다. "
            f"현재 열: {', '.join(map(str, df.columns))}"
        )
    if df[TARGET].nunique() < 2:
        raise ValueError("당뇨병 결과 변수에 두 분류(0과 1)가 모두 있어야 합니다.")
    return df


def age_band(value):
    """UCI Age 코드(1~13)를 네 연령대로 묶는다. Age는 정확한 나이가 아닌 구간 코드다."""
    try:
        age_code = int(value)
    except (TypeError, ValueError):
        return "알 수 없음"
    if 1 <= age_code <= 4:
        return "18–39세"
    if 5 <= age_code <= 7:
        return "40–54세"
    if 8 <= age_code <= 10:
        return "55–69세"
    if 11 <= age_code <= 13:
        return "70세 이상"
    return "알 수 없음"


def safe_divide(numerator, denominator):
    return float(numerator / denominator) if denominator else np.nan


def calculate_metrics(y_true, y_pred, y_score=None):
    """양성 클래스 1(전당뇨/당뇨병 범주) 기준 지표를 계산한다."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    result = {
        "평가 표본 수": int(len(y_true)),
        "실제 양성 수": int(tp + fn),
        "정확도(%)": accuracy_score(y_true, y_pred) * 100,
        "민감도(%)": safe_divide(tp, tp + fn) * 100,
        "특이도(%)": safe_divide(tn, tn + fp) * 100,
        "위음성률(%)": safe_divide(fn, fn + tp) * 100,
        "위양성률(%)": safe_divide(fp, fp + tn) * 100,
    }
    result["ROC-AUC"] = (
        roc_auc_score(y_true, y_score)
        if y_score is not None and len(np.unique(y_true)) == 2
        else np.nan
    )
    return result, cm


@st.cache_data(show_spinner=False)
def run_analysis(sample_size, model_choice, feature_mode, threshold):
    df = load_data()

    # 표본을 층화 추출해 실행 시간을 줄이고 양성·음성 라벨 비율을 유지한다.
    if sample_size < len(df):
        sample_df, _ = train_test_split(
            df,
            train_size=int(sample_size),
            stratify=df[TARGET],
            random_state=42,
        )
    else:
        sample_df = df.copy()

    train_df, test_df = train_test_split(
        sample_df,
        test_size=0.25,
        stratify=sample_df[TARGET],
        random_state=42,
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    excluded = {TARGET, "ID"}
    if feature_mode == "성별·연령 변수 제외":
        excluded.update({"Sex", "Age"})
    feature_cols = [c for c in sample_df.columns if c not in excluded]
    if not feature_cols:
        raise ValueError("모델 학습에 사용할 입력 변수가 없습니다.")

    X_train = train_df[feature_cols]
    y_train = train_df[TARGET]
    X_test = test_df[feature_cols]
    y_test = test_df[TARGET].to_numpy()

    model_names = (
        ["로지스틱 회귀", "랜덤 포레스트"]
        if model_choice == "두 모델 비교"
        else [model_choice]
    )
    overall_rows, group_rows, confusion_matrices = [], [], {}
    evaluation_size = {
        "학습 표본 수": len(train_df),
        "평가 표본 수": len(test_df),
        "사용 입력 변수 수": len(feature_cols),
        "전체 사용 표본 수": len(sample_df),
        "전체 데이터 수": len(df),
        "평가 양성 수": int((y_test == 1).sum()),
        "평가 음성 수": int((y_test == 0).sum()),
        "사용 변수": feature_cols,
    }

    test_groups = test_df.copy()
    test_groups["성별"] = test_groups["Sex"].map({0: "여성", 1: "남성"}).fillna("알 수 없음")
    test_groups["연령대"] = test_groups["Age"].apply(age_band)

    for model_name in model_names:
        if model_name == "로지스틱 회귀":
            model = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    (
                        "classifier",
                        LogisticRegression(
                            max_iter=600,
                            class_weight="balanced",
                            solver="liblinear",
                            random_state=42,
                        ),
                    ),
                ]
            )
        else:
            model = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "classifier",
                        RandomForestClassifier(
                            n_estimators=100,
                            max_depth=14,
                            min_samples_leaf=10,
                            class_weight="balanced_subsample",
                            n_jobs=-1,
                            random_state=42,
                        ),
                    ),
                ]
            )

        model.fit(X_train, y_train)
        positive_index = list(model.classes_).index(1)
        positive_score = model.predict_proba(X_test)[:, positive_index]
        predictions = (positive_score >= float(threshold)).astype(int)
        overall_metrics, cm = calculate_metrics(y_test, predictions, positive_score)
        overall_rows.append({"모델": model_name, **overall_metrics})
        confusion_matrices[model_name] = cm.tolist()

        scored_test = test_groups.copy()
        scored_test["실제값"] = y_test
        scored_test["예측값"] = predictions
        scored_test["양성 점수"] = positive_score

        group_orders = [
            ("성별", ["여성", "남성", "알 수 없음"]),
            ("연령대", ["18–39세", "40–54세", "55–69세", "70세 이상", "알 수 없음"]),
        ]
        for group_col, group_order in group_orders:
            for group_label in group_order:
                group_df = scored_test[scored_test[group_col] == group_label]
                if group_df.empty:
                    continue
                metrics, _ = calculate_metrics(
                    group_df["실제값"].to_numpy(),
                    group_df["예측값"].to_numpy(),
                    group_df["양성 점수"].to_numpy(),
                )
                group_rows.append(
                    {"모델": model_name, "집단 기준": group_col, "집단": group_label, **metrics}
                )

    return (
        pd.DataFrame(overall_rows),
        pd.DataFrame(group_rows),
        confusion_matrices,
        evaluation_size,
    )


try:
    data = load_data()
except Exception as error:
    st.error(
        "UCI 데이터셋을 불러오지 못했습니다. 이 앱은 인터넷 연결이 필요합니다. "
        "잠시 후 다시 실행하거나 아래 공식 데이터 페이지를 확인해 주세요."
    )
    st.markdown(f"[UCI 공식 데이터셋 페이지]({UCI_URL})")
    with st.expander("오류 세부 내용"):
        st.code(str(error))
    st.stop()

positive_rate = float((data[TARGET] == 1).mean() * 100)
c1, c2, c3, c4 = st.columns(4)
c1.metric("전체 관측값", f"{len(data):,}개")
c2.metric("입력 변수", f"{len([c for c in data.columns if c not in [TARGET, 'ID']])}개")
c3.metric("분류 라벨 1 비율", f"{positive_rate:.1f}%")
c4.metric("타깃 결측값", "0개")

with st.expander("데이터셋 설명과 변수 해석"):
    st.markdown(
        """
        **자료 출처:** UCI Machine Learning Repository의 CDC Diabetes Health Indicators.

        - `Diabetes_binary = 0`: 데이터셋에서 당뇨병이 없는 범주
        - `Diabetes_binary = 1`: 전당뇨 또는 당뇨병 범주
        - `Sex`: 0은 여성, 1은 남성
        - `Age`: 실제 나이가 아니라 18세 이상 응답자의 연령 구간 코드(1~13)
        - `HighBP`, `HighChol`, `BMI`, `PhysActivity` 등 건강·생활습관 관련 변수 포함

        이 자료는 미국의 2015년 건강행태 설문 자료를 바탕으로 합니다. 의료기관에서 직접 측정한 검사 기록과 동일하지 않으며,
        결과 라벨은 미래 발병 여부를 뜻하지 않습니다.
        """
    )

left, right = st.columns([1.05, 1])
with left:
    st.subheader("분류 라벨 분포")
    label_counts = (
        data[TARGET]
        .map({0: "0 · 당뇨병 없음", 1: "1 · 전당뇨/당뇨병"})
        .value_counts()
        .rename_axis("분류")
        .reset_index(name="표본 수")
    )
    fig_labels = px.bar(
        label_counts,
        x="분류",
        y="표본 수",
        text="표본 수",
        color="분류",
        title="전체 데이터의 라벨 분포",
        color_discrete_sequence=["#7D9BC8", "#D97757"],
    )
    fig_labels.update_layout(showlegend=False, template="plotly_white", height=340)
    st.plotly_chart(fig_labels, use_container_width=True)
with right:
    st.subheader("데이터 미리보기")
    st.dataframe(data.head(10), use_container_width=True, hide_index=True)

st.divider()
st.header("머신러닝 모델 평가 설정")
st.caption("분석 조건을 선택하고 실행하면 별도의 평가용 데이터에서 모델의 오류율을 비교합니다.")

with st.sidebar:
    st.header("분석 설정")
    with st.form("analysis_form"):
        max_sample = min(100000, len(data))
        min_sample = min(10000, max_sample)
        default_sample = min(50000, max_sample)
        sample_size = st.slider(
            "분석에 사용할 표본 수",
            min_value=min_sample,
            max_value=max_sample,
            value=max_sample if max_sample < 10000 else default_sample,
            step=5000 if max_sample >= 15000 else 1,
            help="전체 데이터 중 층화 추출한 표본으로 학습·평가합니다. 표본이 많을수록 실행 시간이 늘 수 있습니다.",
        )
        model_choice = st.selectbox(
            "비교할 모델",
            ["두 모델 비교", "로지스틱 회귀", "랜덤 포레스트"],
            index=0,
        )
        feature_mode = st.radio(
            "모델 학습에 사용할 변수",
            ["모든 변수 사용", "성별·연령 변수 제외"],
            index=0,
            help="두 조건을 각각 실행해 결과를 비교하면 성별·연령 변수를 직접 넣었을 때의 차이를 살펴볼 수 있습니다. 제외해도 다른 변수에 관련 정보가 반영될 가능성은 있습니다.",
        )
        threshold = st.slider(
            "양성 판정 임계값",
            min_value=0.10,
            max_value=0.90,
            value=0.50,
            step=0.05,
            help="모델 점수가 이 값 이상이면 양성(전당뇨/당뇨병 범주)으로 분류합니다.",
        )
        run_button = st.form_submit_button("분석 실행", type="primary", use_container_width=True)

if "diabetes_analysis_config" not in st.session_state:
    st.session_state["diabetes_analysis_config"] = {
        "sample_size": int(min(50000, min(100000, len(data)))),
        "model_choice": "두 모델 비교",
        "feature_mode": "모든 변수 사용",
        "threshold": 0.50,
    }
if run_button:
    st.session_state["diabetes_analysis_config"] = {
        "sample_size": int(sample_size),
        "model_choice": model_choice,
        "feature_mode": feature_mode,
        "threshold": float(threshold),
    }
config = st.session_state["diabetes_analysis_config"]
st.write(
    f"현재 설정: **{config['model_choice']}** · 표본 **{config['sample_size']:,}개** · "
    f"입력 변수 설정 **{config['feature_mode']}** · 임계값 **{config['threshold']:.2f}**"
)

try:
    with st.spinner("표본을 나누고 모델을 학습·평가하고 있습니다..."):
        overall_df, group_df, confusion_data, size_info = run_analysis(
            config["sample_size"],
            config["model_choice"],
            config["feature_mode"],
            config["threshold"],
        )
except Exception as error:
    st.error("분석 중 오류가 발생했습니다. 표본 수를 줄인 뒤 다시 실행해 주세요.")
    with st.expander("오류 세부 내용"):
        st.code(str(error))
    st.stop()

st.header("1. 전체 성능")
m1, m2, m3, m4 = st.columns(4)
m1.metric("학습 표본", f"{size_info['학습 표본 수']:,}개")
m2.metric("평가 표본", f"{size_info['평가 표본 수']:,}개")
m3.metric("평가 양성 표본", f"{size_info['평가 양성 수']:,}개")
m4.metric("학습 입력 변수", f"{size_info['사용 입력 변수 수']}개")

st.dataframe(
    overall_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "정확도(%)": st.column_config.NumberColumn(format="%.1f%%"),
        "민감도(%)": st.column_config.NumberColumn(format="%.1f%%"),
        "특이도(%)": st.column_config.NumberColumn(format="%.1f%%"),
        "위음성률(%)": st.column_config.NumberColumn(format="%.1f%%"),
        "위양성률(%)": st.column_config.NumberColumn(format="%.1f%%"),
        "ROC-AUC": st.column_config.NumberColumn(format="%.3f"),
    },
)
st.caption(
    "민감도 = 실제 양성 중 양성으로 맞힌 비율 · 위음성률 = 실제 양성 중 음성으로 놓친 비율 · "
    "위양성률 = 실제 음성 중 양성으로 잘못 분류한 비율."
)

st.subheader("혼동행렬")
cm_cols = st.columns(len(confusion_data))
for col, (model_name, matrix) in zip(cm_cols, confusion_data.items()):
    with col:
        st.markdown(f"**{model_name}**")
        cm_df = pd.DataFrame(
            matrix,
            index=["실제 0 · 비당뇨", "실제 1 · 전당뇨/당뇨병"],
            columns=["예측 0", "예측 1"],
        )
        fig_cm = px.imshow(
            cm_df,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="Blues",
            labels={"x": "모델 예측", "y": "실제 라벨", "color": "표본 수"},
        )
        fig_cm.update_layout(
            template="plotly_white", height=320, margin=dict(l=10, r=10, t=25, b=10)
        )
        st.plotly_chart(fig_cm, use_container_width=True)

st.divider()
st.header("2. 성별·연령대별 오류율 비교")
st.write(
    "같은 평가 표본에서 집단별 지표를 따로 계산합니다. 차이가 보인다고 해서 곧바로 차별이나 "
    "인과관계가 증명되는 것은 아니며, 집단별 표본 수와 데이터 특성도 함께 확인해야 합니다."
)
group_choice = st.radio("비교할 집단", ["성별", "연령대"], horizontal=True)
metric_choice = st.selectbox(
    "그래프로 비교할 지표",
    ["위음성률(%)", "민감도(%)", "정확도(%)", "특이도(%)", "위양성률(%)"],
    index=0,
)
plot_df = group_df[group_df["집단 기준"] == group_choice].copy()
if not plot_df.empty:
    max_value = plot_df[metric_choice].max(skipna=True)
    y_max = min(110, max(10, float(max_value) * 1.22)) if pd.notna(max_value) else 100
    fig_group = px.bar(
        plot_df,
        x="집단",
        y=metric_choice,
        color="모델",
        barmode="group",
        text=metric_choice,
        title=f"{group_choice}별 {metric_choice} 비교",
        labels={metric_choice: metric_choice, "집단": group_choice},
        color_discrete_sequence=["#426A9A", "#D97757"],
    )
    fig_group.update_traces(texttemplate="%{text:.1f}%", textposition="outside", cliponaxis=False)
    fig_group.update_layout(
        template="plotly_white",
        height=420,
        yaxis=dict(range=[0, y_max]),
        legend_title_text="모델",
    )
    st.plotly_chart(fig_group, use_container_width=True)

st.subheader("집단별 세부 지표")
st.dataframe(
    group_df[group_df["집단 기준"] == group_choice],
    use_container_width=True,
    hide_index=True,
    column_config={
        "정확도(%)": st.column_config.NumberColumn(format="%.1f%%"),
        "민감도(%)": st.column_config.NumberColumn(format="%.1f%%"),
        "특이도(%)": st.column_config.NumberColumn(format="%.1f%%"),
        "위음성률(%)": st.column_config.NumberColumn(format="%.1f%%"),
        "위양성률(%)": st.column_config.NumberColumn(format="%.1f%%"),
        "ROC-AUC": st.column_config.NumberColumn(format="%.3f"),
    },
)
if not group_df.empty and (group_df["평가 표본 수"] < 100).any():
    st.warning(
        "평가 표본이 100개 미만인 집단이 있습니다. 작은 집단의 오류율은 몇 건의 오분류만으로도 "
        "크게 달라질 수 있으므로 조심해서 해석하세요."
    )

with st.expander("결과 해석 시 주의할 점"):
    st.markdown(
        """
        1. **정확도만으로 판단하지 않기:** 비당뇨 범주가 더 많을 수 있으므로 민감도·특이도·위음성률도 확인합니다.
        2. **집단별 표본 수 확인:** 각 집단의 실제 양성 수가 적으면 오류율이 불안정할 수 있습니다.
        3. **원인이라고 단정하지 않기:** 집단별 성능 차이는 표본 구성, 설문 특성, 입력 변수 등의 영향을 받을 수 있습니다.
        4. **변수 제외가 공정성을 보장하지 않음:** 성별·연령을 모델 입력에서 빼더라도 다른 변수에 관련 정보가 반영될 수 있습니다.
        5. **일반화에 주의:** 미국 2015년 설문 자료의 결과가 현재 한국 인구나 임상 환경에도 그대로 적용된다고 볼 수 없습니다.
        6. **진단 도구가 아님:** 공개 데이터에 대한 교육용 분류 실험이며 개인의 진단이나 치료 판단에 사용하면 안 됩니다.
        """
    )

st.divider()
st.subheader("자료 출처")
st.markdown(
    f"- [UCI Machine Learning Repository — CDC Diabetes Health Indicators]({UCI_URL})\n"
    f"- [데이터 DOI 및 인용 정보]({UCI_DOI})\n\n"
    "보고서에는 자료명, UCI 저장소, DOI를 출처로 기록하세요."
)
