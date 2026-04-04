import os
import re
import requests
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import anthropic

st.set_page_config(page_title="중소기업 애로상담 AI", page_icon="🏢", layout="centered")
st.title("🏢 중소기업 애로상담 AI")
st.caption("중소벤처기업부 비즈니스지원단 상담사례 기반 RAG 시스템")

ANTHROPIC_API_KEY = st.secrets["ANTHROPIC_API_KEY"]
GOV_API_KEY       = st.secrets["GOV_API_KEY"]

# 지원사업 관련 키워드 — 이 키워드 포함 질문에만 공고 API 호출
SUPPORT_KEYWORDS = [
    '지원금', '지원사업', '창업', '보조금', '융자',
    '정책자금', '공고', '신청', '자금', '지원',
    '보증', '대출', '펀드', '투자', '육성'
]

def needs_announcement(query):
    return any(kw in query for kw in SUPPORT_KEYWORDS)

def clean_text(text):
    if not isinstance(text, str):
        return ''
    text = re.sub(r'저는\s+.{0,30}비즈니스지원단.+?감사합니다\.?', '', text, flags=re.DOTALL)
    text = re.sub(r'\d{2,4}-\d{3,4}-\d{4}(~\d+)?', '', text)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'[^\w\s가-힣.,?!()]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

@st.cache_resource
def load_data_and_model():
    df = pd.read_csv('data.csv', encoding='utf-8')
    df['대표답변'] = df['답변1']
    df = df[['번호', '지원분야', '제목', '게시글', '대표답변']].copy()
    df = df.dropna(subset=['게시글', '대표답변']).reset_index(drop=True)
    df['게시글_정제']   = df['게시글'].apply(clean_text)
    df['대표답변_정제'] = df['대표답변'].apply(clean_text)
    df['검색용텍스트']  = df['제목'].fillna('') + ' ' + df['게시글_정제']
    model      = SentenceTransformer('jhgan/ko-sroberta-multitask')
    embeddings = model.encode(df['검색용텍스트'].tolist(), show_progress_bar=False)
    return df, model, embeddings

def retrieve(query, df, model, embeddings, top_k=3):
    query_embedding = model.encode([clean_text(query)])
    similarities    = cosine_similarity(query_embedding, embeddings).flatten()
    top_indices     = similarities.argsort()[::-1][:top_k]
    results = []
    for rank, idx in enumerate(top_indices, 1):
        results.append({
            '순위':     rank,
            '유사도':   round(float(similarities[idx]), 4),
            '지원분야': df.iloc[idx]['지원분야'],
            '제목':     df.iloc[idx]['제목'],
            '질문':     df.iloc[idx]['게시글_정제'],
            '답변':     df.iloc[idx]['대표답변_정제'],
        })
    return results

def get_bizinfo_announcements(keyword, num_rows=3):
    url    = 'https://apis.data.go.kr/1421000/bizinfo/pblancBsnsService'
    params = {'serviceKey': GOV_API_KEY, 'returnType': 'json', 'numOfRows': num_rows, 'pageNo': 1, 'bsnsSumryCn': keyword}
    try:
        res = requests.get(url, params=params, timeout=10)
        data = res.json()
        items = data.get('response', {}).get('body', {}).get('items', {})
        if not items: return []
        item_list = items.get('item', [])
        if isinstance(item_list, dict): item_list = [item_list]
        return [{'출처': '[비즈니스지원단]', '공고명': i.get('pblancNm',''), '지원내용': i.get('bsnsSumryCn','')[:200], '신청기간': f"{i.get('reqstBgnDe','')} ~ {i.get('reqstEndDe','')}", '담당기관': i.get('mngtMssofcNm','')} for i in item_list]
    except: return []

def get_mss_announcements(keyword, num_rows=3):
    url    = 'https://apis.data.go.kr/1421000/mssBizService_v2/getbizList_v2'
    params = {'serviceKey': GOV_API_KEY, 'returnType': 'json', 'numOfRows': num_rows, 'pageNo': 1, 'searchWord': keyword}
    try:
        res = requests.get(url, params=params, timeout=10)
        data = res.json()
        items = data.get('response', {}).get('body', {}).get('items', {})
        if not items: return []
        item_list = items.get('item', [])
        if isinstance(item_list, dict): item_list = [item_list]
        return [{'출처': '[중소벤처기업부]', '공고명': i.get('bizNm',''), '지원내용': i.get('bizSumryCn','')[:200], '신청기간': f"{i.get('reqstBgnDe','')} ~ {i.get('reqstEndDe','')}", '담당기관': i.get('mngtMssofcNm','')} for i in item_list]
    except: return []

def get_all_announcements(keyword, num_rows=3):
    return get_bizinfo_announcements(keyword, num_rows) + get_mss_announcements(keyword, num_rows)

def generate_answer(query, retrieved_cases, announcements):
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    cases_context = '\n\n'.join([
        f"[유사 사례 {c['순위']} | 분야: {c['지원분야']} | 유사도: {c['유사도']}]\n질문: {c['질문']}\n답변: {c['답변']}"
        for c in retrieved_cases
    ])
    ann_context = '\n\n'.join([
        f"[공고명: {a['공고명']}]\n지원내용: {a['지원내용']}\n신청기간: {a['신청기간']}\n담당기관: {a['담당기관']}"
        for a in announcements
    ]) if announcements else '현재 관련 공고 없음'

    prompt = f"""당신은 중소벤처기업부 비즈니스지원단의 전문 상담위원입니다.
아래의 과거 상담사례와 현재 진행 중인 지원사업 공고를 참고하여 답변해주세요.

=== 과거 유사 상담사례 ===
{cases_context}

=== 현재 진행 중인 지원사업 공고 (최신) ===
{ann_context}

=== 새로운 질문 ===
{query}

=== 답변 지침 ===
- 과거 사례를 바탕으로 일반적인 절차와 방법을 안내하세요.
- 현재 신청 가능한 공고가 있다면 구체적으로 안내하세요.
- 항목별로 나누어 구체적이고 상세하게 작성하세요.
- 관련 기관, 절차, 주의사항까지 포함하세요."""

    message = client.messages.create(
        model='claude-sonnet-4-20250514',
        max_tokens=2000,
        messages=[{'role': 'user', 'content': prompt}]
    )
    return message.content[0].text

# ── 메인 UI ──────────────────────────────────────────────
with st.spinner('데이터 및 모델 로딩 중... (처음 실행 시 1~2분 소요)'):
    df, sbert_model, corpus_embeddings = load_data_and_model()

st.success(f'✅ {len(df)}개 상담 사례 로드 완료')
st.divider()

query = st.text_area(
    '질문을 입력하세요',
    placeholder='예) 직원이 퇴직하는데 퇴직금 계산을 어떻게 해야 하나요?',
    height=100
)

if st.button('답변 받기', type='primary', use_container_width=True):
    if not query.strip():
        st.warning('질문을 입력해주세요.')
    else:
        # 유사 사례 검색 (항상 실행)
        with st.spinner('유사 상담사례 검색 중...'):
            retrieved = retrieve(query, df, sbert_model, corpus_embeddings)

        with st.expander('🔍 유사 상담사례 검색 결과', expanded=False):
            for r in retrieved:
                st.markdown(f"**{r['순위']}위** | 유사도: `{r['유사도']}` | {r['지원분야']}")
                st.markdown(f"📌 {r['제목']}")
                st.divider()

        # 지원사업 관련 질문일 때만 공고 검색
        announcements = []
        if needs_announcement(query):
            with st.spinner('최신 지원사업 공고 검색 중...'):
                announcements = get_all_announcements(query)
            if announcements:
                with st.expander('📡 관련 지원사업 공고', expanded=False):
                    for a in announcements:
                        st.markdown(f"**{a['출처']} {a['공고명']}**")
                        st.markdown(f"신청기간: {a['신청기간']} | 담당기관: {a['담당기관']}")
                        st.divider()

        # 답변 생성
        with st.spinner('AI 답변 생성 중...'):
            answer = generate_answer(query, retrieved, announcements)

        st.subheader('✅ AI 답변')
        st.markdown(answer)

        # 만족도 확인 및 전문가 연결
        st.divider()
        st.write("💬 이 답변이 도움이 됐나요?")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("✅ 도움됐어요", use_container_width=True):
                st.success("감사합니다! 더 궁금한 점이 있으면 언제든지 질문해주세요.")

        with col2:
            if st.button("❌ 더 자세한 상담이 필요해요", use_container_width=True):
                st.warning("전문가 심층 상담 게시판으로 연결해드릴게요.")
                st.info("""
**비즈니스지원단 전문가 심층 상담 신청**

AI 답변으로 해결되지 않은 경우, 아래 게시판에 질문을 남기시면
비즈니스지원단 전문 상담위원이 직접 답변해드립니다.

- 🕐 운영시간: 평일 09:00 ~ 18:00
                """)
                st.link_button(
                    "📋 심층 상담 게시판 바로가기",
                    "https://www.smes.go.kr/bizlink/problem/problemList.do#a",
                    use_container_width=True
                )
