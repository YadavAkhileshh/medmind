import streamlit as st
import requests

st.set_page_config(
    page_title="MedMind — Clinical AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
section[data-testid="stSidebar"] { display: none; }
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
.stApp { background: #111 !important; }
.block-container { padding: 2rem 2.5rem 3rem !important; max-width: 1050px !important; }
p, div, span, h1, h2, h3, li { color: #e8e8e8 !important; }
label { color: #888 !important; font-size: 0.72rem !important; text-transform: uppercase !important; letter-spacing: 0.06em !important; font-weight: 400 !important; }
.stTextArea textarea { background: #1a1a1a !important; color: #e8e8e8 !important; border: 1px solid #2d2d2d !important; border-radius: 8px !important; font-size: 0.9rem !important; line-height: 1.6 !important; }
.stTextArea textarea:focus { border-color: #555 !important; box-shadow: none !important; outline: none !important; }
.stTextArea textarea::placeholder { color: #3d3d3d !important; }
.stTextInput > div > div > input { background: #1a1a1a !important; color: #e8e8e8 !important; border: 1px solid #2d2d2d !important; border-radius: 8px !important; font-size: 0.88rem !important; }
.stTextInput > div > div > input:focus { border-color: #555 !important; box-shadow: none !important; }
.stTextInput > div > div > input::placeholder { color: #3d3d3d !important; }
.stNumberInput > div > div > input { background: #1a1a1a !important; color: #e8e8e8 !important; border: 1px solid #2d2d2d !important; border-radius: 8px !important; }
div[data-baseweb="select"] > div { background: #1a1a1a !important; border: 1px solid #2d2d2d !important; border-radius: 8px !important; color: #e8e8e8 !important; }
div[data-baseweb="select"] span { color: #e8e8e8 !important; }
div[data-baseweb="menu"] { background: #1a1a1a !important; border: 1px solid #2d2d2d !important; }
div[data-baseweb="menu"] li { color: #e8e8e8 !important; background: transparent !important; }
div[data-baseweb="menu"] li:hover { background: #2d2d2d !important; }
.stButton > button { background: #fff !important; color: #111 !important; border: none !important; border-radius: 8px !important; font-size: 0.88rem !important; font-weight: 600 !important; width: 100% !important; height: 44px !important; cursor: pointer !important; }
.stButton > button:hover { background: #ddd !important; }
.stButton > button:active { background: #bbb !important; }
.stButton > button p { color: #111 !important; }
.stSpinner > div { border-top-color: #555 !important; }
.stAlert { background: #1a1a1a !important; border-radius: 8px !important; }
.stAlert p { color: #e8e8e8 !important; }
</style>
""", unsafe_allow_html=True)

# header
st.markdown("""
<div style="display:flex;align-items:center;justify-content:space-between;padding-bottom:1.5rem;border-bottom:1px solid #222;margin-bottom:2rem;">
  <div>
    <div style="font-size:1.5rem;font-weight:600;color:#e8e8e8;">MedMind</div>
    <div style="font-size:0.75rem;color:#555;margin-top:3px;">Clinical Decision Support · Fine-tuned LLM + RAG</div>
  </div>
</div>
""", unsafe_allow_html=True)

# stats
s1, s2, s3, s4 = st.columns(4)
for col, num, lbl in [
    (s1, "10,174", "Training samples"),
    (s2, "31%",    "MedQA accuracy"),
    (s3, "1.3B",   "Parameters"),
    (s4, "RAG",    "Architecture"),
]:
    col.markdown(f"""
    <div style="background:#1a1a1a;border:1px solid #222;border-radius:8px;padding:1rem 1.2rem;margin-bottom:1.5rem;">
      <div style="font-size:1.5rem;font-weight:600;color:#e8e8e8;">{num}</div>
      <div style="font-size:0.65rem;color:#555;text-transform:uppercase;letter-spacing:0.07em;margin-top:4px;">{lbl}</div>
    </div>
    """, unsafe_allow_html=True)

# disclaimer
st.markdown("""
<div style="background:#1a1a1a;border:1px solid #2d2d2d;border-radius:8px;
padding:0.7rem 1rem;margin-bottom:1.5rem;font-size:0.75rem;color:#666;line-height:1.6;">
<b style="color:#888;">Educational Project</b> — MedMind demonstrates
end-to-end ML engineering: data pipeline, LLM fine-tuning, RAG architecture,
and API deployment. Answer quality is limited by using a 1.3B parameter model
on free compute. The architecture supports drop-in replacement with larger models.
</div>
""", unsafe_allow_html=True)

# layout
left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown("<div style='font-size:0.7rem;color:#666;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.6rem;'>Patient Input</div>", unsafe_allow_html=True)

    question = st.text_area(
        "clinical_question", label_visibility="collapsed",
        placeholder="Describe the clinical case...\n\ne.g. 35-year-old male with sudden severe headache, fever, neck stiffness for 6 hours.",
        height=160,
    )

    col_a, col_b = st.columns(2)
    with col_a:
        age = st.number_input("Patient age", min_value=0, max_value=120, value=35)
    with col_b:
        gender = st.selectbox("Gender", ["Not specified", "Male", "Female", "Other"])

    symptoms_input = st.text_input(
        "symptoms", label_visibility="collapsed",
        placeholder="Key symptoms: fever, headache, neck stiffness..."
    )

    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
    analyze = st.button("Analyze Case")

    st.markdown("<div style='height:1px;background:#222;margin:1.2rem 0;'></div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.65rem;color:#444;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.5rem;'>Try an example</div>", unsafe_allow_html=True)
    for ex in [
        "35M — sudden severe headache, fever, stiff neck",
        "23F pregnant — burning urination for 1 day",
        "58M — crushing chest pain, left arm radiation",
        "7yr child — rash face to trunk, mild fever",
    ]:
        st.markdown(f"<div style='font-size:0.78rem;color:#444;padding:0.45rem 0;border-top:1px solid #1e1e1e;line-height:1.5;'>{ex}</div>", unsafe_allow_html=True)

with right:
    st.markdown("<div style='font-size:0.7rem;color:#666;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.6rem;'>Analysis</div>", unsafe_allow_html=True)

    if analyze:
        if not question or len(question) < 10:
            st.error("Please enter a clinical question (at least 10 characters).")
        else:
            with st.spinner("Analyzing..."):
                try:
                    symptoms = [s.strip() for s in symptoms_input.split(",") if s.strip()]
                    res = requests.post(
                        "http://localhost:8000/diagnose",
                        json={"question": question, "patient_age": age, "symptoms": symptoms},
                        timeout=180
                    )

                    if res.status_code == 200:
                        data = res.json()
                        conf = data["confidence"]
                        colors = {"high": ("#0d2818", "#4ade80"), "medium": ("#251c08", "#fbbf24"), "low": ("#280d0d", "#f87171")}
                        bg, fg = colors.get(conf, ("#222", "#aaa"))

                        refs = ""
                        for i, s in enumerate(data["sources"]):
                            pct = int(s["relevance"] * 100)
                            refs += f'<div style="padding:0.65rem 0;border-top:1px solid #1e1e1e;"><div style="margin-bottom:0.3rem;"><span style="font-size:0.6rem;background:#1e1e1e;color:#666;padding:0.1rem 0.4rem;border-radius:3px;margin-right:4px;">REF {i+1}</span><span style="font-size:0.6rem;background:#1e1e1e;color:#666;padding:0.1rem 0.4rem;border-radius:3px;margin-right:4px;">{s["source"]}</span><span style="font-size:0.6rem;background:#1e1e1e;color:#666;padding:0.1rem 0.4rem;border-radius:3px;">{pct}% match</span></div><div style="font-size:0.78rem;color:#666;line-height:1.6;">{s["content"]}</div></div>'

                        html = f'<div style="background:#1a1a1a;border:1px solid #222;border-radius:10px;padding:1.5rem;"><div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:1.5rem;"><span style="font-size:0.62rem;text-transform:uppercase;letter-spacing:0.08em;padding:0.2rem 0.65rem;border-radius:4px;background:{bg};color:{fg};font-weight:600;">Confidence: {conf}</span><span style="font-size:0.6rem;color:#444;">{data["model_used"]}</span></div>{data["answer"]}<div style="font-size:0.62rem;text-transform:uppercase;letter-spacing:0.08em;color:#444;margin-bottom:0.8rem;border-top:1px solid #222;padding-top:1.5rem;">References</div>{refs}</div>'
                        st.markdown(html, unsafe_allow_html=True)
                    else:
                        st.error(f"API error: {res.status_code}")

                except requests.exceptions.ConnectionError:
                    st.error("Can't reach the API. Make sure uvicorn is running:\n\nuvicorn api.main:app --port 8000")
                except requests.exceptions.Timeout:
                    st.error("Request timed out — model may still be loading.")
                except Exception as e:
                    st.error(f"Error: {e}")
    else:
        st.markdown("""
        <div style="background:#1a1a1a;border:1px solid #222;border-radius:10px;padding:1.5rem;min-height:340px;display:flex;align-items:center;justify-content:center;">
          <div style="text-align:center;color:#2d2d2d;font-size:0.82rem;line-height:2.2;">
            <div style="font-size:1.6rem;margin-bottom:0.6rem;opacity:0.4;">+</div>
            Enter a clinical case and click Analyze
          </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("""
<div style="margin-top:3rem;padding-top:1rem;border-top:1px solid #1e1e1e;display:flex;justify-content:space-between;font-size:0.67rem;color:#333;">
  <span>MedMind — github.com/YadavAkhileshh</span>
  <span>Not for clinical use</span>
</div>
""", unsafe_allow_html=True)