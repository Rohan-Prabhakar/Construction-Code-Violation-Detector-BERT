import streamlit as st
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="ViolationBERT", page_icon="🏗️", layout="wide")

HF_REPO = "Rohan1103/ViolationBERT"  # change to your HF username

@st.cache_resource
def load_model():
    class ViolationClassifier(nn.Module):
        def __init__(self, model_name, num_categories, num_severities, dropout=0.3):
            super().__init__()
            self.encoder = AutoModel.from_pretrained(model_name)
            h = self.encoder.config.hidden_size
            self.dropout = nn.Dropout(dropout)
            self.category_head = nn.Sequential(
                nn.Linear(h, 256), nn.ReLU(), nn.Dropout(dropout), nn.Linear(256, num_categories))
            self.severity_head = nn.Sequential(
                nn.Linear(h, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, num_severities))
        def forward(self, input_ids, attention_mask):
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            cls = self.dropout(out.last_hidden_state[:, 0, :])
            return self.category_head(cls), self.severity_head(cls)

    model_path = hf_hub_download(repo_id=HF_REPO, filename="final_model.pt")
    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    cfg = ckpt['model_config']
    lm = ckpt['label_maps']

    model = ViolationClassifier(cfg['model_name'], cfg['num_categories'], cfg['num_severities'], cfg['dropout'])
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(cfg['model_name'])
    id2cat = {int(k): v for k, v in lm['id2cat'].items()}
    id2sev = {int(k): v for k, v in lm['id2sev'].items()}

    return model, tokenizer, id2cat, id2sev

@torch.no_grad()
def predict(text, model, tokenizer, id2cat, id2sev):
    enc = tokenizer(text.upper(), max_length=256, padding='max_length', truncation=True, return_tensors='pt')
    c_log, s_log = model(enc['input_ids'], enc['attention_mask'])
    c_probs = torch.softmax(c_log, 1).numpy()[0]
    s_probs = torch.softmax(s_log, 1).numpy()[0]
    return {
        'category': id2cat[c_probs.argmax()],
        'cat_conf': float(c_probs.max()),
        'cat_all': {id2cat[i]: float(p) for i, p in enumerate(c_probs)},
        'severity': id2sev[s_probs.argmax()],
        'sev_conf': float(s_probs.max()),
        'sev_all': {id2sev[i]: float(p) for i, p in enumerate(s_probs)},
    }

model, tokenizer, id2cat, id2sev = load_model()

st.title("🏗️ ViolationBERT")
st.subheader("Building Code Violation Classifier")
st.caption("Fine-tuned RoBERTa model classifying NYC building violations by category and severity")

st.markdown("##")

col1, col2 = st.columns([1.2, 1])

with col1:
    text_input = st.text_area(
        "Enter a violation description:",
        height=150,
        placeholder="e.g. FAILURE TO MAINTAIN BUILDING WALL NOTED BRICKS FALLING FROM FACADE POSING DANGER TO PEDESTRIANS"
    )

    examples = [
        "FAILURE TO MAINTAIN BUILDING WALL NOTED BRICKS FALLING FROM FACADE POSING DANGER TO PEDESTRIANS",
        "WORK WITHOUT A PERMIT CONTRACTOR PERFORMING ELECTRICAL WORK ON 3RD FLOOR WITHOUT DOB APPROVAL",
        "ELEVATOR INSPECTION OVERDUE CERTIFICATE EXPIRED LAST YEAR BUILDING HAS 6 PASSENGER ELEVATORS",
        "FENCE EXCEEDS PERMITTED HEIGHT IN FRONT YARD SETBACK AREA ZONING VIOLATION",
        "FAILURE TO PROVIDE SITE SAFETY MANAGER DURING ACTIVE DEMOLITION OF 5 STORY BUILDING",
        "BOILER FAILED ANNUAL INSPECTION DUE TO CRACKED HEAT EXCHANGER AND GAS LEAK DETECTED",
        "ILLEGAL CONVERSION OF COMMERCIAL SPACE TO RESIDENTIAL USE WITHOUT CERTIFICATE OF OCCUPANCY",
        "EXIT DOOR NOT SELF CLOSING ON 2ND FLOOR OF PUBLIC ASSEMBLY SPACE CAPACITY 300 PERSONS",
    ]

    st.markdown("**Try an example:**")
    selected = st.selectbox("Select example", ["(type your own)"] + examples, label_visibility="collapsed")
    if selected != "(type your own)":
        text_input = selected

    predict_btn = st.button("Classify Violation", type="primary", use_container_width=True)

with col2:
    if predict_btn and text_input.strip():
        result = predict(text_input, model, tokenizer, id2cat, id2sev)

        sev_colors = {'HIGH': '🔴', 'MEDIUM': '🟡', 'LOW': '🟢'}

        st.markdown("### Results")

        st.markdown(f"**Category:** `{result['category']}`  ({result['cat_conf']*100:.1f}% confidence)")
        st.markdown(f"**Severity:** {sev_colors.get(result['severity'], '')} `{result['severity']}`  ({result['sev_conf']*100:.1f}% confidence)")

        if result['sev_conf'] < 0.6:
            st.warning("Low confidence on severity. Flag for human review.")
        if result['severity'] == 'HIGH' and result['sev_conf'] > 0.8:
            st.error("HAZARDOUS: This violation requires immediate attention.")

        st.markdown("####")
        st.markdown("**Category Probabilities:**")
        for cat, prob in sorted(result['cat_all'].items(), key=lambda x: x[1], reverse=True):
            st.progress(prob, text=f"{cat}: {prob*100:.1f}%")

        st.markdown("**Severity Probabilities:**")
        for sev, prob in sorted(result['sev_all'].items(), key=lambda x: x[1], reverse=True):
            st.progress(prob, text=f"{sev}: {prob*100:.1f}%")

    elif predict_btn:
        st.warning("Please enter a violation description.")

st.markdown("##")

mcol1, mcol2, mcol3 = st.columns(3)
with mcol1:
    st.metric("Category F1", "0.898")
with mcol2:
    st.metric("Severity F1", "0.864")
with mcol3:
    st.metric("Training Data", "189K samples")

st.markdown("##")
with st.expander("About this model"):
    st.markdown("""
    **ViolationBERT** is a fine-tuned RoBERTa-base model trained on 189,000+ NYC Department of Buildings 
    violation records. It performs dual classification:
    
    - **8 violation categories**: Construction, Elevators, Mechanical, Plumbing, Quality of Life, Regulatory, Site Safety, Zoning
    - **3 severity levels**: HIGH (immediately hazardous), MEDIUM (significant), LOW (minor)
    
    The model was trained using weighted cross-entropy loss to handle class imbalance, with AdamW optimizer 
    and linear warmup scheduling. Best hyperparameters: lr=1e-5, dropout=0.1, batch_size=32.
    
    **Limitations**: The model misclassifies 6.7% of HIGH severity violations as LOW. It should NOT be used 
    as a standalone system for safety decisions without human oversight.
    
    Dataset: NYC DOB ECB Violations (NYC Open Data)
    """)