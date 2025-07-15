import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

st.set_page_config(page_title="Bayesian A/B Test Calculator", layout="centered")

st.title("🧪 Bayesian A/B Testing Calculator")
st.markdown("""
Compare conversion rates using a Bayesian approach — including:
- Customizable Beta priors
- Posterior probability that Variant B > A
- Confidence flags at 90% / 95%
- Sample-size robustness check
""")

st.header("📥 Input Data")

col1, col2 = st.columns(2)
with col1:
    visitors_a = st.number_input("Visitors – Control (A)", min_value=1, value=1000)
    conversions_a = st.number_input("Conversions – Control (A)", min_value=0, value=50)
with col2:
    visitors_b = st.number_input("Visitors – Variant (B)", min_value=1, value=1000)
    conversions_b = st.number_input("Conversions – Variant (B)", min_value=0, value=70)

st.header("⚙️ Bayesian Priors (α / β for Beta)")
col3, col4 = st.columns(2)
with col3:
    alpha_prior = st.number_input("Alpha (α)", value=1.0, format="%.2f")
with col4:
    beta_prior = st.number_input("Beta (β)", value=1.0, format="%.2f")

alpha_a = alpha_prior + conversions_a
beta_a = beta_prior + visitors_a - conversions_a
alpha_b = alpha_prior + conversions_b
beta_b = beta_prior + visitors_b - conversions_b

samples = 200_000
posterior_a = np.random.beta(alpha_a, beta_a, samples)
posterior_b = np.random.beta(alpha_b, beta_b, samples)

prob_b_better = np.mean(posterior_b > posterior_a)
lift = np.mean((posterior_b - posterior_a) / posterior_a) * 100
robust = all([
    visitors_a >= 1000, visitors_b >= 1000,
    conversions_a >= 20, conversions_b >= 20
])

st.header("📊 Results")
st.metric("Probability B > A", f"{prob_b_better * 100:.2f}%")
st.metric("Expected Lift", f"{lift:.2f}%")

if prob_b_better >= 0.95:
    st.success("✅ ≥95% confidence: strong evidence.")
elif prob_b_better >= 0.90:
    st.info("ℹ️ ≥90% confidence: moderate evidence.")
else:
    st.warning("⚠️ Below 90%: more data needed.")

if robust:
    st.success("✅ Traffic meets robustness criteria.")
else:
    st.warning("⚠️ May need more data for reliability.")

st.header("📈 Posterior Distributions")
x = np.linspace(0, (posterior_b.max() + posterior_a.max()) / 2, 1000)
pdf_a = beta.pdf(x, alpha_a, beta_a)
pdf_b = beta.pdf(x, alpha_b, beta_b)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(x, pdf_a, label="Control (A)", linewidth=2)
ax.plot(x, pdf_b, label="Variant (B)", linewidth=2)
ax.set_xlabel("Conversion Rate")
ax.set_ylabel("Density")
ax.set_title("Posterior Distributions")
ax.legend()
st.pyplot(fig)

st.caption("Developed with SciPy & Streamlit.")
