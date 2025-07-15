import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

# Set Streamlit page config
st.set_page_config(page_title="Bayesian A/B Test Calculator", layout="centered")

# Optional: Refresh button to clear all session data
if st.button("🔄 Reset Calculator"):
    st.session_state.clear()
    st.experimental_rerun()

# App title
st.title("🧪 Easy Bayesian A/B Test Calculator")
st.markdown("""
This calculator helps you compare A/B test results using **Bayesian statistics**.  
It checks both **statistical confidence** and **practical significance** to help you avoid false positives and negatives.
""")

# --- INPUTS ---
st.header("1️⃣ Enter Your Test Data")

with st.expander("ℹ️ What do 'Visitors' and 'Conversions' mean?", expanded=False):
    st.markdown("""
    - **Visitors**: Number of users who saw each version.
    - **Conversions**: Number of users who completed the desired action (e.g., signed up or purchased).
    - This calculator compares the **conversion rates** of group A and group B.
    """)

col1, col2 = st.columns(2)
with col1:
    visitors_a = st.number_input("Visitors to A (Original)", min_value=1, value=1000)
    conversions_a = st.number_input("Conversions from A", min_value=0, value=50)
with col2:
    visitors_b = st.number_input("Visitors to B (Test Variant)", min_value=1, value=1000)
    conversions_b = st.number_input("Conversions from B", min_value=0, value=70)

# --- PRIORS ---
st.header("2️⃣ Set Prior Beliefs (Optional)")
st.markdown("These numbers reflect how confident you are **before** the test. If unsure, leave both at 1.")

col3, col4 = st.columns(2)
with col3:
    alpha_prior = st.number_input("Prior Alpha (α)", min_value=0.01, value=1.0)
with col4:
    beta_prior = st.number_input("Prior Beta (β)", min_value=0.01, value=1.0)

# --- THRESHOLD SELECTION ---
st.header("3️⃣ Choose Your Confidence Threshold")
confidence_choice = st.selectbox(
    "Pick how sure you want to be before calling B a winner:",
    options=[95, 90, 80],
    index=0
)
prob_threshold = confidence_choice / 100.0

# --- PRACTICAL SIGNIFICANCE (ROPE) ---
st.header("4️⃣ Set Minimum Meaningful Difference")
practical_effect = st.slider(
    "What is the smallest improvement you care about?",
    min_value=0.0, max_value=0.05, value=0.005, step=0.001,
    help="Helps filter out 'false winners' — results that are technically better but not big enough to matter."
)

# --- POSTERIOR CALCULATION ---
alpha_a = alpha_prior + conversions_a
beta_a = beta_prior + visitors_a - conversions_a
alpha_b = alpha_prior + conversions_b
beta_b = beta_prior + visitors_b - conversions_b

samples = 200_000
posterior_a = np.random.beta(alpha_a, beta_a, samples)
posterior_b = np.random.beta(alpha_b, beta_b, samples)
delta = posterior_b - posterior_a
lift = (delta / posterior_a) * 100

prob_b_better = np.mean(delta > 0)
expected_lift = np.mean(lift)

ci_low, ci_high = np.percentile(delta, [2.5, 97.5])
ci_width = ci_high - ci_low
ci_robust = ci_width < 0.01
in_rope = np.mean((delta > -practical_effect) & (delta < practical_effect))

sufficient_data = all([
    visitors_a >= 1000,
    visitors_b >= 1000,
    conversions_a >= 20,
    conversions_b >= 20
])

# --- RESULTS ---
st.header("📊 Results Summary")

col5, col6 = st.columns(2)
with col5:
    st.metric("Probability B > A", f"{prob_b_better * 100:.2f}%")
    st.metric("Expected Lift", f"{expected_lift:.2f}%")
with col6:
    st.metric("95% Credible Interval Width", f"{ci_width:.4f}")
    st.metric("In ROPE Zone", f"{in_rope * 100:.1f}%")

st.subheader("🧠 Interpretation")

if prob_b_better >= prob_threshold:
    st.success(f"✅ High confidence: B is better than A with ≥ {confidence_choice}% certainty.")
else:
    st.warning(f"⚠️ Not enough evidence yet to say B is better with {confidence_choice}% confidence.")

if ci_robust:
    st.success("✅ Statistically robust (narrow credible interval).")
else:
    st.warning("⚠️ Wide credible interval: collect more data.")

if in_rope > 0.95:
    st.info("ℹ️ The result is likely too small to matter in real-world terms.")
elif in_rope < 0.05:
    st.success("✅ The difference is likely meaningful.")
else:
    st.warning("⚠️ The difference might be too small to care about.")

if sufficient_data:
    st.success("✅ Your sample size looks sufficient.")
else:
    st.warning("⚠️ Consider gathering more traffic to increase reliability.")

# --- VISUALIZATION ---
st.header("📈 Conversion Rate Distributions")

x = np.linspace(0, max(posterior_a.max(), posterior_b.max()), 1000)
pdf_a = beta.pdf(x, alpha_a, beta_a)
pdf_b = beta.pdf(x, alpha_b, beta_b)

fig1, ax1 = plt.subplots(figsize=(8, 4))
ax1.plot(x, pdf_a, label='Control (A)', linewidth=2)
ax1.plot(x, pdf_b, label='Variant (B)', linewidth=2)
ax1.set_xlabel("Conversion Rate")
ax1.set_ylabel("Density")
ax1.set_title("Posterior Distributions")
ax1.legend()
st.pyplot(fig1)

st.subheader("📉 Difference in Conversion Rates (B - A)")

fig2, ax2 = plt.subplots(figsize=(7, 3))
ax2.hist(delta, bins=100, color="gray", alpha=0.7)
ax2.axvline(0, color='red', linestyle='--', label="No Difference")
ax2.axvline(-practical_effect, color='blue', linestyle=':', label="ROPE bounds")
ax2.axvline(practical_effect, color='blue', linestyle=':')
ax2.set_xlabel("Difference in Conversion Rate (B - A)")
ax2.set_ylabel("Frequency")
ax2.set_title("Posterior Distribution of the Difference")
ax2.legend()
st.pyplot(fig2)

st.caption("Made for CRO professionals who care about accuracy and real-world significance. 🚀")

Sent from Outlook for Android
From: Jamie Hewitt <jamiehewitt@outlook.com>
Sent: Tuesday, July 15, 2025 5:12:12 PM
To: jamie.hewitt@theaa.com <jamie.hewitt@theaa.com>
Subject: Re: Bayesian
 
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

Sent from Outlook for Android
From: Jamie Hewitt <jamiehewitt@outlook.com>
Sent: Tuesday, July 15, 2025 4:55:14 PM
To: jamie.hewitt@theaa.com <jamie.hewitt@theaa.com>
Subject: Re: Bayesian
 
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

st.set_page_config(page_title="Bayesian A/B Test Calculator", layout="centered")

st.title("🧪 Bayesian A/B Testing Calculator")

st.markdown("""
Use this tool to compare conversion rates of two variants (A and B) using Bayesian inference.  
You can adjust priors and evaluate if your test has reached **90%** or **95%** probability.
""")

st.header("📥 Input Data")

col1, col2 = st.columns(2)
with col1:
    visitors_a = st.number_input("Visitors - Control (A)", min_value=1, value=1000)
    conversions_a = st.number_input("Conversions - Control (A)", min_value=0, value=50)
with col2:
    visitors_b = st.number_input("Visitors - Variant (B)", min_value=1, value=1000)
    conversions_b = st.number_input("Conversions - Variant (B)", min_value=0, value=70)

st.header("⚙️ Bayesian Priors (Beta Distribution)")

col3, col4 = st.columns(2)
with col3:
    alpha_prior = st.number_input("Alpha (α)", min_value=0.01, value=1.0)
with col4:
    beta_prior = st.number_input("Beta (β)", min_value=0.01, value=1.0)

# Posterior distributions
alpha_a = alpha_prior + conversions_a
beta_a = beta_prior + visitors_a - conversions_a
alpha_b = alpha_prior + conversions_b
beta_b = beta_prior + visitors_b - conversions_b

# Sampling from posteriors
samples = 100_000
posterior_a = np.random.beta(alpha_a, beta_a, samples)
posterior_b = np.random.beta(alpha_b, beta_b, samples)

prob_b_better = np.mean(posterior_b > posterior_a)
lift = np.mean((posterior_b - posterior_a) / posterior_a) * 100

# Robustness check (you can tune this)
robust = (visitors_a >= 1000 and visitors_b >= 1000 and conversions_a >= 20 and conversions_b >= 20)

st.header("📊 Results")

st.metric("Probability B > A", f"{prob_b_better*100:.2f}%")
st.metric("Expected Lift", f"{lift:.2f}%")

if prob_b_better >= 0.95:
    st.success("✅ Statistically confident at 95%.")
elif prob_b_better >= 0.90:
    st.info("ℹ️ Statistically confident at 90%.")
else:
    st.warning("⚠️ Not statistically confident yet.")

if robust:
    st.success("✅ Traffic is sufficient for a robust result.")
else:
    st.warning("⚠️ Consider collecting more data for robustness.")

st.header("📈 Posterior Distributions")

x = np.linspace(0, max(posterior_a.max(), posterior_b.max()), 1000)
pdf_a = beta.pdf(x, alpha_a, beta_a)
pdf_b = beta.pdf(x, alpha_b, beta_b)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(x, pdf_a, label='Control (A)', linewidth=2)
ax.plot(x, pdf_b, label='Variant (B)', linewidth=2)
ax.set_title("Posterior Distributions of Conversion Rates")
ax.set_xlabel("Conversion Rate")
ax.set_ylabel("Density")
ax.legend()
st.pyplot(fig)

st.caption("Built with ❤️ by an expert in Bayesian CRO testing.")

Sent from Outlook for Android
From: Jamie Hewitt <jamiehewitt@outlook.com>
Sent: Tuesday, July 15, 2025 4:51:15 PM
To: jamie.hewitt@theaa.com <jamie.hewitt@theaa.com>
Subject: Bayesian
 
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

st.set_page_config(page_title="Bayesian A/B Test Calculator", layout="centered")

st.title("🧪 Bayesian A/B Testing Calculator")

st.markdown("""
Use this tool to compare conversion rates of two variants (A and B) using Bayesian inference.  
You can adjust priors and evaluate if your test has reached **90%** or **95%** probability.
""")

st.header("📥 Input Data")

col1, col2 = st.columns(2)
with col1:
    visitors_a = st.number_input("Visitors - Control (A)", min_value=1, value=1000)
    conversions_a = st.number_input("Conversions - Control (A)", min_value=0, value=50)
with col2:
    visitors_b = st.number_input("Visitors - Variant (B)", min_value=1, value=1000)
    conversions_b = st.number_input("Conversions - Variant (B)", min_value=0, value=70)

st.header("⚙️ Bayesian Priors (Beta Distribution)")

col3, col4 = st.columns(2)
with col3:
    alpha_prior = st.number_input("Alpha (α)", min_value=0.01, value=1.0)
with col4:
    beta_prior = st.number_input("Beta (β)", min_value=0.01, value=1.0)

# Posterior distributions
alpha_a = alpha_prior + conversions_a
beta_a = beta_prior + visitors_a - conversions_a
alpha_b = alpha_prior + conversions_b
beta_b = beta_prior + visitors_b - conversions_b

# Sampling from posteriors
samples = 100_000
posterior_a = np.random.beta(alpha_a, beta_a, samples)
posterior_b = np.random.beta(alpha_b, beta_b, samples)

prob_b_better = np.mean(posterior_b > posterior_a)
lift = np.mean((posterior_b - posterior_a) / posterior_a) * 100

# Robustness check (you can tune this)
robust = (visitors_a >= 1000 and visitors_b >= 1000 and conversions_a >= 20 and conversions_b >= 20)

st.header("📊 Results")

st.metric("Probability B > A", f"{prob_b_better*100:.2f}%")
st.metric("Expected Lift", f"{lift:.2f}%")

if prob_b_better >= 0.95:
    st.success("✅ Statistically confident at 95%.")
elif prob_b_better >= 0.90:
    st.info("ℹ️ Statistically confident at 90%.")
else:
    st.warning("⚠️ Not statistically confident yet.")

if robust:
    st.success("✅ Traffic is sufficient for a robust result.")
else:
    st.warning("⚠️ Consider collecting more data for robustness.")

st.header("📈 Posterior Distributions")

x = np.linspace(0, max(posterior_a.max(), posterior_b.max()), 1000)
pdf_a = beta.pdf(x, alpha_a, beta_a)
pdf_b = beta.pdf(x, alpha_b, beta_b)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(x, pdf_a, label='Control (A)', linewidth=2)
ax.plot(x, pdf_b, label='Variant (B)', linewidth=2)
ax.set_title("Posterior Distributions of Conversion Rates")
ax.set_xlabel("Conversion Rate")
ax.set_ylabel("Density")
ax.legend()
st.pyplot(fig)

st.caption("Built with ❤️ by an expert in Bayesian CRO testing.")
