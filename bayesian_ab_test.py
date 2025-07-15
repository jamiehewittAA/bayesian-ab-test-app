import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

# Reset button
if st.button("🔄 Reset Calculator"):
    st.session_state.clear()
    st.rerun()

# Page config
st.set_page_config(page_title="Bayesian A/B Test Calculator", layout="centered")

# App title
st.title("🧪 Easy Bayesian A/B Test Calculator")
st.markdown("""
This calculator uses **Bayesian statistics** to help you evaluate A/B test results.  
It controls for both **statistical significance** and **practical relevance**, and helps avoid false positives and negatives.
""")

# --------------------
# 1. Input Test Data
# --------------------
st.header("1️⃣ Enter Your Test Data")

with st.expander("ℹ️ What do 'Visitors' and 'Conversions' mean?", expanded=False):
    st.markdown("""
    - **Visitors**: Number of users who saw each version.
    - **Conversions**: Number of users who completed the desired action (e.g., purchased).
    - The calculator compares **conversion rates** between group A and group B.
    """)

col1, col2 = st.columns(2)
with col1:
    visitors_a = st.number_input("Visitors to A (Original)", min_value=1, value=1000)
    conversions_a = st.number_input("Conversions from A", min_value=0, value=50)
with col2:
    visitors_b = st.number_input("Visitors to B (Variant)", min_value=1, value=1000)
    conversions_b = st.number_input("Conversions from B", min_value=0, value=70)

# --------------------
# 2. Set Priors
# --------------------
st.header("2️⃣ Set Prior Beliefs (Optional)")
st.markdown("These reflect your prior beliefs about conversion rates. Leave both at 1 for no strong prior.")

col3, col4 = st.columns(2)
with col3:
    alpha_prior = st.number_input("Prior Alpha (α)", min_value=0.01, value=1.0)
with col4:
    beta_prior = st.number_input("Prior Beta (β)", min_value=0.01, value=1.0)

# --------------------
# 3. Confidence Threshold
# --------------------
st.header("3️⃣ Choose Confidence Threshold")
confidence_choice = st.selectbox(
    "How sure do you want to be before calling B a winner?",
    options=[95, 90, 80],
    index=0
)
prob_threshold = confidence_choice / 100.0

# --------------------
# 4. Practical Significance (ROPE)
# --------------------
st.header("4️⃣ Set Practical Significance Threshold")
practical_effect = st.slider(
    "What's the smallest lift in conversion rate that would matter to you?",
    min_value=0.0, max_value=0.05, value=0.005, step=0.001,
    help="Helps ignore changes that are statistically real but too small to act on."
)

# --------------------
# Bayesian Posterior Calculation
# --------------------
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
in_rope = np.mean((delta > -practical_effect) & (delta < practical_effect))

# --------------------
# Robustness Check (Statistical + Practical)
# --------------------
statistically_significant = ci_low > 0 or ci_high < 0
robust = (
    ci_width < 0.01 and
    statistically_significant and
    in_rope < 0.95
)

# --------------------
# Results Display
# --------------------
st.header("📊 Results Summary")

col5, col6 = st.columns(2)
with col5:
    st.metric("Probability B > A", f"{prob_b_better * 100:.2f}%")
    st.metric("Expected Lift", f"{expected_lift:.2f}%")
with col6:
    st.metric("95% Credible Interval Width", f"{ci_width:.4f}")
    st.metric("In ROPE (Too Small to Matter)", f"{in_rope * 100:.1f}%")

st.subheader("🧠 Interpretation")

if prob_b_better >= prob_threshold:
    st.success(f"✅ B is better than A with ≥ {confidence_choice}% certainty.")
else:
    st.warning(f"⚠️ Not enough evidence to conclude B is better with {confidence_choice}% confidence.")

if ci_width < 0.01:
    st.success("✅ The credible interval is narrow (high precision).")
else:
    st.warning("⚠️ Wide interval: more data may improve precision.")

if statistically_significant:
    st.success("✅ The credible interval excludes 0 (statistically significant).")
else:
    st.warning("⚠️ The interval includes 0 — we can't rule out no effect.")

if in_rope > 0.95:
    st.info("ℹ️ Even if B is better, the lift is likely too small to matter.")
elif in_rope < 0.05:
    st.success("✅ The improvement is likely practically meaningful.")
else:
    st.warning("⚠️ The difference might not be large enough to matter.")

if robust:
    st.success("🎯 The result is statistically robust and practically useful.")
else:
    st.warning("⚠️ The result is not yet robust — continue testing.")

# --------------------
# Visualization: Posterior Distributions
# --------------------
st.header("📈 Posterior Distributions")

x = np.linspace(0, max(posterior_a.max(), posterior_b.max()), 1000)
pdf_a = beta.pdf(x, alpha_a, beta_a)
pdf_b = beta.pdf(x, alpha_b, beta_b)

fig1, ax1 = plt.subplots(figsize=(8, 4))
ax1.plot(x, pdf_a, label='Control (A)', linewidth=2)
ax1.plot(x, pdf_b, label='Variant (B)', linewidth=2)
ax1.set_xlabel("Conversion Rate")
ax1.set_ylabel("Density")
ax1.set_title("Posterior Distributions for A and B")
ax1.legend()
st.pyplot(fig1)

# --------------------
# Visualization: Posterior of Difference
# --------------------
st.subheader("📉 Difference in Conversion Rates (B - A)")

fig2, ax2 = plt.subplots(figsize=(7, 3))
ax2.hist(delta, bins=100, color="gray", alpha=0.7)
ax2.axvline(0, color='red', linestyle='--', label="No Difference")
ax2.axvline(-practical_effect, color='blue', linestyle=':', label="ROPE Bounds")
ax2.axvline(practical_effect, color='blue', linestyle=':')
ax2.set_xlabel("Difference in Conversion Rate")
ax2.set_ylabel("Frequency")
ax2.set_title("Posterior Distribution of the Difference")
ax2.legend()
st.pyplot(fig2)

st.caption("Built for reliable, real-world A/B testing. 🚀")
