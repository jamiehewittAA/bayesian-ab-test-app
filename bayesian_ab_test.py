import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

# Reset button
if st.button("🔄 Reset Calculator"):
    st.session_state.clear()
    st.rerun()

# Page setup
st.set_page_config(page_title="Bayesian A/B Test Calculator", layout="centered")

st.title("🧪 Easy Bayesian A/B Test Calculator")
st.markdown("""
This calculator uses **Bayesian statistics** to evaluate A/B test results clearly and accurately.  
It gives you practical insights, even if you don’t have a statistics background.
""")

# Toggle for plain-English or stats mode
simple_mode = st.toggle("🧠 Show plain-English explanations", value=True)

# --- 1. Input Data ---
st.header("1️⃣ Enter Your A/B Test Data")
with st.expander("ℹ️ What are 'Visitors' and 'Conversions'?"):
    st.markdown("""
    - **Visitors**: Number of users who saw each version.
    - **Conversions**: Users who completed your goal (e.g. signed up, purchased).
    - This tool compares the **conversion rates** between A and B.
    """)

col1, col2 = st.columns(2)
with col1:
    visitors_a = st.number_input("Visitors to A (Original)", min_value=1, value=1000)
    conversions_a = st.number_input("Conversions from A", min_value=0, value=50)
with col2:
    visitors_b = st.number_input("Visitors to B (Variant)", min_value=1, value=1000)
    conversions_b = st.number_input("Conversions from B", min_value=0, value=70)

# --- 2. Prior Beliefs ---
st.header("2️⃣ Prior Beliefs (Optional)")
st.markdown("If you’re unsure, leave both values at 1 for a neutral starting point.")
col3, col4 = st.columns(2)
with col3:
    alpha_prior = st.number_input("Prior Alpha (α)", min_value=0.01, value=1.0)
with col4:
    beta_prior = st.number_input("Prior Beta (β)", min_value=0.01, value=1.0)

# --- 3. Confidence Threshold ---
st.header("3️⃣ Confidence Level")
confidence_choice = st.selectbox("How confident do you want to be in your result?", [95, 90, 80], index=0)
prob_threshold = confidence_choice / 100.0
ci_tail = (1 - prob_threshold) / 2 * 100
ci_low_percentile = ci_tail
ci_high_percentile = 100 - ci_tail

# --- 4. ROPE ---
st.header("4️⃣ Minimum Meaningful Difference (ROPE)")
practical_effect = st.slider(
    "Ignore differences smaller than this (0%–5%)",
    min_value=0.0,
    max_value=0.05,
    value=0.005,
    step=0.001,
    format="%.1f%%",
    help="ROPE = Region of Practical Equivalence: differences smaller than this are considered too small to matter."
)

# --- 5. Daily Traffic for Estimating Test Duration ---
daily_traffic = st.number_input("Daily visitors to A + B", min_value=1, value=2000, help="Used to estimate days left if your result isn’t robust yet.")

# --- Bayesian Inference ---
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
ci_low, ci_high = np.percentile(delta, [ci_low_percentile, ci_high_percentile])
ci_width = ci_high - ci_low
in_rope = np.mean((delta > -practical_effect) & (delta < practical_effect))
statistically_significant = ci_low > 0 or ci_high < 0
robust = ci_width < 0.01 and statistically_significant and in_rope < 0.95

# --- Data Needed Estimate ---
scale_factor = (ci_width / 0.01) ** 2 if ci_width > 0 else 1
suggested_total_visitors = int((visitors_a + visitors_b) * scale_factor)
additional_visitors = max(suggested_total_visitors - (visitors_a + visitors_b), 0)
estimated_days = int(np.ceil(additional_visitors / daily_traffic)) if daily_traffic else None

# --- Result Summary ---
st.header("📊 Summary of Your A/B Test")

if simple_mode:
    if prob_b_better >= prob_threshold:
        st.success(f"✅ B is likely better than A, with about **{prob_b_better*100:.1f}% certainty**.")
    else:
        st.warning(f"⚠️ We can't be confident B is better than A (only {prob_b_better*100:.1f}% certainty).")

    st.markdown(f"📈 The expected improvement is about **{expected_lift:.2f}%**.")

    if in_rope > 0.95:
        st.info("ℹ️ The improvement is probably too small to matter.")
    elif in_rope < 0.05:
        st.success("✅ The improvement is likely meaningful.")
    else:
        st.warning("⚠️ It’s unclear if the difference is big enough to act on.")

    if robust:
        st.success("🎯 This result looks reliable and trustworthy.")
    else:
        st.warning("🚧 The result isn’t stable yet — more data is needed.")
        st.markdown(f"📊 You should collect about **{additional_visitors:,} more visitors** (≈ **{estimated_days} more days**) before deciding.")
else:
    st.write(f"Probability B > A: {prob_b_better*100:.2f}%")
    st.write(f"Expected Lift: {expected_lift:.2f}%")
    st.write(f"{confidence_choice}% Credible Interval Width: {ci_width:.4f}")
    st.write(f"ROPE Overlap: {in_rope*100:.1f}%")
    st.write("Statistically Significant:", statistically_significant)
    st.write("Robust:", robust)

# --- Posterior Distributions ---
st.header("📈 Posterior Distributions")

x = np.linspace(0, max(posterior_a.max(), posterior_b.max()) * 1.1, 1000)
fig1, ax1 = plt.subplots(figsize=(8, 4))
ax1.plot(x, beta.pdf(x, alpha_a, beta_a), label="A", linewidth=2)
ax1.plot(x, beta.pdf(x, alpha_b, beta_b), label="B", linewidth=2)
ax1.set_xlabel("Estimated Conversion Rate")
ax1.set_ylabel("Density")
ax1.set_title("Posterior Distributions for A and B")
ax1.legend()
st.pyplot(fig1)

# --- Delta Histogram ---
st.subheader("📉 Difference in Conversion Rates (B − A)")
fig2, ax2 = plt.subplots(figsize=(7, 3))
ax2.hist(delta, bins=100, color="gray", alpha=0.7)
ax2.axvline(0, color='red', linestyle='--', label="No Difference")
ax2.axvline(-practical_effect, color='blue', linestyle=':', label="ROPE Bounds")
ax2.axvline(practical_effect, color='blue', linestyle=':')
ax2.set_xlabel("Conversion Rate Difference (B - A)")
ax2.set_ylabel("Frequency")
ax2.set_title("Posterior Distribution of the Difference")
ax2.legend()
st.pyplot(fig2)

st.caption("Built for CRO professionals. 100% Bayesian. No jargon. Just insights. 🚀")
