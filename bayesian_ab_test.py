import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

# --- Reset Button Logic ---
if "reset" not in st.session_state:
    st.session_state.reset = False

if st.button("🔄 Reset Calculator"):
    st.session_state.reset = True

if st.session_state.reset:
    for key in list(st.session_state.keys()):
        if key != "reset":
            del st.session_state[key]
    st.session_state.reset = False
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
show_robustness_explanation = st.toggle("📘 Explain Robustness Criteria", value=False)
show_decision_mode = st.toggle("🎯 Show Decision Recommendation Mode", value=True)
no_more_traffic = st.toggle("⚡ I don’t have more traffic — interpret the result anyway", value=False)

# Optional business value input
conversion_value = st.number_input(
    "💰 Optional: Each conversion is worth (in $ or £)",
    min_value=0.0, value=0.0, step=0.1,
    help="Used to estimate expected gains/losses for action guidance."
)

# --- 1. Input Data ---
st.header("1️⃣ Enter Your A/B Test Data")
with st.expander("ℹ️ What are 'Visitors' and 'Conversions'?", expanded=False):
    st.markdown("""
    - **Visitors**: Number of users who saw each version.
    - **Conversions**: Users who completed your goal (e.g. signed up, purchased).
    - This tool compares the **conversion rates** between A and B.

These inputs are critical: more visitors increase certainty; conversions drive the lift estimate.
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
st.markdown("If you have a strong expectation from past experience, adjust these. Otherwise, leave at 1 for a neutral start.")
with st.expander("📝 Why set priors?", expanded=False):
    st.markdown("""
    **Priors** let you incorporate what you already believe before seeing the new data:

    - **Alpha (α)** and **Beta (β)** shape the **Beta distribution** that represents your initial guess on conversion rate.
    - Setting α > β biases the model toward higher rates; α < β biases toward lower rates.
    - A neutral prior (α=1, β=1) treats all rates equally likely between 0 and 1.
    - **Increasing both** α and β by the same amount reflects **confidence** in your prior without shifting the mean: e.g., α=10, β=10 means you're quite sure the rate is around 50% before the test.

Examples:
- α=2, β=8 encodes a belief the conversion is around 20%, but with low confidence.
- α=20, β=80 encodes the same mean (20%) but with high confidence.
    """)
col3, col4 = st.columns(2)
with col3:
    alpha_prior = st.number_input("Prior Alpha (α)", min_value=0.01, value=1.0)
with col4:
    beta_prior = st.number_input("Prior Beta (β)", min_value=0.01, value=1.0)

# --- 3. Confidence Threshold ---
st.header("3️⃣ Confidence Level")
confidence_choice = st.selectbox(
    "How confident do you want to be in your result?", [95, 90, 80], index=0
)
prob_threshold = confidence_choice / 100.0
ci_tail = (1 - prob_threshold) / 2 * 100
ci_low_percentile = ci_tail
ci_high_percentile = 100 - ci_tail

robust_width_target = st.slider(
    f"Required CI width for robustness at {confidence_choice}% confidence",
    min_value=0.005,
    max_value=0.03,
    value={95: 0.01, 90: 0.012, 80: 0.015}.get(confidence_choice),
    step=0.001,
    help="Lower = more strict. A narrow interval means more precise estimates."
)

# --- 4. ROPE Slider ---
st.header("4️⃣ Minimum Meaningful Difference (ROPE)")
practical_effect_display = st.slider(
    "Ignore differences smaller than this (0%–5%)",
    min_value=0.0,
    max_value=5.0,
    value=0.5,
    step=0.1,
    help="ROPE = Region of Practical Equivalence: changes below this are too small to matter."
)
practical_effect = practical_effect_display / 100.0

# --- 5. Test Duration ---
test_days = st.number_input(
    "Number of days the test has been running",
    min_value=1, value=7,
    help="Used to estimate how many more days you'd need for robust results."
)

# --- Bayesian Inference ---
alpha_a = alpha_prior + conversions_a
beta_a = beta_prior + visitors_a - conversions_a
alpha_b = alpha_prior + conversions_b
beta_b = beta_prior + visitors_b - conversions_b

samples = 200_000
posterior_a = np.random.beta(alpha_a, beta_a, samples)
posterior_b = np.random.beta(alpha_b, beta_b, samples)

mean_a = np.mean(posterior_a)
mean_b = np.mean(posterior_b)
expected_absolute_lift = mean_b - mean_a
expected_relative_lift = (expected_absolute_lift / mean_a) * 100

delta = posterior_b - posterior_a
prob_b_better = np.mean(delta > 0)
ci_low, ci_high = np.percentile(delta, [ci_low_percentile, ci_high_percentile])
ci_width = ci_high - ci_low
in_rope = np.mean((delta > -practical_effect) & (delta < practical_effect))
statistically_significant = ci_low > 0 or ci_high < 0
robust = ci_width < robust_width_target and statistically_significant and in_rope < 0.95

# --- Estimate Additional Visitors Needed ---
total_visitors = visitors_a + visitors_b
scale_factor = (ci_width / robust_width_target) ** 2 if ci_width > 0 else 1
suggested_total_visitors = int(total_visitors * scale_factor)
additional_visitors = max(suggested_total_visitors - total_visitors, 0)
avg_daily_visitors = total_visitors / test_days if test_days else 1
estimated_days = int(np.ceil(additional_visitors / avg_daily_visitors)) if avg_daily_visitors else None

# --- Expected Value Calculation ---
expected_gain = None
if conversion_value > 0:
    expected_gain = expected_absolute_lift * conversion_value * visitors_b

# --- Result Summary ---
st.header("📊 Summary of Your A/B Test")

if simple_mode:
    st.markdown(
        f"📈 The expected improvement is about **{expected_relative_lift:.2f}%** (relative) or **{expected_absolute_lift:.4f}** (absolute)."
    )
    if prob_b_better >= prob_threshold:
        st.success(
            f"✅ B is likely better than A, with about **{prob_b_better*100:.1f}% certainty**."
        )
    else:
        st.warning(
            f"⚠️ We can't be confident B is better than A (only {prob_b_better*100:.1f}% certainty)."
        )

    if in_rope > 0.95:
        st.info("ℹ️ The improvement is probably too small to matter.")
    elif in_rope < 0.05:
        st.success("✅ The improvement is likely meaningful.")
    else:
        st.warning("⚠️ It’s unclear if the difference is big enough to act on.")

    if robust:
        st.success("🎯 This result looks reliable and trustworthy.")
    elif no_more_traffic:
        st.warning(
            "⚠️ This result is promising but not yet robust. Proceed with caution, especially for critical decisions."
        )
    else:
        st.warning("🚧 The result isn’t stable yet — more data is needed.")
        if estimated_days is not None:
            st.markdown(
                f"📊 You should collect about **{additional_visitors:,} more visitors** (≈ **{estimated_days} more days**) before deciding."
            )
        else:
            st.markdown(
                f"📊 You should collect about **{additional_visitors:,} more visitors** before deciding."
            )
    if expected_gain is not None:
        st.markdown(
            f"💡 If you deploy Variant B now, expected gain is **{conversion_value:,.2f}/conversion**, "
            f"totaling approx **{expected_gain:,.2f}** over {visitors_b} visitors."
        )
else:
    st.write(f"Expected Lift: {expected_relative_lift:.2f}%")
    st.write(f"Absolute Difference: {expected_absolute_lift:.4f}")
    st.write(f"Probability B > A: {prob_b_better*100:.2f}%")
    st.write(f"{confidence_choice}% Credible Interval Width: {ci_width:.4f}")
    st.caption(
        "A credible interval shows where the true difference likely lies. Smaller width = more certainty."
    )
    st.write(f"ROPE Overlap: {in_rope*100:.1f}%")
    st.write("Statistically Significant:", statistically_significant)
    st.write("Robust:", robust)

if show_robustness_explanation:
    st.markdown("""
    #### 🤔 What does 'Robust' mean?
    A result is considered **robust** only if:
    - ✅ The credible interval is **narrow enough** (threshold set above)
    - ✅ The result is **statistically significant** (interval does not cross 0)
    - ✅ The effect is **not mostly trivial** (low ROPE overlap)

    This ensures your decision is **confident, precise, and meaningful**.
    """)

if show_decision_mode:
    st.subheader("🧠 Decision Recommendation")
    if robust:
        st.success("✅ You can confidently act on this result — it’s robust, meaningful, and stable.")
    elif prob_b_better >= prob_threshold and in_rope < 0.5:
        st.info(
            "🟡 Promising result — consider acting **if business impact is high** or retesting later."
        )
    else:
        st.warning("🚫 Don’t act yet — not enough certainty, precision, or practical impact.")

# --- Posterior Distributions ---
st.header("📈 Posterior Distributions")
mean_rate_a = conversions_a / visitors_a if visitors_a else 0
mean_rate_b = conversions_b / visitors_b if visitors_b else 0
max_rate = max(mean_rate_a, mean_rate_b)
x = np.linspac
