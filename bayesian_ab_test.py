import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

# Page setup
st.set_page_config(page_title="Bayesian A/B Test Calculator", layout="centered")

# Title and description
st.title("🧪 Easy Bayesian A/B Test Calculator")
st.markdown("""
Use **Bayesian analysis** to make clear, data-driven decisions in A/B testing.  
No jargon—just straightforward insights.
""")
st.markdown("---")

# Mode toggles and business value
col1, col2 = st.columns(2)
with col1:
    simple_mode = st.checkbox("Show plain-English explanations", value=True)
    no_more_traffic = st.checkbox("I don’t have more traffic—interpret result anyway", value=True)
with col2:
    show_robustness_explanation = st.checkbox("Explain Robustness Criteria", value=True)
    show_decision_mode = st.checkbox("Show Decision Guidance", value=True)
st.markdown("---")

conversion_value = st.number_input(
    "Optional: Value per conversion (e.g. £10)",
    min_value=0.0, value=0.0, step=0.1,
    help="Enter how much each conversion is worth to estimate monetary impact."
)
st.markdown("---")

# 1. Test Data Inputs
st.header("1. Test Data")
st.markdown("Enter visitors and conversions for Control and Variant.")
with st.expander("Why these matter", expanded=False):
    st.markdown("""
    - **Visitors**: Number of users shown each version.  
    - **Conversions**: Users who completed your goal (e.g., purchase).  
    More data leads to more reliable insights.
""")
col3, col4 = st.columns(2)
with col3:
    visitors_a = st.number_input("Visitors (Control)", min_value=1, value=1000)
    conversions_a = st.number_input("Conversions (Control)", min_value=0, value=50)
with col4:
    visitors_b = st.number_input("Visitors (Variant)", min_value=1, value=1000)
    conversions_b = st.number_input("Conversions (Variant)", min_value=0, value=70)
st.markdown("---")

# 2. Prior Beliefs
st.header("2. Priors (Optional)")
st.markdown("Adjust prior alpha/beta if you have historical knowledge; otherwise leave at 1.")
with st.expander("What are priors?", expanded=False):
    st.markdown("""
    Priors shape your initial expectation:
    - α and β form a Beta distribution for conversion rate.
    - α=1, β=1 is neutral.  
    - α>β tilts higher, α<β tilts lower.  
    - Increasing both equally adds confidence around the mean.
""")
col5, col6 = st.columns(2)
with col5:
    alpha_prior = st.number_input("Prior Alpha (α)", min_value=0.01, value=1.0)
with col6:
    beta_prior = st.number_input("Prior Beta (β)", min_value=0.01, value=1.0)
st.markdown("---")

# 3. Confidence & Robustness
st.header("3. Confidence & Robustness")
confidence_choice = st.selectbox("Select confidence level (%)", [95, 90, 80], index=0)
prob_threshold = confidence_choice / 100.0
ci_tail = (1 - prob_threshold) / 2 * 100
ci_low_pct, ci_high_pct = ci_tail, 100 - ci_tail
robust_width_target = st.slider(
    f"Max CI width for robust result at {confidence_choice}%", 0.005, 0.03,
    value={95:0.01,90:0.012,80:0.015}[confidence_choice], step=0.001,
    help="A narrow CI means a more precise estimate."
)
st.markdown("---")

# 4. Practical Impact (ROPE)
st.header("4. Practical Impact (ROPE)")
practical_display = st.slider(
    "Ignore changes smaller than (%)", 0.0, 5.0, 0.5, 0.1,
    help="ROPE = range where differences are too small to matter."
)
practical_effect = practical_display / 100.0
st.markdown("---")

# 5. Test Duration
st.header("5. Test Duration")
test_days = st.number_input("Days test has been running", min_value=1, value=7,
    help="Estimate days needed if more precision is required."
)
st.markdown("---")

# ——— Perform Bayesian Calculations ———
alpha_a = alpha_prior + conversions_a
beta_a  = beta_prior + visitors_a - conversions_a
alpha_b = alpha_prior + conversions_b
beta_b  = beta_prior + visitors_b - conversions_b

samples = 200_000
post_a = np.random.beta(alpha_a, beta_a, samples)
post_b = np.random.beta(alpha_b, beta_b, samples)

mean_a, mean_b = np.mean(post_a), np.mean(post_b)
delta = post_b - post_a

decision_prob = np.mean(delta > 0)
abs_lift      = mean_b - mean_a
rel_lift      = (abs_lift / mean_a) * 100
ci_low, ci_high = np.percentile(delta, [ci_low_pct, ci_high_pct])
ci_width      = ci_high - ci_low
rope_overlap  = np.mean((delta > -practical_effect) & (delta < practical_effect))
statsig       = (ci_low > 0) or (ci_high < 0)
robust        = statsig and (ci_width < robust_width_target) and (rope_overlap < 0.95)

# Estimate extra visitors/time
total_vis    = visitors_a + visitors_b
scale_factor = (ci_width / robust_width_target) ** 2 if ci_width > 0 else 1
needed_total = int(total_vis * scale_factor)
extra_vis    = max(needed_total - total_vis, 0)
avg_vis_day  = total_vis / test_days if test_days else 1
days_needed  = int(np.ceil(extra_vis / avg_vis_day)) if avg_vis_day else None

# Financial projections
monthly_gain = annual_gain = None
if conversion_value > 0:
    visitors_per_month = avg_vis_day * 30
    monthly_gain = abs_lift * conversion_value * visitors_per_month
    annual_gain = monthly_gain * 12

# ——— Results Summary & Visualization ———
st.header("📊 Results Summary")
if simple_mode:
    st.markdown(f"**Expected lift:** {rel_lift:.2f}%")
    st.markdown(f"**Chance Variant > Control:** {decision_prob*100:.1f}%")
    if conversion_value > 0:
        st.markdown(f"💰 **Expected monthly gain:** £{monthly_gain:,.2f}")
        st.caption("Projected monthly gain based on test traffic.")
        st.markdown(f"📈 **Expected annual gain:** £{annual_gain:,.2f}")
        st.caption("Projected annual gain based on test traffic.")
    # Decision logic based on probability
    if decision_prob >= prob_threshold:
        st.success("✅ Variant likely outperforms Control.")
    elif (1 - decision_prob) >= prob_threshold:
        st.error("⛔ Control likely outperforms Variant — do NOT implement Variant.")
        st.caption("High confidence that the control is better. Revert traffic to Control or test new ideas.")
    else:
        st.warning("⚠️ Insufficient confidence that Variant outperforms Control.")
    # Robustness check
    if robust:
        st.success("🔒 Result is robust: precise, significant, meaningful.")
    else:
        if no_more_traffic:
            if decision_prob >= 0.5:
                st.warning("⚠️ Promising but not robust—proceed with caution.")
                st.caption("Consider limiting exposure, monitoring metrics closely, and planning follow-up tests to verify performance before full rollout.")
            else:
                st.warning("⚠️ Variant underperforms Control—consider focusing on Control or gathering more data.")
                st.caption("Based on current data, the variant is less effective than the control. Consider reverting or testing new variants.")
        else:
            st.warning("🚧 Not yet robust—consider collecting more data.")
            if days_needed:
                st.markdown(f"🔍 Collect ~{extra_vis:,} more visitors (~{days_needed} days) for robust results.")
        st.warning("⚠️ Insufficient confidence that Variant outperforms Control.")
    # Robustness check:
        st.success("🔒 Result is robust: precise, significant, meaningful.")
    else:
        if no_more_traffic:
            if decision_prob >= 0.5:
                st.warning("⚠️ Promising but not robust—proceed with caution.")
                st.caption("Consider limiting exposure, monitoring metrics closely, and planning follow-up tests to verify performance before full rollout.")
            else:
                st.warning("⚠️ Variant underperforms Control—consider focusing on Control or gathering more data.")
                st.caption("Based on current data, the variant is less effective than the control. Consider reverting or testing new variants.")
        else:
            st.warning("🚧 Not yet robust—consider collecting more data.")
            if days_needed:
                st.markdown(f"🔍 Collect ~{extra_vis:,} more visitors (~{days_needed} days) for robust results.")
else:
    st.subheader("Detailed Metrics")
    st.write(f"- Expected lift: {rel_lift:.2f}%")
    st.write(f"- Absolute lift: {abs_lift:.4f}")
    st.write(f"- P(Variant>Control): {decision_prob*100:.2f}%")
    st.write(f"- {confidence_choice}% CI: [{ci_low:.4f}, {ci_high:.4f}] (width {ci_width:.4f})")
    st.write(f"- ROPE overlap: {rope_overlap*100:.1f}%")
    st.write(f"- Stat sig: {statsig}")
    st.write(f"- Robust: {robust}")

if show_robustness_explanation:
    st.info("**Robust** = statistically significant, precise (narrow CI), and practically meaningful.")
if show_decision_mode:
    st.subheader("🧠 Decision Guidance")
    if robust:
        st.success("Implement Variant — reliable result.")
    elif decision_prob >= prob_threshold and rope_overlap < 0.5:
        st.info("Consider Variant if benefits outweigh risks.")
    else:
        st.warning("Hold off — not enough evidence to implement Variant.")

st.markdown("---")
# Posterior distributions
x = np.linspace(0, max(mean_a, mean_b)*1.5, 1000)
fig1, ax1 = plt.subplots(figsize=(6,3))
ax1.plot(x, beta.pdf(x, alpha_a, beta_a), label='Control')
ax1.plot(x, beta.pdf(x, alpha_b, beta_b), label='Variant')
ax1.set_xlabel('Conversion rate')
ax1.set_ylabel('Density')
ax1.legend()
st.pyplot(fig1)

# Difference histogram
fig2, ax2 = plt.subplots(figsize=(6,3))
ax2.hist(delta, bins=50, color='gray', alpha=0.7)
ax2.axvline(0, color='red', linestyle='--', label='No difference')
ax2.set_xlabel('Difference (Variant − Control)')
ax2.set_ylabel('Frequency')
ax2.legend()
st.pyplot(fig2)
