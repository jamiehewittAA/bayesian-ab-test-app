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
st.markdown("---")  # separator

# Mode toggles
col_toggle1, col_toggle2 = st.columns(2)
with col_toggle1:
    simple_mode = st.checkbox("Show plain-English explanations", value=True)
    no_more_traffic = st.checkbox("I don’t have more traffic—interpret result anyway", value=False)
with col_toggle2:
    show_robustness_explanation = st.checkbox("Explain Robustness Criteria", value=False)
    show_decision_mode = st.checkbox("Show Decision Guidance", value=True)
st.markdown("---")

# Optional business value input
conversion_value = st.number_input(
    "Optional: Value per conversion (e.g. £10)",
    min_value=0.0, value=0.0, step=0.1,
    help="Enter how much each conversion is worth to estimate monetary impact."
)
st.write("")  # spacing

# 1. Input data
st.header("1. Test Data")
st.markdown("Enter the number of visitors and conversions for both versions.")
with st.expander("Why these numbers matter", expanded=False):
    st.markdown("""
    - **Visitors**: Total users exposed to each variant.
    - **Conversions**: Users who completed your goal (e.g., purchase).
    More data = more confidence; conversions define the lift.
""")
col1, col2 = st.columns(2)
with col1:
    visitors_a = st.number_input("Visitors to A (Control)", min_value=1, value=1000)
    conversions_a = st.number_input("Conversions from A", min_value=0, value=50)
with col2:
    visitors_b = st.number_input("Visitors to B (Variant)", min_value=1, value=1000)
    conversions_b = st.number_input("Conversions from B", min_value=0, value=70)
st.markdown("---")

# 2. Prior beliefs
st.header("2. Priors (Optional)")
st.markdown("Adjust if you have prior knowledge; otherwise leave at 1 for neutral.")
with st.expander("What are priors?", expanded=False):
    st.markdown("""
    Priors incorporate past experience:
    - **α (alpha)** & **β (beta)** shape the initial conversion guess.
    - **α=1, β=1** = no prior preference.
    - **Increase both equally** (e.g., α=10, β=10) to express confidence around 50%.
    - **α>β** favors higher rates; **α<β** favors lower.
""")
col3, col4 = st.columns(2)
with col3:
    alpha_prior = st.number_input("Prior Alpha (α)", min_value=0.01, value=1.0)
with col4:
    beta_prior = st.number_input("Prior Beta (β)", min_value=0.01, value=1.0)
st.markdown("---")

# 3. Confidence & robustness
st.header("3. Confidence & Robustness")
st.markdown("Choose how sure you want to be and how precise the result must be.")
confidence_choice = st.selectbox("Confidence level (%)", [95, 90, 80], index=0)
prob_threshold = confidence_choice / 100.0
ci_tail = (1 - prob_threshold) / 2 * 100
ci_low_pct, ci_high_pct = ci_tail, 100 - ci_tail
robust_width_target = st.slider(
    f"Max width of credible interval (CI) for robust result:",
    min_value=0.005, max_value=0.03,
    value={95:0.01,90:0.012,80:0.015}[confidence_choice], step=0.001,
    help="A narrower CI means more trust in the estimated difference."
)
st.markdown("---")

# 4. Practical significance (ROPE)
st.header("4. Practical Impact (ROPE)")
st.markdown("Define what change size you consider too small to matter.")
practical_effect_display = st.slider(
    "Ignore changes smaller than (%):",
    min_value=0.0, max_value=5.0, value=0.5, step=0.1,
    help="ROPE: Range where difference is practically negligible."
)
practical_effect = practical_effect_display / 100.0
st.markdown("---")

# 5. Test duration
st.header("5. Test Duration")
st.markdown("Tell us how many days the test has been running.")
test_days = st.number_input(
    "Days test has been running", min_value=1, value=7,
    help="Used to estimate days needed if we require more precision."
)
st.markdown("---")

# Bayesian calculations
alpha_a = alpha_prior + conversions_a
beta_a = beta_prior + visitors_a - conversions_a
alpha_b = alpha_prior + conversions_b
beta_b = beta_prior + visitors_b - conversions_b

samples = 200_000
post_a = np.random.beta(alpha_a, beta_a, samples)
post_b = np.random.beta(alpha_b, beta_b, samples)
mean_a, mean_b = np.mean(post_a), np.mean(post_b)
delta = post_b - post_a

# Metrics
decision_prob = np.mean(delta > 0)
abs_lift = mean_b - mean_a
rel_lift = (abs_lift / mean_a) * 100
ci_low, ci_high = np.percentile(delta, [ci_low_pct, ci_high_pct])
ci_width = ci_high - ci_low
rope_overlap = np.mean((delta > -practical_effect) & (delta < practical_effect))
statsig = (ci_low > 0) or (ci_high < 0)
robust = statsig and (ci_width < robust_width_target) and (rope_overlap < 0.95)

# Estimate extra time needed
total_vis = visitors_a + visitors_b
scale = (ci_width / robust_width_target) ** 2 if ci_width > 0 else 1
needed_vis = int(total_vis * scale)
extra_vis = max(needed_vis - total_vis, 0)
avg_vis_day = total_vis / test_days if test_days else 1
days_needed = int(np.ceil(extra_vis / avg_vis_day)) if avg_vis_day > 0 else None

# Expected monetary impact
exp_gain = None
monthly_gain = None
annual_gain = None
if conversion_value > 0:
    exp_gain = abs_lift * conversion_value * visitors_b
    # Calculate per month and annual gains
    avg_daily_visitors = (visitors_a + visitors_b) / test_days if test_days else 0
    visitors_per_month = avg_daily_visitors * 30
    monthly_gain = abs_lift * conversion_value * visitors_per_month
    annual_gain = monthly_gain * 12

# Summary outputs
st.header("📊 Results Summary")

if simple_mode:
    st.markdown(f"**Expected lift:** {rel_lift:.2f}% (or {abs_lift:.4f} points)")
    st.markdown(f"**Probability B > A:** {decision_prob*100:.1f}%")
    if exp_gain is not None:
        st.markdown(f"💡 **Expected gain for test sample:** £{exp_gain:,.2f}")
        st.caption("Based on the current test traffic.")
        st.markdown(f"💰 **Expected monthly gain:** £{monthly_gain:,.2f}")
        st.caption("Projected monthly gain assuming similar daily traffic.")
        st.markdown(f"📈 **Expected annual gain:** £{annual_gain:,.2f}")
        st.caption("Projected annual gain assuming similar traffic and performance.")
    if decision_prob >= prob_threshold:
        st.success("🎉 B is likely better than A!")
    else:
        st.warning("⚠️ Not enough confidence that B is better.")
    if robust:
        st.success("🔒 Result is robust: precise, significant, and meaningful.")
    else:
        if no_more_traffic:
            st.warning("⚠️ Promising but not robust—use caution if acting now.")
        else:
            st.warning("🚧 Result not yet robust—consider collecting more data.")
            if days_needed:
                st.markdown(f"🔍 Collect ~{extra_vis:,} more visitors (~{days_needed} days) for robust results.")

else:
    st.subheader("Detailed Statistics")
    st.write(f"- Expected lift: {rel_lift:.2f}%")
    st.write(f"- Absolute lift: {abs_lift:.4f}")
    st.write(f"- Probability B > A: {decision_prob*100:.2f}%")
    st.write(f"- {confidence_choice}% credible interval: [{ci_low:.4f}, {ci_high:.4f}] (width {ci_width:.4f})")
    st.caption("A narrower interval indicates more precise estimates.")
    st.write(f"- ROPE overlap: {rope_overlap*100:.1f}%")
    st.write(f"- Statistically significant: {statsig}")
    st.write(f"- Robust: {robust}")

# Additional Explanations
if show_robustness_explanation:
    st.info("**Robust** means the result is statistically significant, the credible interval is narrow enough, and the effect size is practically meaningful.")

if show_decision_mode:
    st.subheader("🧠 Decision Guidance")
    if robust:
        st.success("✅ Recommendation: Implement Variant B — results are reliable and meaningful.")
    elif decision_prob >= prob_threshold and rope_overlap < 0.5:
        st.info("🟡 Consider implementing Variant B if the potential gains justify the risk.")
    else:
        st.warning("🚫 Recommendation: Do not implement Variant B yet — evidence is insufficient.")

# Posterior distributions
x = np.linspace(0, max(mean_a, mean_b) * 1.5, 1000)
fig1, ax1 = plt.subplots(figsize=(6, 3))
ax1.plot(x, beta.pdf(x, alpha_a, beta_a), label='A (Control)')
ax1.plot(x, beta.pdf(x, alpha_b, beta_b), label='B (Variant)')
ax1.set_xlabel('Conversion rate')
ax1.set_ylabel('Density')
ax1.legend()
st.pyplot(fig1)

# Difference histogram
fig2, ax2 = plt.subplots(figsize=(6, 3))
ax2.hist(delta, bins=50, color='gray', alpha=0.7)
ax2.axvline(0, color='red', linestyle='--', label='No difference')
ax2.set_xlabel('Difference B − A')
ax2.set_ylabel('Frequency')
ax2.legend()
st.pyplot(fig2)
