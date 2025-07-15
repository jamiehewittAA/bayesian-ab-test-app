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

# Title and description
st.title("🧪 CRO Bayesian A/B Test Calculator")
st.markdown("""
Use **Bayesian analysis** to make clear, data-driven decisions in A/B testing.  
No jargon—just straightforward insights.
""")
st.write("")  # spacing

# Mode toggles
simple_mode = st.checkbox("Show plain-English explanations", value=True)
show_robustness_explanation = st.checkbox("Explain Robustness Criteria", value=False)
show_decision_mode = st.checkbox("Show Decision Guidance", value=True)
no_more_traffic = st.checkbox("I don’t have more traffic—interpret result anyway", value=False)
st.write("")  # spacing

# Optional business value input
conversion_value = st.number_input(
    "Optional: Value per conversion (e.g. £10)",
    min_value=0.0, value=0.0, step=0.1,
    help="Enter how much each conversion is worth to estimate monetary impact."
)
st.write("")  # spacing

# 1. Input data
st.header("1. Enter Results from Your A/B Test")
with st.expander("Why these numbers matter", expanded=False):
    st.markdown("""
    - **Visitors** = how many people saw each version (A/B).  
    - **Conversions** = how many completed your goal (e.g., purchase, sign-up).  
    More visitors = more reliable results; conversions determine lift.
""")
col1, col2 = st.columns(2)
with col1:
    visitors_a = st.number_input("Visitors to A (Control)", min_value=1, value=1000)
    conversions_a = st.number_input("Conversions from A", min_value=0, value=50)
with col2:
    visitors_b = st.number_input("Visitors to B (Variant)", min_value=1, value=1000)
    conversions_b = st.number_input("Conversions from B", min_value=0, value=70)
st.write("")  # spacing

# 2. Prior beliefs
st.header("2. (Optional) Prior Beliefs")
st.markdown("Set your starting assumptions based on past experience or leave at 1 for neutral.")
with st.expander("What are priors?", expanded=False):
    st.markdown("""
    Priors let you include previous knowledge:
    - **α (alpha)** and **β (beta)** shape your initial guess of conversion rate.  
    - **α=1, β=1** = no prior preference.  
    - **Increase both equally** (e.g. α=10, β=10) to express confidence around 50%.  
    - **α > β** favors higher rates; **α < β** favors lower.
""")
col3, col4 = st.columns(2)
with col3:
    alpha_prior = st.number_input("Prior Alpha (α)", min_value=0.01, value=1.0)
with col4:
    beta_prior = st.number_input("Prior Beta (β)", min_value=0.01, value=1.0)
st.write("")  # spacing

# 3. Confidence & robustness
st.header("3. Confidence & Robustness")
confidence_choice = st.selectbox(
    "Select desired confidence level:", [95, 90, 80], index=0
)
prob_threshold = confidence_choice / 100.0
ci_tail = (1 - prob_threshold) / 2 * 100
ci_low_pct, ci_high_pct = ci_tail, 100 - ci_tail
robust_width_target = st.slider(
    f"Max credible interval width for robustness (at {confidence_choice}%):",
    min_value=0.005, max_value=0.03,
    value={95:0.01, 90:0.012, 80:0.015}[confidence_choice], step=0.001,
    help="A narrower interval means more precise estimates."
)
st.write("")  # spacing

# 4. Practical significance (ROPE)
st.header("4. Practical Significance (ROPE)")
practical_effect_display = st.slider(
    "Ignore changes smaller than (0%–5%):",
    min_value=0.0, max_value=5.0, value=0.5, step=0.1,
    help="ROPE = range where differences are too small to matter."
)
practical_effect = practical_effect_display / 100.0
st.write("")  # spacing

# 5. Test duration
st.header("5. Test Duration")
test_days = st.number_input(
    "Days test has been running:", min_value=1, value=7,
    help="Used to estimate how many more days you need for reliable results."
)
st.write("")  # spacing

# Bayesian calculations
alpha_a = alpha_prior + conversions_a
beta_a = beta_prior + visitors_a - conversions_a
alpha_b = alpha_prior + conversions_b
beta_b = beta_prior + visitors_b - conversions_b
samples = 200000
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
st.write("")  # spacing

# Estimate extra time needed
total_vis = visitors_a + visitors_b
scale = (ci_width / robust_width_target)**2 if ci_width>0 else 1
needed_vis = int(total_vis * scale)
extra_vis = max(needed_vis - total_vis, 0)
avg_vis_day = total_vis / test_days
days_needed = int(np.ceil(extra_vis / avg_vis_day)) if avg_vis_day>0 else None
st.write("")  # spacing

# Expected monetary impact
exp_gain = None
if conversion_value > 0:
    exp_gain = abs_lift * conversion_value * visitors_b

# Summary outputs
st.header("📊 Results Summary")
if simple_mode:
    st.markdown(f"**Expected lift:** {rel_lift:.2f}% (or {abs_lift:.4f} points)")
    st.write(f"**Chance B > A:** {decision_prob*100:.1f}%")
    if decision_prob >= prob_threshold:
        st.success("✅ B is likely better than A.")
    else:
        st.warning("⚠️ Not enough confidence B is better.")
    if exp_gain:
        st.info(f"💡 Expected gain: {exp_gain:,.2f}")
    if robust:
        st.success("🎯 Result is robust: precise, significant, and meaningful.")
    else:
        if no_more_traffic:
            st.warning("⚠️ Promising but not robust—proceed with caution.")
        else:
            st.warning("🚧 Result not yet robust—consider more data.")
            if days_needed:
                st.write(f"Collect ~{extra_vis:,} more visitors (~{days_needed} days). ")
else:
    st.write(f"Expected lift: {rel_lift:.2f}%")
    st.write(f"Absolute lift: {abs_lift:.4f}")
    st.write(f"Credible interval: [{ci_low:.4f}, {ci_high:.4f}] (width={ci_width:.4f})")
    st.caption("Smaller width = more certainty in the estimate.")
    st.write(f"ROPE overlap: {rope_overlap*100:.1f}%")
    st.write(f"Statistically significant: {statsig}")
    st.write(f"Robust: {robust}")
st.write("")  # spacing

if show_robustness_explanation:
    st.info("**Robust =** credible interval excludes 0, is narrow enough, and effect is meaningful.")

if show_decision_mode:
    st.subheader("🧠 Decision Guidance")
    if robust:
        st.success("Action recommended: result is reliable.")
    elif decision_prob>=prob_threshold and rope_overlap<0.5:
        st.info("Consider action if benefits outweigh risks.")
    else:
        st.warning("Hold off: not enough evidence for safe decision.")
st.write("")  # spacing

# Visualizations
st.header("📈 Visualizations")
# Posterior distributions
x = np.linspace(0, max(mean_a,mean_b)*1.5, 1000)
fig1, ax1 = plt.subplots(figsize=(6,3))
ax1.plot(x, beta.pdf(x, alpha_a, beta_a), label='A (Control)')
ax1.plot(x, beta.pdf(x, alpha_b, beta_b), label='B (Variant)')
ax1.set_xlabel('Conversion rate')
ax1.set_ylabel('Density')
ax1.legend()
st.pyplot(fig1)

# Difference histogram
fig2, ax2 = plt.subplots(figsize=(6,3))
ax2.hist(delta, bins=50, color='gray', alpha=0.7)
ax2.axvline(0, color='red', linestyle='--', label='No difference')
ax2.set_xlabel('Difference B − A')
ax2.set_ylabel('Frequency')
ax2.legend()
st.pyplot(fig2)
