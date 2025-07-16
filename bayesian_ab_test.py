import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

# Page setup
st.set_page_config(page_title="Bayesian CRO Test Calculator", layout="centered")

# Title and description
st.title("🧪 Bayesian CRO Test Calculator")
st.markdown(
    "Use Bayesian analysis to make clear, data-driven A/B test decisions with practical advice."
)
st.markdown("---")

# INPUTS
st.header("🔢 Inputs")
st.markdown(
    """
    Enter your A/B test details below. Plain English hints:  
    • **Visitors**: Users exposed to each version (more = more precise).  
    • **Conversions**: Goal completions per version (drives lift estimate).  
    • **Priors**: (Optional) α/β to encode existing beliefs (1,1 = neutral).  
    • **Confidence level**: Desired certainty (95%, 90%, 80%).  
    • **Max CI width**: Maximum uncertainty (%) for a result to be robust.  
    • **ROPE**: Minimum meaningful difference (%) to ignore tiny changes.  
    • **Days run**: How many days the test has been live (for time estimates).  
    • **Value per conversion**: (Optional) Monetary value per conversion.  
    """
)
col1, col2 = st.columns(2)
with col1:
    visitors_a = st.number_input("Visitors (Control)", min_value=1, value=1000,
        help="Number of users who saw the Control version.")
    conversions_a = st.number_input("Conversions (Control)", min_value=0, value=50,
        help="Successful conversions in the Control group.")
with col2:
    visitors_b = st.number_input("Visitors (Variant)", min_value=1, value=1000,
        help="Number of users who saw the Variant version.")
    conversions_b = st.number_input("Conversions (Variant)", min_value=0, value=70,
        help="Successful conversions in the Variant group.")
alpha_prior = st.number_input("Prior Alpha (α)", min_value=0.01, value=1.0,
    help="Shape parameter α of Beta prior.")
beta_prior = st.number_input("Prior Beta (β)", min_value=0.01, value=1.0,
    help="Shape parameter β of Beta prior.")
confidence = st.selectbox("Confidence level (%)", [95, 90, 80], index=0)
ci_level = confidence / 100.0
ci_tail = (1 - ci_level) / 2 * 100
ci_low_pct, ci_high_pct = ci_tail, 100 - ci_tail
max_ci_width_pct = st.slider(
    "Max CI width (%) for robustness", 0.5, 3.0, 1.0, step=0.1,
    help="Maximum acceptable credible interval width in percentage points."
)
rope_pct = st.slider("ROPE (%) - ignore differences smaller than", 0.0, 5.0, 0.5, step=0.1,
    help="Differences within ±this range are practically negligible."
)
days_run = st.number_input("Days test has run", min_value=1, value=7,
    help="Number of days the test has been live."
)
conversion_value = st.number_input("Value per conversion (£)", min_value=0.0, value=0.0, step=0.1,
    help="Monetary value per conversion for revenue estimates."
)
st.markdown("---")

# BAYESIAN CALCULATIONS
alpha_a = alpha_prior + conversions_a
beta_a = beta_prior + visitors_a - conversions_a
alpha_b = alpha_prior + conversions_b
beta_b = beta_prior + visitors_b - conversions_b

samples = 200_000
post_a = np.random.beta(alpha_a, beta_a, samples)
post_b = np.random.beta(alpha_b, beta_b, samples)
mean_a = np.mean(post_a)
mean_b = np.mean(post_b)
delta = post_b - post_a

prob_b = np.mean(delta > 0)
abs_lift = mean_b - mean_a
rel_lift = abs_lift / mean_a * 100
ci_low, ci_high = np.percentile(delta, [ci_low_pct, ci_high_pct])
ci_width_pct = (ci_high - ci_low) * 100
rope_overlap = np.mean((delta > -rope_pct/100) & (delta < rope_pct/100))
is_significant = (ci_low > 0) or (ci_high < 0)
robust = is_significant and (ci_width_pct <= max_ci_width_pct) and (rope_overlap < 0.95)

# Data needs for precision
total_vis = visitors_a + visitors_b
scale = (ci_width_pct / max_ci_width_pct)**2 if max_ci_width_pct>0 else 1
needed_total = int(total_vis * scale)
extra_vis = max(needed_total - total_vis, 0)
daily_vis = total_vis / days_run
days_more = int(np.ceil(extra_vis / daily_vis)) if daily_vis>0 else None

# RESULTS SUMMARY
st.header("📊 Key Results")
st.markdown(f"**P(Variant > Control):** {prob_b*100:.1f}%")
st.markdown(f"**Expected relative uplift:** {rel_lift:.2f}%")
st.markdown(f"**Expected absolute uplift:** {abs_lift*100:.2f} pp")
st.markdown(
    f"**{confidence}% CI for lift:** [{ci_low*100:.2f}%, {ci_high*100:.2f}%] "
    f"(width {ci_width_pct:.2f}%)"
)
st.markdown("---")

# DECISION & INTERPRETATION
if prob_b >= ci_level:
    st.success("✅ Variant likely outperforms Control.")
elif (1-prob_b) >= ci_level:
    st.error("⛔ Control likely outperforms Variant — do NOT implement Variant.")
else:
    st.warning("⚠️ No clear winner yet — consider next steps.")

st.subheader("🔍 What does this mean?")
if robust:
    st.markdown("Your result is robust: confident in both direction and magnitude.")
elif prob_b >= ci_level:
    if ci_low < 0:
        st.markdown(
            "Variant seems better, but the CI includes negative values, so a loss is still possible."
        )
    else:
        st.markdown("Variant seems better, and even the lower CI bound is positive — likely beneficial.")
else:
    st.markdown("No clear benefit of Variant — it may underperform Control.")

# ACTIONABLE ADVICE
st.subheader("🛠️ What to do next?")
# Fast ramp advice
if robust:
    st.info("🚀 Robust results — roll out Variant to 100% of traffic now.")
else:
    holdback = min(max(int((days_more/(days_run+days_more))*100), 5), 10) if days_more else 10
    monitor = min([d for d in [days_more, days_run, 3] if d is not None])
    st.info(
        f"⚙️ Fast ramp: push Variant to {100-holdback}% of traffic, hold back {holdback}% for Control, "
        f"monitoring for {monitor} days."
    )
    if days_more:
        st.info(f"🔍 To fully robust: collect ~{extra_vis} more visitors (~{days_more} days).")

st.markdown("---")

# VISUALIZATIONS
# Posterior distributions
st.subheader("📈 Posterior Distributions of Conversion Rates")
st.markdown(
    """
    Shows plausible conversion rates (CVR) for each version:
    - Blue: Control
    - Green: Variant
    Peaks = most likely rates.
    """
)
x = np.linspace(0, max(mean_a, mean_b)*1.5, 1000)
fig1, ax1 = plt.subplots(figsize=(7,4))
ax1.fill_between(x, beta.pdf(x, alpha_a, beta_a), color='skyblue', alpha=0.5)
ax1.plot(x, beta.pdf(x, alpha_a, beta_a), color='blue', label=f"Control mean {mean_a*100:.2f}%")
ax1.fill_between(x, beta.pdf(x, alpha_b, beta_b), color='lightgreen', alpha=0.5)
ax1.plot(x, beta.pdf(x, alpha_b, beta_b), color='green', label=f"Variant mean {mean_b*100:.2f}%")
ax1.set_xlabel('Conversion rate (%)')
ax1.set_ylabel('Density')
ax1.set_title('Posterior Distributions of CVR')
ax1.legend(loc='upper right')
ax1.grid(alpha=0.3)
fig1.tight_layout()
st.pyplot(fig1)

# Lift distribution
st.subheader("📉 Posterior Distribution of Lift (Variant − Control)")
st.markdown(
    """
    Distribution of possible differences in CVR:
    - Right of zero = Variant > Control
    - Left of zero = Control > Variant
    """
)
fig2, ax2 = plt.subplots(figsize=(7,4))
counts, bins, patches = ax2.hist(delta*100, bins=50, edgecolor='white')
for patch, edge in zip(patches, bins[:-1]):
    patch.set_facecolor('lightgreen' if edge>0 else 'salmon')
ax2.axvline(0, color='black', linestyle='--')
ax2.set_xlabel('Lift (%)')
ax2.set_ylabel('Frequency')
ax2.set_title('Lift Distribution')
ax2.grid(alpha=0.3)
st.pyplot(fig2)

# CI width vs sample size
if True:
    st.subheader("📊 CI Width vs. Total Sample Size")
    st.markdown(
        """
        As total visitors ↑, credible interval width ↓.
        X-axis = total visitors; Y-axis = CI width (%).
        """
    )
    sizes = np.linspace(total_vis, total_vis*3, 50, dtype=int)
    ci_ws = []
    for n in sizes:
        va = n/2
        vb = n/2
        pa = np.random.beta(alpha_prior+conversions_a*(va/visitors_a), beta_prior+va-conversions_a*(va/visitors_a), 10000)
        pb = np.random.beta(alpha_prior+conversions_b*(vb/visitors_b), beta_prior+vb-conversions_b*(vb/visitors_b), 10000)
        d = pb - pa
        lw, hw = np.percentile(d, [ci_low_pct, ci_high_pct])
        ci_ws.append((hw-lw)*100)
    fig3, ax3 = plt.subplots(figsize=(7,4))
    ax3.plot(sizes, ci_ws, marker='o')
    ax3.set_xlabel('Total visitors')
    ax3.set_ylabel('CI width (%)')
    ax3.set_title('CI Width vs. Sample Size')
    ax3.grid(alpha=0.3)
    fig3.tight_layout()
    st.pyplot(fig3)
