import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

# Page setup
st.set_page_config(page_title="Bayesian A/B Test Calculator", layout="centered")

# Title
st.title("🧪 Bayesian CRO Test Calculator")
st.markdown("Use Bayesian analysis to understand A/B test results and get clear, actionable insights.")
st.markdown('---')

# Inputs
st.header("🔢 Inputs")
st.markdown("""
Enter your A/B test data and settings:
- **Visitors**: How many users saw each version (Control/Variant).
- **Conversions**: How many completed your goal (e.g., sign-up, purchase).
- **Priors**: (Optional) Encode any existing belief about conversion rates.
- **Confidence level**: How sure you want to be (95%, 90%, or 80%).
- **CI width**: Maximum acceptable uncertainty for a robust result.
- **ROPE**: The minimal difference you consider meaningful.
- **Test days**: How many days the test has run so far.
- **Value per conversion**: (Optional) For revenue impact estimates.
""" )
col1, col2 = st.columns(2)
with col1:
    visitors_a = st.number_input(
        "Visitors (Control)", min_value=1, value=1000,
        help="Number of users who saw the Control version. More visitors → more precise results."
    )
    conversions_a = st.number_input(
        "Conversions (Control)", min_value=0, value=50,
        help="Number of conversions from Control. Drives uplift estimates."
    )
with col2:
    visitors_b = st.number_input(
        "Visitors (Variant)", min_value=1, value=1000,
        help="Number of users who saw the Variant."
    )
    conversions_b = st.number_input(
        "Conversions (Variant)", min_value=0, value=70,
        help="Number of conversions from Variant."
    )
st.markdown('---')

# Bayesian calculations
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
ci_low, ci_high = np.percentile(delta, [ci_lower, ci_upper])
ci_width = (ci_high - ci_low) * 100
rope_overlap = np.mean((delta > -rope/100) & (delta < rope/100))

# Robustness
is_significant = ci_low > 0 or ci_high < 0
is_robust = is_significant and (ci_width <= max_ci_width) and (rope_overlap < 0.95)

# Data needs
total_vis = visitors_a + visitors_b
factor = (ci_width/100) / (max_ci_width/100)
if factor > 0:
    needed = int(total_vis * factor**2)
else:
    needed = total_vis
extra = max(needed - total_vis, 0)
daily = total_vis / test_days
days_more = int(np.ceil(extra / daily)) if daily>0 else None

# Results
st.header("📊 Key Results")
st.markdown(f"**P(Variant > Control):** {prob_b*100:.1f}%")
st.markdown(f"**Expected relative uplift:** {rel_lift:.2f}%")
st.markdown(f"**Expected absolute uplift:** {abs_lift*100:.2f} pp")
st.markdown(f"**{confidence}% CI for lift:** [{ci_low*100:.2f}%, {ci_high*100:.2f}%] (width {ci_width:.2f}% )")
st.markdown('---')

# Decision & Interpretation
if prob_b >= CI:
    st.success("✅ Variant likely outperforms Control.")
elif (1-prob_b) >= CI:
    st.error("⛔ Control likely outperforms Variant. Do not implement Variant.")
else:
    st.warning("⚠️ No clear winner. Proceed with caution.")

st.subheader("🔍 What does this mean?")
if is_robust:
    st.markdown("Your result is robust: both direction and magnitude are reliable.")
elif prob_b >= CI:
    st.markdown("Variant seems better, but range of effect is wide: true uplift uncertain.")
else:
    st.markdown("No strong evidence Variant is better. Consider sticking with Control or collecting more data.")

st.subheader("🛠️ What to do next?")
if is_robust:
    st.info("🚀 Deploy Variant to 100% of traffic.")
else:
    if days_more:
        st.info(f"⚙️ Roll out Variant to 100% of traffic but hold back ~{int((days_more/(test_days+days_more))*100)}% for Control over the next {days_more} days.")
        st.info(f"🔍 Or collect ~{extra} more visitors (~{days_more} days) to reach desired precision.")
    else:
        st.info("🔍 Consider collecting more data to improve precision.")
st.markdown('---')

# Visualizations
# Posterior distributions
st.subheader("📈 Posterior Distributions")
x = np.linspace(0, max(mean_a, mean_b)*1.5, 1000)
fig, ax = plt.subplots(figsize=(7,4))
ax.fill_between(x, beta.pdf(x, alpha_a, beta_a), color='skyblue', alpha=0.5)
ax.plot(x, beta.pdf(x, alpha_a, beta_a), color='blue', label='Control')
ax.fill_between(x, beta.pdf(x, alpha_b, beta_b), color='lightgreen', alpha=0.5)
ax.plot(x, beta.pdf(x, alpha_b, beta_b), color='green', label='Variant')
ax.axvline(mean_a, color='blue', linestyle='--', label=f"Control mean: {mean_a*100:.2f}%")
ax.axvline(mean_b, color='green', linestyle='--', label=f"Variant mean: {mean_b*100:.2f}%")
ax.set_xlabel('Conversion rate (%)')
ax.set_ylabel('Density')
ax.set_title('Posterior Distributions')
ax.legend()
st.pyplot(fig)

# Difference distribution
st.subheader("📉 Distribution of Lift (Variant − Control)")
fig2, ax2 = plt.subplots(figsize=(7,4))
counts, bins, patches = ax2.hist(delta*100, bins=50, edgecolor='white')
for patch, edge in zip(patches, bins[:-1]):
    patch.set_facecolor('lightgreen' if edge>0 else 'salmon')
ax2.axvline(0, color='black', linestyle='--')
ax2.set_xlabel('Lift (%)')
ax2.set_ylabel('Frequency')
ax2.set_title('Posterior Distribution of Lift')
st.pyplot(fig2)

# Time vs precision
if True:
    st.subheader("⏳ Days to Desired Precision")
    widths = np.linspace(0.5,3.0,50)
    scales = (ci_width/widths)**2
    totals = total_vis*scales
    extras = np.maximum(totals-total_vis,0)
    days = np.ceil(extras/daily)
    fig3, ax3 = plt.subplots(figsize=(7,4))
    ax3.plot(widths, days, marker='o')
    ax3.axvline(max_ci_width, color='red', linestyle='--', label='Your threshold')
    ax3.scatter([max_ci_width],[days[np.argmin(abs(widths-max_ci_width))]],color='red')
    ax3.set_xlabel('CI width (%)')
    ax3.set_ylabel('Days remaining')
    ax3.set_title('Time vs. Precision')
    ax3.legend()
    st.pyplot(fig3)
