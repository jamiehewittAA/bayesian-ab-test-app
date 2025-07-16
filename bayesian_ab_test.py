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

# Mode toggles
col1, col2 = st.columns(2)
with col1:
    simple_mode = st.checkbox("Show plain-English explanations", value=True)
    no_more_traffic = st.checkbox("I don’t have more traffic—interpret result anyway", value=True)
with col2:
    show_robustness_explanation = st.checkbox("Explain Robustness Criteria", value=True)
    show_decision_mode = st.checkbox("Show Decision Guidance", value=True)
st.markdown("---")

# Optional business value
conversion_value = st.number_input(
    "Optional: Value per conversion (e.g. £10)", min_value=0.0, value=0.0, step=0.1,
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

# Display current conversion rates
cvr_control = (conversions_a / visitors_a) * 100 if visitors_a > 0 else 0
cvr_variant = (conversions_b / visitors_b) * 100 if visitors_b > 0 else 0
st.markdown(f"**Control CVR:** {cvr_control:.2f}%  |  **Variant CVR:** {cvr_variant:.2f}%")

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
# 3. Confidence & Robustness
st.header("3. Confidence & Robustness")
confidence_choice = st.selectbox("Select confidence level (%)", [95, 90, 80], index=0)
prob_threshold = confidence_choice / 100.0
ci_tail = (1 - prob_threshold) / 2 * 100
ci_low_pct, ci_high_pct = ci_tail, 100 - ci_tail
# Insert theoretical CVR ranges based on selected CI
alpha_a0 = alpha_prior + conversions_a
beta_a0 = beta_prior + visitors_a - conversions_a
alpha_b0 = alpha_prior + conversions_b
beta_b0 = beta_prior + visitors_b - conversions_b
control_ci_low, control_ci_high = beta.ppf(ci_low_pct/100, alpha_a0, beta_a0) * 100, beta.ppf(ci_high_pct/100, alpha_a0, beta_a0) * 100
variant_ci_low, variant_ci_high = beta.ppf(ci_low_pct/100, alpha_b0, beta_b0) * 100, beta.ppf(ci_high_pct/100, alpha_b0, beta_b0) * 100
st.markdown(f"**Theoretical Control CVR range ({confidence_choice}% CI):** {control_ci_low:.2f}% – {control_ci_high:.2f}%")
st.markdown(f"**Theoretical Variant CVR range ({confidence_choice}% CI):** {variant_ci_low:.2f}% – {variant_ci_high:.2f}%")
st.markdown("---")
# Slider in percent for clarity and granularity
robust_width_pct = st.slider(
    f"Max CI width (percentage points) for robust result at {confidence_choice}% confidence:",
    min_value=0.5, max_value=3.0,
    value={95:1.0, 90:1.2, 80:1.5}[confidence_choice],
    step=0.1,
    help="Set how wide the credible interval can be (in %) to consider results robust. Lower = more strict."
)
robust_width_target = robust_width_pct / 100
st.markdown("---")
# 4. Practical Impact (ROPE) (ROPE)
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

# — Perform Bayesian Calculations —
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

# — Results Summary —
st.header("📊 Results Summary")
if simple_mode:
    st.markdown(f"**Expected lift:** {rel_lift:.2f}%")
    st.markdown(f"**Chance Variant > Control:** {decision_prob*100:.1f}%")
    if conversion_value > 0:
        st.markdown(f"💰 **Expected monthly gain:** £{monthly_gain:,.2f}")
        st.caption("Projected monthly gain based on test traffic.")
        st.markdown(f"📈 **Expected annual gain:** £{annual_gain:,.2f}")
        st.caption("Projected annual gain based on test traffic.")
    # Decision logic
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
            if decision_prob >= prob_threshold:
                st.warning("⚠️ Promising but not robust—proceed with caution.")
                st.caption("Consider limiting exposure, monitoring metrics closely, and planning follow-up tests to verify performance before full rollout.")
            else:
                st.warning("⚠️ Variant underperforms Control—do NOT implement Variant but monitor and test alternatives.")
                st.caption("Based on current data, Control is stronger. You may switch back or explore new variants.")
        else:
            st.warning("🚧 Not yet robust—consider collecting more data.")
            if days_needed:
                st.markdown(f"🔍 Collect ~{extra_vis:,} more visitors (~{days_needed} days) for robust results.")
else:
    st.subheader("🧮 Detailed Metrics")
    st.write(f"- **Expected lift**: {rel_lift:.2f}%")
    st.write(f"- **Absolute lift**: {abs_lift:.4f}")
    st.write(f"- **Probability Variant > Control**: {decision_prob*100:.2f}%")
    st.write(f"- **{confidence_choice}% CI**: [{ci_low:.4f}, {ci_high:.4f}] (width {ci_width:.4f})")
    st.caption("A narrower CI indicates greater precision.")
    st.write(f"- **ROPE overlap**: {rope_overlap*100:.1f}%")
    st.write(f"- **Statistically significant**: {statsig}")
    st.write(f"- **Robust**: {robust}")

# Posterior distributions
st.markdown("---")
st.header("📈 Posterior Distributions of Conversion Rates")
# Introduce plot
st.markdown("""
This chart shows the full range of conversion rates the data support for Control and Variant.  
The shaded areas represent the posterior distributions; peaks show the most likely rates.
""")

# Prepare plot
max_rate = max(mean_a, mean_b)
x = np.linspace(0, max_rate*1.5, 1000)
fig1, ax1 = plt.subplots(figsize=(7,4))

# Plot distributions with shading
ax1.fill_between(x, beta.pdf(x, alpha_a, beta_a), color='skyblue', alpha=0.5)
ax1.plot(x, beta.pdf(x, alpha_a, beta_a), color='blue', label='Control')
ax1.fill_between(x, beta.pdf(x, alpha_b, beta_b), color='lightgreen', alpha=0.5)
ax1.plot(x, beta.pdf(x, alpha_b, beta_b), color='green', label='Variant')

# Mark means
ax1.axvline(mean_a, color='blue', linestyle='--', label=f"Control mean: {mean_a*100:.2f}%")
ax1.axvline(mean_b, color='green', linestyle='--', label=f"Variant mean: {mean_b*100:.2f}%")

# Styling
ticks = np.linspace(0, max_rate*1.5, 6)
ax1.set_xticks(ticks)
ax1.set_xticklabels([f"{t*100:.1f}%" for t in ticks])
ax1.set_xlabel('Conversion rate (%)')
ax1.set_ylabel('Density')
ax1.set_title('Posterior Distributions of Conversion Rates', pad=20)
ax1.grid(alpha=0.3)
ax1.legend(loc='upper right', framealpha=0.8)
fig1.tight_layout()
st.pyplot(fig1)

# Difference histogram
st.subheader("📉 Posterior Distribution of Difference: Variant − Control")
st.markdown(
    "This chart shows the distribution of possible differences in conversion rate (Variant minus Control).  
    Bars to the right of zero indicate the Variant is better; bars to the left indicate the Control is better."
)
fig2, ax2 = plt.subplots(figsize=(7,4))
# Plot histogram
data = delta
tot_bins = 50
counts, bins, patches = ax2.hist(data, bins=tot_bins, color='lightgray', edgecolor='white')
# Color negative and positive
for patch, edge in zip(patches, bins[:-1]):
    patch.set_facecolor('salmon' if edge < 0 else 'lightgreen')
# Zero line
ax2.axvline(0, color='black', linestyle='--', linewidth=1)
# Annotations
ax2.text(max_rate*1.0, max(counts)*0.9, f"P(Variant > Control): {decision_prob*100:.1f}%", color='darkgreen', fontsize=10)
ax2.text(-max_rate*0.2, max(counts)*0.9, f"P(Control > Variant): {(1-decision_prob)*100:.1f}%", color='darkred', fontsize=10)

# Styling
ax2.set_xlabel('Conversion rate difference (%)')
ax2.set_ylabel('Frequency')
ax2.set_title('Posterior Distribution of the Difference', pad=20)
ax2.grid(alpha=0.3)
fig2.tight_layout()
st.pyplot(fig2)
st.subheader("📉 Difference (Variant − Control)")
st.markdown("""
This chart shows how much the conversion rate is likely to change when you move from Control to Variant.  
- **Right of zero**: Variant outperforms Control.
- **Left of zero**: Control outperforms Variant.
The taller the bar, the more likely that difference is.
""")
fig2, ax2 = plt.subplots(figsize=(6,3))
# Plot full distribution outline
counts, bins, patches = ax2.hist(delta, bins=50, color='lightgray', alpha=1)
# Highlight negative and positive regions
for patch, edge in zip(patches, bins[:-1]):
    if edge < 0:
        patch.set_facecolor('salmon')
    else:
        patch.set_facecolor('lightgreen')
# Vertical line at zero
ax2.axvline(0, color='black', linestyle='--', linewidth=1)
# Simplified legend
ax2.legend(["Control better", "Variant better"], loc='upper left')
# Clean up labels and layout
ax2.set_xlabel('Conversion rate difference (Variant − Control)')
ax2.set_ylabel('Frequency')
ax2.set_title('Posterior Distribution of the Difference', pad=15)
fig2.tight_layout()
st.pyplot(fig2)

# ⏳ Days Remaining vs Robustness Threshold
if show_decision_mode:
    st.subheader("⏳ How Many More Days to Reach Your Precision Goal?")
    st.markdown(
        """
        This chart shows **additional days** needed on top of your current test duration to achieve the desired **credible interval (CI) width**.
        - The **blue line** plots days remaining vs. different CI width thresholds.
        - The **red dashed line** marks the CI width you selected.
        - The **red dot** shows the **remaining** days needed for your chosen CI width.
        """
    )
    robust_widths = np.linspace(0.005, 0.03, 50)
    scale_factors = (ci_width / robust_widths) ** 2
    suggested_total = total_vis * scale_factors
    extra_visitors = np.maximum(suggested_total - total_vis, 0)
    days_remaining = np.ceil(extra_visitors / avg_vis_day)
    
    fig3, ax3 = plt.subplots(figsize=(6,3))
    ax3.plot(robust_widths * 100, days_remaining, marker='o', label='Days Remaining')
    current_x = robust_width_target * 100
    idx = np.argmin(np.abs(robust_widths - robust_width_target))
    current_y = days_remaining[idx]
    # vertical line and dot for chosen threshold
    ax3.axvline(current_x, color='red', linestyle='--', linewidth=1.5, label='Chosen CI Width')
    ax3.scatter([current_x], [current_y], color='red', zorder=5, label='Remaining Days')
    # mark elapsed test days
    ax3.scatter([current_x], [0], color='blue', marker='X', s=100, label='Days Elapsed')
    ax3.text(current_x + 0.05, 0, f"{test_days} days run", va='bottom')
    ax3.text(current_x + 0.05, current_y, f"+{int(current_y)} days more", va='bottom')
    
    ax3.set_xlabel("CI Width Threshold (%)")
    ax3.set_ylabel("Days Remaining")
    ax3.set_xlim(robust_widths.min()*100, robust_widths.max()*100)
    ax3.set_ylim(0, days_remaining.max()*1.1)
    ax3.set_title("Time Needed to Reach Precision")
    ax3.legend()
    fig3.tight_layout()
    st.pyplot(fig3)
    st.caption(f"You have run {test_days} days so far; the red dot shows how many additional days are needed.")
