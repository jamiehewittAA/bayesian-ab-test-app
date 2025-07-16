import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

# Page setup
st.set_page_config(page_title="Bayesian CRO Test Calculator", layout="centered")

# Title and description
st.title("🧪 Easy Bayesian CRO Test Calculator")
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
# Partial rollout advice percentage
holdback_pct = st.slider(
    "If not fully confident, hold back this % of traffic for control when rolling out Variant:",
    min_value=0, max_value=100, value=20, step=5,
    help="Specify what percentage of traffic to keep on control to validate uplift in live environment."
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

# 2. Priors (Optional)
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
# Theoretical CVR ranges based on selected CI
alpha_a0 = alpha_prior + conversions_a
beta_a0 = beta_prior + visitors_a - conversions_a
alpha_b0 = alpha_prior + conversions_b
beta_b0 = beta_prior + visitors_b - conversions_b
control_ci_low, control_ci_high = beta.ppf(ci_low_pct/100, alpha_a0, beta_a0) * 100, beta.ppf(ci_high_pct/100, alpha_a0, beta_a0) * 100
variant_ci_low, variant_ci_high = beta.ppf(ci_low_pct/100, alpha_b0, beta_b0) * 100, beta.ppf(ci_high_pct/100, alpha_b0, beta_b0) * 100
st.markdown(f"**Theoretical Control CVR range ({confidence_choice}% CI):** {control_ci_low:.2f}% – {control_ci_high:.2f}%")
st.markdown(f"**Theoretical Variant CVR range ({confidence_choice}% CI):** {variant_ci_low:.2f}% – {variant_ci_high:.2f}%")
st.markdown("---")
# Robustness threshold slider
robust_width_pct = st.slider(
    f"Max CI width (percentage points) for robust result at {confidence_choice}% confidence:",
    min_value=0.5, max_value=3.0,
    value={95:1.0,90:1.2,80:1.5}[confidence_choice], step=0.1,
    help="Set how wide the CI can be (in %) to consider results robust. Lower = more strict."
)
robust_width_target = robust_width_pct / 100
st.markdown("---")

# 4. Practical Impact (ROPE)
st.header("4. Practical Impact (ROPE)")
practical_display = st.slider("Ignore changes smaller than (%)",0.0,5.0,0.5,0.1,help="ROPE = range where differences are too small to matter.")
practical_effect = practical_display / 100.0
st.markdown("---")

# 5. Test Duration
st.header("5. Test Duration")
test_days = st.number_input("Days test has been running",min_value=1,value=7,help="Estimate days needed if more precision is required.")
st.markdown("---")

# Bayesian calculations
alpha_a, beta_a = alpha_prior+conversions_a, beta_prior+visitors_a-conversions_a
alpha_b, beta_b = alpha_prior+conversions_b, beta_prior+visitors_b-conversions_b
samples=200000
post_a = np.random.beta(alpha_a,beta_a,samples)
post_b = np.random.beta(alpha_b,beta_b,samples)
mean_a,mean_b=np.mean(post_a),np.mean(post_b)
delta=post_b-post_a

decision_prob=np.mean(delta>0)
abs_lift=mean_b-mean_a
rel_lift=(abs_lift/mean_a)*100
ci_low,ci_high=np.percentile(delta,[ci_low_pct,ci_high_pct])
ci_width=ci_high-ci_low
rope_overlap=np.mean((delta>-practical_effect)&(delta<practical_effect))
statsig=(ci_low>0)|(ci_high<0)
robust=statsig and (ci_width<robust_width_target) and (rope_overlap<0.95)

# Estimate data needs
total_vis=visitors_a+visitors_b
scale=(ci_width/robust_width_target)**2 if ci_width>0 else 1
needed=int(total_vis*scale)
extra_vis=max(needed-total_vis,0)
avg_vis_day=total_vis/test_days if test_days else 1
days_needed=int(np.ceil(extra_vis/avg_vis_day)) if avg_vis_day else None

# Financial projections
visitors_per_month=avg_vis_day*30
if conversion_value>0:
    monthly_gain=abs_lift*conversion_value*visitors_per_month
    annual_gain=monthly_gain*12
else:
    monthly_gain=annual_gain=None

# Financial modelling scenarios
if conversion_value>0:
    abs_low_lift=(variant_ci_low/100)-(control_ci_high/100)
    abs_high_lift=(variant_ci_high/100)-(control_ci_low/100)
    abs_avg_lift=abs_lift
    monthly_low_gain=abs_low_lift*conversion_value*visitors_per_month
    monthly_avg_gain=abs_avg_lift*conversion_value*visitors_per_month
    monthly_high_gain=abs_high_lift*conversion_value*visitors_per_month
    annual_low_gain=monthly_low_gain*12
    annual_avg_gain=monthly_avg_gain*12
    annual_high_gain=monthly_high_gain*12
    st.header("💼 Financial Projections Scenarios")
    st.markdown(
        """
        These figures estimate monthly and annual revenue impact under different uplift scenarios:

        - **Lowest uplift**: minimal increase from CI bounds
        - **Average uplift**: expected increase
        - **Highest uplift**: maximal increase from CI bounds
        """
    )
    st.markdown(f"**Lowest uplift ({abs_low_lift*100:.2f}%):** £{monthly_low_gain:,.2f}/month, £{annual_low_gain:,.2f}/year")
    st.markdown(f"**Average uplift ({abs_avg_lift*100:.2f}%):** £{monthly_avg_gain:,.2f}/month, £{annual_avg_gain:,.2f}/year")
    st.markdown(f"**Highest uplift ({abs_high_lift*100:.2f}%):** £{monthly_high_gain:,.2f}/month, £{annual_high_gain:,.2f}/year")
else:
    st.info("💡 Enter a 'Value per conversion' to see financial projections scenarios.")

st.markdown("---")
# Decision logic
if decision_prob>=prob_threshold:
    st.success("✅ Variant likely outperforms Control.")
elif(1-decision_prob)>=prob_threshold:
    st.error("⛔ Control likely outperforms Variant — do NOT implement Variant.")
    st.caption("High confidence Control is better. Revert traffic or test new ideas.")
else:
    st.warning("⚠️ Insufficient confidence that Variant > Control.")

# Interpretation guidance
if simple_mode:
    st.subheader("🔍 What does this mean?")
    if robust:
        st.markdown("Result is robust: confident in direction and magnitude.")
    elif decision_prob>=prob_threshold:
        st.markdown("Variant likely better, but effect size is uncertain.")
    else:
        st.markdown("No clear benefit of Variant—stick with Control or test more.")

# What to do next?
st.subheader("🛠️ What to do next?")
if robust:
    st.info("🚀 You have robust results—roll out Variant to 100% of traffic using your CRO tool.")
else:
    st.info(f"⚙️ Deploy Variant to {100-holdback_pct}% of traffic, hold back {holdback_pct}% as Control to validate uplift.")
    if not no_more_traffic and days_needed:
        st.info(f"🔍 Or collect ~{extra_vis:,} more visitors (~{days_needed} days) to achieve desired precision.")

# ⏳ Days Remaining vs Precision Goal
if show_decision_mode:
    st.markdown("---")
    st.subheader("⏳ How Many More Days to Reach Your Precision Goal?")
    st.markdown("""
    This chart shows how many extra days you need, beyond\
    your current run time, to achieve your chosen CI width:\
    - **Blue dots**: days remaining at each CI width
    - **Red dashed line**: your selected CI width
    - **Red dot**: days still needed for that width
    - **Blue X**: days already run
    """
    )
    robust_widths=np.linspace(0.005,0.03,50)
    scale_factors=(ci_width/robust_widths)**2
    suggested_total=total_vis*scale_factors
    extra_visitors=np.maximum(suggested_total-total_vis,0)
    days_remaining=np.ceil(extra_visitors/avg_vis_day)
    fig3,ax3=plt.subplots(figsize=(7,4))
    ax3.plot(robust_widths*100,days_remaining,marker='o',label='Days Remaining')
    cx=robust_width_target*100
    idx=np.argmin(np.abs(robust_widths-robust_width_target))
    cy=days_remaining[idx]
    ax3.axvline(cx,color='red',linestyle='--',linewidth=1.5,label='Selected CI Width')
    ax3.scatter([cx],[cy],color='red',zorder=5,label='Days Still Needed')
    ax3.scatter([cx],[0],color='blue',marker='X',s=100,label='Days Elapsed')
    ax3.text(cx+0.1,cy,f"+{int(cy)} days",va='bottom')
    ax3.text(cx+0.1,-max(days_remaining)*0.05,f"{test_days} days run",va='top')
    ax3.set_xlabel('CI Width Threshold (%)')
    ax3.set_ylabel('Days')
    ax3.set_title('Time to Desired Precision',pad=15)
    ax3.legend(loc='upper right',framealpha=0.8)
    fig3.tight_layout()
    st.pyplot(fig3)
    st.caption(f"You have run {test_days} days; the red dot shows extra days needed.")
