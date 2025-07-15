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
    st.experimental_rerun()

# Page setup
st.set_page_config(page_title="Bayesian A/B Test Calculator", layout="centered")

st.title("🧪 Easy Bayesian A/B Test Calculator")
st.markdown("""
This calculator uses **Bayesian statistics** to evaluate A/B test results clearly and accurately.  
It gives you practical insights, even if you don’t have a statistics background.
""")

# Toggles
simple_mode = st.toggle("🧠 Show plain-English explanations", value=True)
show_robustness_explanation = st.toggle("📘 Explain Robustness Criteria", value=False)
show_decision_mode = st.toggle("🎯 Show Decision Recommendation Mode", value=True)
no_more_traffic = st.toggle("⚡ I don’t have more traffic — interpret the result anyway", value=False)

# Optional business value
conversion_value = st.number_input(
    "💰 Optional: Each conversion is worth (in $ or £)",
    min_value=0.0, value=0.0, step=0.1,
    help="Used to estimate expected gains/losses for action guidance."
)

# 1. Input Data
st.header("1️⃣ Enter Your A/B Test Data")
with st.expander("ℹ️ What are 'Visitors' and 'Conversions'?", expanded=False):
    st.markdown("""
    - **Visitors**: Number of users who saw each version.
    - **Conversions**: Users who completed your goal (e.g. signed up, purchased).
""")
col1, col2 = st.columns(2)
with col1:
    visitors_a = st.number_input("Visitors to A (Original)", min_value=1, value=1000)
    conversions_a = st.number_input("Conversions from A", min_value=0, value=50)
with col2:
    visitors_b = st.number_input("Visitors to B (Variant)", min_value=1, value=1000)
    conversions_b = st.number_input("Conversions from B", min_value=0, value=70)

# 2. Priors
st.header("2️⃣ Prior Beliefs (Optional)")
with st.expander("📝 Why set priors?", expanded=False):
    st.markdown("""
    Priors incorporate your existing beliefs:
    - α > β biases higher rates; α < β biases lower.
    - α=1,β=1 is neutral; α=10,β=10 means strong belief around 50%.
""")
col3, col4 = st.columns(2)
with col3:
    alpha_prior = st.number_input("Prior Alpha (α)", min_value=0.01, value=1.0)
with col4:
    beta_prior = st.number_input("Prior Beta (β)", min_value=0.01, value=1.0)

# 3. Confidence & Robustness
st.header("3️⃣ Confidence & Robustness")
confidence_choice = st.selectbox("Confidence level", [95, 90, 80], index=0)
prob_threshold = confidence_choice/100
ci_tail = (1-prob_threshold)/2*100
ci_low_pct, ci_high_pct = ci_tail, 100-ci_tail
robust_width_target = st.slider(
    f"CI width for robust result (at {confidence_choice}% confidence)",
    min_value=0.005, max_value=0.03,
    value={95:0.01,90:0.012,80:0.015}[confidence_choice], step=0.001,
    help="How narrow the credible interval must be to call a result robust."
)

# 4. ROPE
st.header("4️⃣ Practical Impact (ROPE)")
practical_display = st.slider(
    "Ignore differences smaller than (0%–5%)",
    min_value=0.0, max_value=5.0, value=0.5, step=0.1,
    help="Minor lifts below this % are 'too small to matter'."
)
practical_effect = practical_display/100

# 5. Test Duration
st.header("5️⃣ Test Duration")
test_days = st.number_input(
    "Days test has run", min_value=1, value=7,
    help="Use this to estimate days left if result isn't robust."
)

# Bayesian inference
alpha_a, beta_a = alpha_prior+conversions_a, beta_prior+visitors_a-conversions_a
alpha_b, beta_b = alpha_prior+conversions_b, beta_prior+visitors_b-conversions_b
samples=200_000
post_a = np.random.beta(alpha_a, beta_a, samples)
post_b = np.random.beta(alpha_b, beta_b, samples)
mean_a, mean_b = np.mean(post_a), np.mean(post_b)
delta = post_b-post_a

prob_b_better = np.mean(delta>0)
exp_abs_lift = mean_b-mean_a
exp_rel_lift = exp_abs_lift/mean_a*100
ci_low, ci_high = np.percentile(delta, [ci_low_pct, ci_high_pct])
ci_width = ci_high-ci_low
in_rope = np.mean((delta>-practical_effect)&(delta<practical_effect))
statsig = ci_low>0 or ci_high<0
robust = statsig and ci_width<robust_width_target and in_rope<0.95

# Estimate more data needed
total_vis=visitors_a+visitors_b
factor=(ci_width/robust_width_target)**2 if ci_width>0 else 1
needed_tot=int(total_vis*factor)
more_vis=max(needed_tot-total_vis,0)
daily_vis=total_vis/test_days
days_left=int(np.ceil(more_vis/daily_vis)) if daily_vis else None

# Expected gain
exp_gain=None
if conversion_value>0: exp_gain=exp_abs_lift*conversion_value*visitors_b

# Summary
st.header("📊 Test Summary")
if simple_mode:
    st.markdown(f"📈 Expected lift: **{exp_rel_lift:.2f}%** or **{exp_abs_lift:.4f}** pts")
    st.metric("P(B>A)",f"{prob_b_better*100:.1f}%")
    if prob_b_better>=prob_threshold: st.success("✅ B likely beats A")
    else: st.warning("⚠️ Not confident B beats A")
    if exp_gain: st.info(f"💡 Expected gain: {exp_gain:,.2f}")
    if robust: st.success("🎯 Result is robust")
    elif no_more_traffic: st.warning("⚠️ Promising but not robust; caution advised")
    else:
        st.warning("🚧 Not robust; more data helps")
        if days_left: st.markdown(f"Collect **{more_vis:,} more** users (~**{days_left} days**)")
else:
    st.write(f"Expected lift: {exp_rel_lift:.2f}%")
    st.write(f"Absolute lift: {exp_abs_lift:.4f}")
    st.write(f"P(B>A): {prob_b_better*100:.2f}%")
    st.write(f"CI width: {ci_width:.4f}")
    st.write(f"ROPE overlap: {in_rope*100:.1f}%")
    st.write("Stat sig:",statsig)
    st.write("Robust:",robust)

if show_robustness_explanation:
    st.markdown("""
    **Robust =** stat sig + precise (narrow CI) + practical
    """)

if show_decision_mode:
    st.subheader("🧠 Decision Guidance")
    if robust: st.success("Act now")
    elif prob_b_better>=prob_threshold and in_rope<0.5: st.info("Consider action with caution")
    else: st.warning("Hold off")

# Plots
st.header("📈 Posterior & CI")
maxr=max(mean_a,mean_b)
x=np.linspace(0,maxr*1.5,1000)
fig,ax=plt.subplots()
ax.plot(x,beta.pdf(x,alpha_a,beta_a),label='A')
ax.plot(x,beta.pdf(x,alpha_b,beta_b),label='B')
ax.set_xlabel('Conversion Rate')
ax.set_ylabel('Density')
ax.legend()
st.pyplot(fig)

st.subheader("📉 Difference Distribution")
fig2,ax2=plt.subplots()
ax2.hist(delta,bins=100,alpha=0.7)
ax2.axvline(0, color='red',linestyle='--')
ax2.set_xlabel('Difference (B-A)')
ax2.set_ylabel('Frequency')
st.pyplot(fig2)
