import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

# Page setup
st.set_page_config(page_title="Bayesian CRO Test Calculator", layout="centered")

# --- Title & Description ----------------------------------------------------
st.title("🧪 Bayesian CRO Test Calculator")
st.markdown("Use Bayesian analysis to make clear, data‑driven A/B‑test decisions with practical advice.")
st.markdown("---")

# --- Mode Toggles -----------------------------------------------------------
col1, col2 = st.columns(2)
with col1:
    simple_mode           = st.checkbox("Show plain‑English explanations", value=True)
    no_more_traffic       = st.checkbox("I don’t have more traffic—interpret anyway", value=True)
with col2:
    show_robustness_expl  = st.checkbox("Explain Robustness Criteria", value=True)
    show_decision_mode    = st.checkbox("Show Decision Guidance",  value=True)
st.markdown("---")

# --- Optional Business Value -----------------------------------------------
conversion_value = st.number_input(
    "Optional: Value per conversion (e.g. £10)",
    min_value=0.0, value=0.0, step=0.1,
    help="Enter £ value to turn lift into revenue impact."
)
st.markdown("---")

# --- INPUTS -----------------------------------------------------------------
st.header("🔢 Inputs")
st.markdown(
    """
    • **Visitors**: users exposed to each version  
    • **Conversions**: goal completions per version  
    • **Priors**: encode historical belief (α/β). Neutral = 1,1  
    • **Confidence level**: 95 %, 90 % or 80 %  
    • **Max CI width**: required precision (%) for robustness  
    • **ROPE**: % change considered negligible  
    • **Days run**: test runtime so far (for time estimates)  
    • **Value per conversion**: optional £ value for revenue calcs
    """
)

# 1️⃣ Test Data ----------------------------------------------------------------
colA, colB = st.columns(2)
with colA:
    visitors_a    = st.number_input("Visitors – Control",  min_value=1, value=1_000)
    conversions_a = st.number_input("Conversions – Control", min_value=0, value=50)
with colB:
    visitors_b    = st.number_input("Visitors – Variant",  min_value=1, value=1_000)
    conversions_b = st.number_input("Conversions – Variant", min_value=0, value=70)

cvr_a = conversions_a/visitors_a*100
cvr_b = conversions_b/visitors_b*100
st.markdown(f"**Control CVR:** {cvr_a:.2f}% | **Variant CVR:** {cvr_b:.2f}%")
st.markdown("---")

# 2️⃣ Priors -------------------------------------------------------------------
st.header("2. Priors (Optional)")
with st.expander("What are priors?", expanded=False):
    st.markdown(
        """
        A **prior** expresses what you believe *before* seeing today’s data.  
        In a Beta‑Binomial model:
        * Prior mean = α / (α+β)  
        * Strength   = α+β  (acts like that many *pseudo‑visitors*)
        """
    )

preset = st.selectbox(
    "Choose a prior preset",
    [
        "Neutral (no prior)",
        "Mild uplift expected (+2 %, 20 pseudo‑visitors)",
        "Moderate uplift expected (+3.5 %, 30 pseudo‑visitors)",
        "Strong uplift expected (+5 %, 50 pseudo‑visitors)",
        "Very strong uplift expected (+7.5 %, 70 pseudo‑visitors)",
        "Exceptional uplift expected (+10 %, 100 pseudo‑visitors)",
        "Custom"
    ],
    index=0,
)
base_cvr = conversions_a/visitors_a if visitors_a>0 else 0.05

if preset == "Neutral (no prior)":
    alpha_prior, beta_prior = 1.0, 1.0
elif preset == "Mild uplift expected (+2 %, 20 pseudo‑visitors)":
    strength    = 20
    prior_mean  = max(1e-4, base_cvr*1.02)
    alpha_prior = strength*prior_mean
    beta_prior  = strength*(1-prior_mean)
elif preset == "Moderate uplift expected (+3.5 %, 30 pseudo‑visitors)":
    strength    = 30
    prior_mean  = max(1e-4, base_cvr*1.035)
    alpha_prior = strength*prior_mean
    beta_prior  = strength*(1-prior_mean)
elif preset == "Strong uplift expected (+5 %, 50 pseudo‑visitors)":
    strength    = 50
    prior_mean  = max(1e-4, base_cvr*1.05)
    alpha_prior = strength*prior_mean
    beta_prior  = strength*(1-prior_mean)
elif preset == "Very strong uplift expected (+7.5 %, 70 pseudo‑visitors)":
    strength    = 70
    prior_mean  = max(1e-4, base_cvr*1.075)
    alpha_prior = strength*prior_mean
    beta_prior  = strength*(1-prior_mean)
elif preset == "Exceptional uplift expected (+10 %, 100 pseudo‑visitors)":
    strength    = 100
    prior_mean  = max(1e-4, base_cvr*1.10)
    alpha_prior = strength*prior_mean
    beta_prior  = strength*(1-prior_mean)
else:  # Custom
    ca, cb = st.columns(2)
    with ca:
        alpha_prior = st.number_input("Prior α", min_value=0.01, value=1.0)
    with cb:
        beta_prior  = st.number_input("Prior β", min_value=0.01, value=1.0)

st.caption(f"Prior ≈ **{alpha_prior+beta_prior:.0f} pseudo‑visitors** at **{alpha_prior/(alpha_prior+beta_prior)*100:.2f}%** CVR")
st.markdown("---")

# 3️⃣ Confidence & Robustness --------------------------------------------------
confidence = st.selectbox("Confidence level (%)", [95,90,80], index=0)
prob_threshold = confidence/100
ci_tail = (1-prob_threshold)/2*100
ci_low_pct, ci_high_pct = ci_tail, 100-ci_tail

max_ci_width_pct = st.slider("Max CI width (%) for robustness",0.5,3.0,1.0,0.1)
max_ci_width     = max_ci_width_pct/100

# ROPE ------------------------------------------------------------------------
rope_pct = st.slider("ROPE (%) – ignore changes smaller than",0.0,5.0,0.5,0.1)
rope      = rope_pct/100

# Test days -------------------------------------------------------------------
test_days = st.number_input("Days test has run",min_value=1,value=7)

st.markdown("---")

# --- Bayesian Computation ----------------------------------------------------
α_a, β_a = alpha_prior+conversions_a, beta_prior+visitors_a-conversions_a
α_b, β_b = alpha_prior+conversions_b, beta_prior+visitors_b-conversions_b
samp=200_000
post_a = np.random.beta(α_a,β_a,samp)
post_b = np.random.beta(α_b,β_b,samp)
mean_a, mean_b = post_a.mean(), post_b.mean()
Δ      = post_b - post_a

p_better = (Δ>0).mean()
abs_lift = mean_b - mean_a
rel_lift = abs_lift/mean_a*100
ci_low, ci_high = np.percentile(Δ,[ci_low_pct,ci_high_pct])
ci_width = ci_high - ci_low

sig      = (ci_low>0) | (ci_high<0)
robust   = sig and (ci_width<=max_ci_width) and ((np.abs(Δ)<rope).mean()<0.95)

# Precision/time calcs
n_total    = visitors_a+visitors_b
scale      = (ci_width/max_ci_width)**2
need_total = int(n_total*scale)
extra_vis  = max(need_total-n_total,0)
avg_day    = n_total/test_days
more_days  = int(np.ceil(extra_vis/avg_day)) if avg_day>0 else None

# --- Financial Scenarios -----------------------------------------------------
if conversion_value>0:
    visitors_month = avg_day*30
    worst_gain = ci_low*conversion_value*visitors_month
    exp_gain   = abs_lift*conversion_value*visitors_month
    best_gain  = ci_high*conversion_value*visitors_month
    st.header("💼 Financial Scenarios (per month)")
    st.markdown(f"Worst‑case: £{worst_gain:,.2f} | Expected: £{exp_gain:,.2f} | Best‑case: £{best_gain:,.2f}")
else:
    st.info("💡 Enter £ value per conversion to see revenue estimates.")

# --- Key Results -------------------------------------------------------------
st.header("📊 Key Results")
st.markdown(f"**P(Variant > Control):** {p_better*100:.1f}%")
st.markdown(f"**Expected lift:** {rel_lift:.2f}% relative ({abs_lift*100:.2f} pp)")
st.markdown(f"**{confidence}% CI:** [{ci_low*100:.2f}%, {ci_high*100:.2f}%]  (width {ci_width*100:.2f}%)")
st.markdown("---")

# --- Interpretation ----------------------------------------------------------
if simple_mode:
    st.subheader("🔍 What does this mean?")
    if robust:
        st.markdown("Result is **robust**: confident in direction & size.")
    elif p_better >= prob_threshold:
        msg = "Variant likely better, but CI spans zero — effect might be small or even negative."
        if conversion_value>0:
            msg += " Worst‑case revenue scenario reflects this risk."
        st.markdown(msg)
    else:
        st.markdown("No clear benefit — Variant could underperform Control.")

# --- Actionable Next Steps ---------------------------------------------------
st.subheader("🛠️ What to do next?")
if robust:
    st.success("🚀 Roll out Variant to 100 % of traffic — result is robust.")
else:
    holdback = max(5, min(10, int((more_days/(more_days+test_days))*100) if more_days else 10))
    st.info(
        f"⚙️ **Fast option:** Ship Variant to {100-holdback}% of traffic, keep {holdback}% on Control for a short {min(more_days or 3,3)}‑day monitor."
    )
    if more_days:
        st.info(f"🔍 **Data option:** gather ~{extra_vis:,} more visitors (~{more_days} days) for robustness.")

# --- Plots -------------------------------------------------------------------
## Posterior CVR
st.subheader("📈 Posterior Distributions of CVR")
x = np.linspace(0, max(mean_a,mean_b)*1.5, 1000)
fig1, ax1 = plt.subplots(figsize=(7,4))
ax1.fill_between(x, beta.pdf(x,α_a,β_a), color='skyblue', alpha=0.5)
ax1.plot(x, beta.pdf(x,α_a,β_a), color='blue', label=f"Control mean {mean_a*100:.2f}%")
ax1.fill_between(x, beta.pdf(x,α_b,β_b), color='lightgreen', alpha=0.5)
ax1.plot(x, beta.pdf(x,α_b,β_b), color='green', label=f"Variant mean {mean_b*100:.2f}%")
ax1.set_xlabel('Conversion rate (%)'); ax1.set_ylabel('Density'); ax1.legend(); ax1.grid(alpha=0.3)
st.pyplot(fig1)

## Lift distribution
st.subheader("📉 Posterior Distribution of Lift (Variant − Control)")
fig2, ax2 = plt.subplots(figsize=(7,4))
counts,bins,patches = ax2.hist(Δ*100, bins=50, edgecolor='white')
for p,e in zip(patches,bins[:-1]):
    p.set_facecolor('lightgreen' if e>0 else 'salmon')
ax2.axvline(0,color='black',ls='--'); ax2.set_xlabel('Lift (%)'); ax2.set_ylabel('Frequency'); ax2.grid(alpha=0.3)
st.pyplot(fig2)

## CI width vs sample size
st.subheader("📊 CI Width vs Total Sample Size")
size_grid = np.linspace(n_total, n_total*3, 40, dtype=int)
ci_w=[]
