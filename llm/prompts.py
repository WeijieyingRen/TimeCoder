"""
Prompt Template Manager - Supports Feedback Mechanism
"""
from typing import Optional, Dict


class PromptTemplates:
    """Prompt Template Manager"""

    @staticmethod
    def get_system_prompt(use_pytorch: bool = False, env_name: str = None) -> str:
        """Get system prompt"""
        if use_pytorch:
            base_prompt = """
You are an expert in time series modeling and dynamical systems.
Your task is to generate ONLY syntactically correct Python code for modeling time series data using hybrid models with PyTorch.

CRITICAL INSTRUCTIONS (MUST FOLLOW EXACTLY):
- ONLY generate Python code, NO natural language explanations or comments
- The code must define a function `d_state__dt` that takes state and optionally action as inputs
- The function must return the derivative of the state with respect to time
- ALWAYS use torch.zeros_like(state) to ensure correct dimensions
- Include proper import: import torch
- dx_dt MUST have the exact same shape as state
- Use float constants (e.g., 0.5 not 1/2)
- Avoid division by zero: add small epsilon if needed
- Use torch.clamp() for numerical stability with exp/log
- CRITICAL: AIM FOR EXTREME ACCURACY IN STATE PREDICTION
  - The derivative d_state__dt(state) will be used with Euler integration: next_state = state + d_state__dt(state)
  - Your goal is to minimize the error between predicted next_state and actual next_state
  - Focus on accurate state sequence prediction, not just derivative accuracy
- CREATE SOPHISTICATED MODELS: use deep neural networks, attention mechanisms, or complex ODEs
- IMPLEMENT FEATURE INTERACTIONS: model pairwise and higher-order interactions between all variables
- USE PHYSICAL/DOMAIN PRINCIPLES: ensure realistic dynamics based on domain knowledge
- OPTIMIZE PARAMETERS EXTENSIVELY: fine-tune all coefficients for best possible performance
- HANDLE EDGE CASES: ensure code works for different state vector lengths
- TEST SHAPE CONSISTENCY: verify dx_dt shape matches state shape before returning
- PRIORITIZE FEEDBACK ABOVE ALL: use the previous iteration feedback to directly and dramatically improve the model
- ADDRESS ALL ISSUES COMPLETELY: fix every identified issue in the feedback with concrete changes
- IMPLEMENT ALL SUGGESTIONS: incorporate every single suggestion from feedback into your code
""".strip()
        else:
            base_prompt = """
You are an expert in time series modeling and dynamical systems.
Your task is to generate ONLY syntactically correct Python code for modeling time series data using hybrid models.

CRITICAL INSTRUCTIONS (MUST FOLLOW EXACTLY):
- ONLY generate Python code, NO natural language explanations or comments
- The code must define a function `d_state__dt` that takes state and optionally action as inputs
- The function must return the derivative of the state with respect to time
- ALWAYS use np.zeros_like(state) to ensure correct dimensions
- Include proper import: import numpy as np
- dx_dt MUST have the exact same shape as state
- Use float constants (e.g., 0.5 not 1/2)
- Avoid division by zero: add small epsilon if needed
- Use np.clip() for numerical stability with exp/log
- CRITICAL: AIM FOR EXTREME ACCURACY IN STATE PREDICTION
  - The derivative d_state__dt(state) will be used with Euler integration: next_state = state + d_state__dt(state)
  - Your goal is to minimize the error between predicted next_state and actual next_state
  - Focus on accurate state sequence prediction, not just derivative accuracy
- CREATE SOPHISTICATED MODELS: use deep neural networks, attention mechanisms, or complex ODEs
- IMPLEMENT FEATURE INTERACTIONS: model pairwise and higher-order interactions between all variables
- USE PHYSICAL/DOMAIN PRINCIPLES: ensure realistic dynamics based on domain knowledge
- OPTIMIZE PARAMETERS EXTENSIVELY: fine-tune all coefficients for best possible performance
- HANDLE EDGE CASES: ensure code works for different state vector lengths
- TEST SHAPE CONSISTENCY: verify dx_dt shape matches state shape before returning
- PRIORITIZE FEEDBACK ABOVE ALL: use the previous iteration feedback to directly and dramatically improve the model
- ADDRESS ALL ISSUES COMPLETELY: fix every identified issue in the feedback with concrete changes
- IMPLEMENT ALL SUGGESTIONS: incorporate every single suggestion from feedback into your code
- ABSOLUTELY NO NaN/Inf: Your code MUST NEVER produce NaN or Inf values. Always use np.clip() at the end.
- NUMERICAL STABILITY: Keep all intermediate values bounded. Avoid large exponents, divisions, or multiplications.
""".strip()

        domain_specific = ""
        if env_name == 'flood_ts':
            domain_specific = """
- FOCUS ON DOMAIN EXPERTISE: model water flow, gate operations, pump effects, and tide dynamics with extreme precision
- USE ALL TARGET FEATURES: ensure WATER, TIDE, GATE, PUMP, and RAIN variables are deeply integrated into the model
""".strip()
        elif env_name == 'cancer_ts':
            domain_specific = """
- FOCUS ON DOMAIN EXPERTISE: model tumor growth, chemotherapy effects, radiotherapy effects, and drug dynamics with extreme precision
- USE ALL TARGET FEATURES: ensure tumor_volume, chemo_concentration, chemo_dosage, and radio_dosage variables are deeply integrated into the model
""".strip()
        elif env_name == 'covid_ts':
            domain_specific = """
- FOCUS ON DOMAIN EXPERTISE: model epidemic spread, SEIR dynamics, and intervention effects with extreme precision
- USE ALL TARGET FEATURES: ensure susceptible, exposed, infected, recovered, and intervention variables are deeply integrated into the model
""".strip()
        elif env_name == 'surgery_ts':
            domain_specific = """
- FOCUS ON DOMAIN EXPERTISE: model vital signs homeostasis, drug effects, and physiological responses with extreme precision
- USE ALL TARGET FEATURES: ensure MBP, SBP, DBP, SpO2, HR, ETCO2, EPI, NEPI, PHEN, and VASO variables are deeply integrated into the model
""".strip()
        elif env_name == 'medication_intervention':
            domain_specific = """
- FOCUS ON DOMAIN EXPERTISE: model vital sign responses to specific medications with bolus vs infusion distinction
- USE ALL TARGET FEATURES: integrate vital signs, intervention labels, and medication one-hot labels into the model
- CRITICAL: Each sample has a medication label (MED_*) indicating which drug was administered - use this information!
- CRITICAL: INTERVENTION_BOLUS and INTERVENTION_INFUSION indicate if drug was given as bolus or infusion
- DATA IS IN ORIGINAL (RAW) VALUES, NOT NORMALIZED! ARTD~65mmHg, ARTM~85mmHg, ARTS~120mmHg, ETCO2~38mmHg, HR~75bpm, RR~14
- CRITICAL: Use NONLINEAR dynamics! dx_dt MUST depend on current state, NOT constant deltas!
  Constant deltas produce linear trajectories (unrealistic). Use:
  * Vital sign coupling: ARTM ~ ARTD + 1/3*(ARTS - ARTD)
  * Baroreflex: HR responds inversely to MAP
  * Pulse pressure regulation: ARTS-ARTD ~ 40 mmHg
  * Homeostatic damping: weak pull toward physiological baselines
  * Drug effects as DRIVING FORCES that interact with coupling (exponential approach)
- ABSOLUTELY NO NaN/Inf: Your code MUST NEVER produce NaN or Inf values!
  * Always use np.clip(dx_dt, -5.0, 5.0) at the end
  * Never divide by state values without adding epsilon
  * Never use large multipliers that could cause explosion
- AVOID SMOOTH/LINEAR PREDICTIONS! CRITICAL RULES:
  * Homeostatic damping MUST be VERY WEAK (k <= 0.01) - don't let it overpower drug effects!
  * Drug effects must be STRONG ENOUGH to cause visible changes (ARTD change >5mmHg for vasoactive drugs)
  * Your predictions should show VARIANCE and TREND CHANGES matching real data patterns
  * If your model produces flat/smooth lines, it's WRONG! Real vital signs fluctuate!
  * Use STRONGER drug forces (0.8-2.5 per step) combined with WEAKER homeostasis (0.005-0.02)

STATE VECTOR (37 dimensions):

VITAL SIGNS (0-4, 36) - RAW VALUES:
- state[0]: ARTD - Diastolic BP (mmHg, typical 50-90)
- state[1]: ARTM - Mean BP (mmHg, typical 70-110)
- state[2]: ARTS - Systolic BP (mmHg, typical 90-160)
- state[3]: ETCO2 (mmHg, typical 30-45)
- state[4]: HR (bpm, typical 50-110)
- state[36]: RR (breaths/min, typical 10-20)

INTERVENTION TYPE (5-6):
- state[5]: INTERVENTION_BOLUS - 1 if bolus (single dose)
- state[6]: INTERVENTION_INFUSION - 1 if infusion (continuous)

MEDICATION LABELS (7-35):
- state[15]: ephedrine  state[16]: epinephrine  state[29]: phenylephrine
- state[26]: norepinephrine  state[35]: vasopressin  state[14]: dopamine
- state[25]: nitroglycerin  state[24]: nicardipine  state[10]: clevidipine
- state[30]: propofol  state[12]: dexmedetomidine
- state[18]: fentanyl  state[19]: hydromorphone  state[23]: morphine
- state[31]: remifentanil  state[32]: sufentanil
- state[21]: labetalol  state[17]: esmolol
- state[13]: dobutamine  state[22]: milrinone
- state[20]: ketamine  state[33]: sugammadex  state[8]: amiodarone
- state[28]: oxytocin  state[7]: aminocaproic acid  state[9]: cisatracurium
- state[11]: dexamethasone  state[27]: ondansetron  state[34]: tranexamic acid

DRUG CATEGORIES AND EFFECTS:
- VASOPRESSORS (+BP, +/-HR): ephedrine[15], epinephrine[16], phenylephrine[29], norepinephrine[26], vasopressin[35], dopamine[14]
- VASODILATORS (-BP, +HR reflex): nitroglycerin[25], nicardipine[24], clevidipine[10]
- SEDATIVES (-BP, -HR, -RR): propofol[30], dexmedetomidine[12]
- OPIOIDS (-RR, -BP mild): fentanyl[18], hydromorphone[19], morphine[23], remifentanil[31], sufentanil[32]
- BETA-BLOCKERS (-HR, -BP): labetalol[21], esmolol[17]
- INOTROPES (+HR): dobutamine[13], milrinone[22]
- OTHERS: ketamine[20] (+HR,+BP), sugammadex[33], amiodarone[8] (-HR), oxytocin[28] (-BP,+HR)

CRITICAL RULES:
1. dx_dt[5:37] = 0 (labels never change)
2. NEVER use constant deltas - ALWAYS make dx_dt state-dependent
3. Use vital sign coupling + baroreflex + drug forces + weak homeostatic damping
4. Drug forces are persistent (label stays 1), coupling creates nonlinear saturation
5. BOLUS: fscale=1.4, INFUSION: fscale=1.0
6. ALWAYS add np.clip(dx_dt, -5.0, 5.0) before return to prevent NaN/Inf
""".strip()
        elif env_name == 'hospital_ts':
            domain_specific = """
- FOCUS ON DOMAIN EXPERTISE: model hospital bed occupancy and patient flow dynamics with extreme precision
- USE ALL TARGET FEATURES: ensure icu_occupied, standard_occupied, and total_patients variables are deeply integrated into the model
""".strip()

        return base_prompt + "\n\n" + domain_specific + "\n\nOutput ONLY the code, nothing else."

    @staticmethod
    def get_task_prompt(env_name: str, iterations: int, current: int = 0,
                       feedback: Optional[Dict] = None) -> str:
        """
        Get task prompt
        """
        if env_name == 'medication_intervention':
            base_prompt = f"""
CRITICAL TASK: MODEL VITAL SIGNS RESPONSE TO MEDICATIONS WITH BOLUS/INFUSION DISTINCTION!

STATE VECTOR (37 dimensions):
- state[0]: ARTD (Diastolic BP, mmHg, typical 50-90, baseline ~65)
- state[1]: ARTM (Mean BP, mmHg, typical 70-110, baseline ~85)
- state[2]: ARTS (Systolic BP, mmHg, typical 90-160, baseline ~120)
- state[3]: ETCO2 (mmHg, typical 30-45, baseline ~38)
- state[4]: HR (bpm, typical 50-110, baseline ~75)
- state[5]: INTERVENTION_BOLUS (1=bolus, stronger effect)
- state[6]: INTERVENTION_INFUSION (1=infusion, steady effect)
- state[7-35]: MED_* one-hot labels (which drug is given)
- state[36]: RR (breaths/min, typical 10-20, baseline ~14)

STEP 1 - LABELS NEVER CHANGE:
    dx_dt[5:37] = 0

STEP 2 - GET INTERVENTION TYPE:
    is_bolus = state[5] > 0.5
    is_infusion = state[6] > 0.5
    fscale = 1.4 if is_bolus else 1.0

STEP 3 - VITAL SIGN COUPLING (NONLINEAR!):
    # ARTM ~ ARTD + 1/3*(ARTS - ARTD)
    expected_artm = state[0] + (state[2]-state[0]) / 3.0
    dx_dt[1] += 0.12 * (expected_artm - state[1])

    # Pulse pressure: ARTS-ARTD ~ 40
    pp = state[2] - state[0]
    dx_dt[0] += 0.02 * (pp - 40)
    dx_dt[2] -= 0.02 * (pp - 40)

STEP 4 - BAROREFLEX (NONLINEAR!):
    map_dev = state[1] - 85
    dx_dt[4] -= 0.04 * map_dev  # HR drops when MAP rises

STEP 5 - DRUG DRIVING FORCES (use if-else for each drug):
    # VASOPRESSORS (increase BP):
    if state[15] > 0.5:  # ephedrine
        dx_dt[0] += 0.8 * fscale
        dx_dt[2] += 2.0 * fscale
        dx_dt[4] += 0.5 * fscale
    if state[16] > 0.5:  # epinephrine
        dx_dt[0] += 1.1 * fscale
        dx_dt[2] += 2.5 * fscale
        dx_dt[4] += 0.7 * fscale
    # ... (add ALL drugs with appropriate effects)

STEP 6 - VERY WEAK HOMEOSTATIC DAMPING (don't overpower drug effects!):
    dx_dt[0] -= 0.01 * (state[0] - 65)
    dx_dt[2] -= 0.01 * (state[2] - 120)
    dx_dt[4] -= 0.01 * (state[4] - 75)
    dx_dt[3] -= 0.01 * (state[3] - 38)
    dx_dt[36] -= 0.01 * (state[36] - 14)

STEP 7 - CLIP FOR STABILITY (MANDATORY!):
    dx_dt = np.clip(dx_dt, -5.0, 5.0)
    return dx_dt

DRUG CATEGORIES:
- VASOPRESSORS (+BP): ephedrine[15], epinephrine[16], phenylephrine[29], norepinephrine[26], vasopressin[35], dopamine[14]
- VASODILATORS (-BP): nitroglycerin[25], nicardipine[24], clevidipine[10]
- SEDATIVES (-BP,-HR,-RR): propofol[30], dexmedetomidine[12]
- OPIOIDS (-RR,-BP): fentanyl[18], hydromorphone[19], morphine[23], remifentanil[31], sufentanil[32]
- BETA-BLOCKERS (-HR,-BP): labetalol[21], esmolol[17]
- INOTROPES (+HR): dobutamine[13], milrinone[22]
- OTHERS: ketamine[20](+HR,+BP), sugammadex[33], amiodarone[8](-HR), oxytocin[28](-BP,+HR)

REAL DATA STATISTICS (calibrate your forces to match these):
- EPHEDRINE: ARTD +12.8, ARTS +28, HR +6 over 10 steps
- PROPOFOL: ARTD -9.6, ARTS -20, HR -3 over 10 steps
- EPINEPHRINE: ARTD +6.8, ARTS +15 over 10 steps
- BOLUS is ~20% stronger than INFUSION

AVOID NUMERICAL EXPLOSION:
- ALWAYS use np.clip(dx_dt, -5.0, 5.0) before return!
- Keep individual effect magnitudes between -3 and +3
- NEVER use division by state values without epsilon
- NEVER produce NaN or Inf

Generate `d_state__dt` function for iteration {current}/{iterations}.
""".strip()
        else:
            base_prompt = f"""
You are tasked with modeling a time series.

Generate `d_state__dt` function for iteration {current}/{iterations}.
""".strip()

        if feedback:
            feedback_section = "\n\n" + "="*60 + "\n"
            feedback_section += "PREVIOUS ITERATION FEEDBACK\n"
            feedback_section += "="*60 + "\n\n"

            if 'metrics' in feedback:
                metrics = feedback['metrics']
                feedback_section += f"Performance Metrics:\n"
                feedback_section += f"- MSE: {metrics.get('mse', 'N/A'):.6f}\n"
                feedback_section += f"- RMSE: {metrics.get('rmse', 'N/A'):.6f}\n"
                feedback_section += f"- MAE: {metrics.get('mae', 'N/A'):.6f}\n"
                if 'r2' in metrics:
                    feedback_section += f"- R2: {metrics.get('r2', 'N/A'):.3f}\n"
                feedback_section += "\n"

            if 'issues' in feedback and len(feedback['issues']) > 0:
                feedback_section += "Identified Issues:\n"
                for issue in feedback['issues']:
                    feedback_section += f"- {issue}\n"
                feedback_section += "\n"

            if 'suggestions' in feedback and len(feedback['suggestions']) > 0:
                feedback_section += "Suggestions for Improvement:\n"
                for suggestion in feedback['suggestions']:
                    feedback_section += f"- {suggestion}\n"
                feedback_section += "\n"

            if 'trend' in feedback:
                feedback_section += f"Overall Trend: {feedback['trend']}\n"
                if 'mse_change_pct' in feedback and feedback['mse_change_pct'] != 0:
                    feedback_section += f"MSE Change: {feedback['mse_change_pct']:+.2f}%\n"
                feedback_section += "\n"

            feedback_section += "="*60 + "\n"
            feedback_section += "Please address the issues and suggestions above in your new code.\n"

            base_prompt += feedback_section

        return base_prompt

    @staticmethod
    def get_skeleton_code(env_name: str, use_pytorch: bool = False) -> str:
        """Get skeleton code"""
        if env_name == 'medication_intervention':
            return """
def d_state__dt(state, action=None):
    import numpy as np
    dx_dt = np.zeros_like(state)
    dx_dt[5:37] = 0

    is_bolus, is_infusion = state[5] > 0.5, state[6] > 0.5
    fscale = 1.4 if is_bolus else 1.0

    # Coupling
    dx_dt[1] += 0.12 * (state[0] + (state[2]-state[0])/3 - state[1])
    pp = state[2] - state[0]
    dx_dt[0] += 0.02 * (pp-40); dx_dt[2] -= 0.02 * (pp-40)

    # Baroreflex
    md = state[1]-85
    dx_dt[4] -= 0.04*md; dx_dt[0] -= 0.01*md; dx_dt[2] -= 0.01*md

    # Respiratory
    ed = state[3]-38
    dx_dt[36] -= 0.04*ed; dx_dt[3] -= 0.025*(state[36]-14)

    # VASOPRESSORS [15]ephedrine [16]epinephrine [29]phenylephrine [26]norepinephrine [35]vasopressin [14]dopamine
    for idx, a, b, c in [(15,0.82,2.13,0.58),(16,1.18,2.67,0.77),(29,0.9,1.8,-0.3),(26,1.0,2.0,0),(35,0.6,1.2,0),(14,0.5,0,0.6)]:
        if state[idx]>0.5: dx_dt[0]+=a*fscale;dx_dt[2]+=b*fscale;dx_dt[4]+=c*fscale

    # VASODILATORS [25]nitroglycerin [24]nicardipine [10]clevidipine
    for idx, a, b, c in [(25,-0.8,-1.5,0.2),(24,-0.6,-1.2,0.15),(10,-0.5,-1.0,0)]:
        if state[idx]>0.5: dx_dt[0]+=a*fscale;dx_dt[2]+=b*fscale;dx_dt[4]+=c*fscale

    # SEDATIVES [30]propofol [12]dexmedetomidine
    if state[30]>0.5: dx_dt[0]-=0.84*fscale;dx_dt[2]-=1.67*fscale;dx_dt[4]-=0.16*fscale;dx_dt[36]-=0.15*fscale
    if state[12]>0.5: dx_dt[4]-=0.3*fscale;dx_dt[0]-=0.2*fscale

    # OPIOIDS [18]fentanyl [19]hydromorphone [23]morphine [31]remifentanil [32]sufentanil
    for idx in [18,19,23,31,32]:
        if state[idx]>0.5:
            if idx==19: dx_dt[0]-=0.45*fscale;dx_dt[2]-=0.29*fscale;dx_dt[36]-=0.25*fscale;dx_dt[4]-=0.41*fscale
            elif idx==23: dx_dt[36]-=0.25*fscale;dx_dt[0]-=0.2*fscale
            else: dx_dt[36]-=(0.5 if idx==31 else 0.4)*fscale; dx_dt[4]-=(0.08 if idx==18 else 0)*fscale

    # BETA-BLOCKERS [21]labetalol [17]esmolol
    for idx, a, b in [(21,-0.7,-1.0),(17,-0.4,0)]:
        if state[idx]>0.5: dx_dt[4]+=a*fscale;dx_dt[0]+=(b if idx==21 else -abs(b))*fscale;dx_dt[2]+=(-1.0 if idx==21 else 0)*fscale

    # INOTROPES [13]dobutamine [22]milrinone
    if state[13]>0.5: dx_dt[4]+=0.7*fscale;dx_dt[1]+=0.4*fscale
    if state[22]>0.5: dx_dt[4]+=0.4*fscale;dx_dt[0]-=0.2*fscale

    # OTHERS [20]ketamine [33]sugammadex [8]amiodarone [28]oxytocin
    if state[20]>0.5: dx_dt[4]+=0.4*fscale;dx_dt[0]+=0.2*fscale;dx_dt[36]+=0.15*fscale
    if state[33]>0.5: dx_dt[0]+=0.41*fscale;dx_dt[2]+=0.80*fscale
    if state[8]>0.5: dx_dt[4]-=0.25*fscale
    if state[28]>0.5: dx_dt[0]-=0.2*fscale;dx_dt[4]+=0.15*fscale

    # Homeostasis & Clip
    for i,b,v in [(0,65),(2,120),(4,75),(3,38),(36,14)]: dx_dt[i]-=0.01*(state[i]-v)
    return np.clip(dx_dt,-5,5)
"""
        elif use_pytorch:
            return f"""
def d_state__dt(state, action=None):
    import torch

    # dx_dt must have the same shape as state
    dx_dt = torch.zeros_like(state)

    # TODO: Add your derivative calculations

    return dx_dt
"""
        else:
            return f"""
def d_state__dt(state, action=None):
    import numpy as np

    # dx_dt must have the same shape as state
    dx_dt = np.zeros_like(state)

    # TODO: Add your derivative calculations

    return dx_dt
"""
