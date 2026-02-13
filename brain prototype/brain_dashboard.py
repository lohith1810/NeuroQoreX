import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib

try:
    matplotlib.use('TkAgg') 
except:
    pass 


CURRENT_PATIENT_DATA = {
    "id": "PX-001-HEALTHY",
    "name": "Asshuuu",
    "biomarkers": {
        "ROS": 0.79,    # Low Oxidative Stress (Normal is < 0.3)
        "Tau": 0.4,    # Minimal Protein Tangling (Normal is < 0.2)
        "ATP": 0.67,    # High Cellular Energy (Normal is > 0.8)
        "Redox": 0.15   # Balanced Chemical State (Normal is < 0.3)
    }
}

def run_diagnosis(data):
    """
    Analyzes biomarkers to determine diagnosis, risk, and color theme.
    Returns: dict with interpreted medical data.
    """
    b = data["biomarkers"]
    

    health_index = (b["ATP"] * 0.4) + ((1.0 - b["ROS"]) * 0.3) + ((1.0 - b["Tau"]) * 0.3)
    
    # Invert for Risk Factor (1.0 = Max Risk)
    risk_factor = 1.0 - health_index
    risk_factor = np.clip(risk_factor, 0, 1)

    # --- DETERMINE DIAGNOSIS & THEME ---
    if risk_factor < 0.3:
        diag = "NORMAL / HEALTHY"
        status_color = "#2ecc71" # Green/Blue
        base_cmap = "winter"     # Cool colors
        pulse_speed = 0.05       # Calm pulse
    elif risk_factor < 0.5:
        diag = "MILD OXIDATIVE STRESS"
        status_color = "#f1c40f" # Yellow
        base_cmap = "Wistia"     # Yellow/Orange
        pulse_speed = 0.1        # Mild agitation
    elif risk_factor < 0.75:
        diag = "EARLY STAGE NEURODEGENERATION"
        status_color = "#e67e22" # Orange
        base_cmap = "inferno"    # Dark Orange/Red
        pulse_speed = 0.2        # Fast pulse
    else:
        diag = "CRITICAL ISCHEMIC RISK"
        status_color = "#e74c3c" # Red
        base_cmap = "hot"        # Red/Black/White
        pulse_speed = 0.4        # Critical pulsing

    return {
        "diagnosis": diag,
        "risk_factor": risk_factor,
        "color": status_color,
        "cmap": base_cmap,
        "pulse_speed": pulse_speed
    }

# Run the diagnosis immediately
ANALYSIS = run_diagnosis(CURRENT_PATIENT_DATA)

# ==========================================
# 3. VISUALIZATION ENGINE
# ==========================================
plt.style.use('dark_background')
plt.rcParams['toolbar'] = 'None'

def generate_brain():
    """Generates 3D brain geometry."""
    n_points = 2500
    phi = np.random.uniform(0, 2*np.pi, n_points)
    costheta = np.random.uniform(-1, 1, n_points)
    theta = np.arccos(costheta)
    r = 1.0 * np.cbrt(np.random.uniform(0, 1, n_points))
    
    x = r * np.sin(theta) * np.cos(phi) * 1.5
    y = r * np.sin(theta) * np.sin(phi) * 1.8
    z = r * np.cos(theta) * 1.0
    
    # Calculate colors based on risk
    # If healthy, mostly uniform. If sick, high contrast.
    colors = np.random.uniform(0, 1, n_points)
    if ANALYSIS["risk_factor"] > 0.5:
        # Create a "Lesion" in the data (cluster of high values)
        dist_from_center = np.sqrt(x**2 + y**2 + z**2)
        mask = dist_from_center < 0.8
        colors[mask] = 1.0 # Hot spot
        
    return x, y, z, colors

if __name__ == "__main__":
    x, y, z, c_vals = generate_brain()

    fig = plt.figure(figsize=(16, 9), facecolor='#0b0f19')
    fig.canvas.manager.set_window_title(f"NeuroQoreX | {ANALYSIS['diagnosis']}")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 0.5])

    # --- 3D BRAIN PLOT ---
    ax1 = fig.add_subplot(gs[0], projection='3d', facecolor='#0b0f19')
    sc = ax1.scatter(x, y, z, c=c_vals, cmap=ANALYSIS["cmap"], alpha=0.6, s=10, edgecolors='none')
    ax1.axis('off')
    
    # --- HUD OVERLAY ---
    ax1.text2D(0.05, 0.95, f"PATIENT: {CURRENT_PATIENT_DATA['name']}", transform=ax1.transAxes, color='white', fontsize=14, fontweight='bold')
    ax1.text2D(0.05, 0.90, f"DIAGNOSIS: {ANALYSIS['diagnosis']}", transform=ax1.transAxes, color=ANALYSIS['color'], fontsize=12, fontweight='bold')
    ax1.text2D(0.05, 0.85, f"RISK INDEX: {int(ANALYSIS['risk_factor']*100)}%", transform=ax1.transAxes, color='white', fontsize=10)

    # --- BIOMARKER CHARTS ---
    ax2 = fig.add_subplot(gs[1], facecolor='#0b0f19')
    labels = list(CURRENT_PATIENT_DATA["biomarkers"].keys())
    values = list(CURRENT_PATIENT_DATA["biomarkers"].values())
    y_pos = np.arange(len(labels))
    
    bars = ax2.barh(y_pos, values, color=ANALYSIS['color'], alpha=0.7)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels, color='white')
    ax2.set_xlim(0, 1.0)
    ax2.set_title("BIOMARKER LEVELS", color='white')
    ax2.invert_yaxis()
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_color('white')
    ax2.spines['left'].set_visible(False)
    ax2.tick_params(axis='x', colors='white')

    # --- ANIMATION ---
    def update(frame):
        # Rotation
        ax1.view_init(elev=15, azim=frame * 0.3)
        
        # Dynamic Pulsing based on Diagnosis Severity
        pulse = np.sin(frame * ANALYSIS["pulse_speed"])
        
        # Healthy patients = Gentle breathing
        # Critical patients = Erratic pulsing
        scale_var = 5 if ANALYSIS["risk_factor"] < 0.5 else 25
        
        sizes = np.full(len(x), 10.0)
        sizes = sizes + (scale_var * (pulse + 1))
        sc.set_sizes(sizes)
        
        return sc, bars

    anim = FuncAnimation(fig, update, frames=np.arange(0, 360, 1), interval=50)
    plt.show()