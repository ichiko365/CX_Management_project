import streamlit as st
import streamlit.components.v1 as components

# Use wide layout so our custom HTML can occupy full width and collapse sidebar for symmetry
st.set_page_config(
  page_title="Login Page",
  page_icon="🔑",
  layout="wide",
  initial_sidebar_state="collapsed",
)

# Set page background (you can swap the image URL)
st.markdown(
    """
    <style>
  [data-testid="stAppViewContainer"]{
    position: relative;
        background-image:url('https://images.unsplash.com/photo-1501785888041-af3ef285b470');
        background-size:cover; background-position:center;
    }
    /* optional subtle dark overlay to improve contrast */
    [data-testid="stAppViewContainer"]::before{
        content:""; position:fixed; inset:0; pointer-events:none;
        background: linear-gradient(to bottom, rgba(0,0,0,.25), rgba(0,0,0,.15) 35%, rgba(0,0,0,.35));
        z-index:0;
    }

  /* Make Streamlit content full-bleed so the component can center on the whole screen */
  .block-container{
    max-width: 100% !important;
    padding-left: 0 !important;
    padding-right: 0 !important;
    padding-top: 0 !important;
  }
    </style>
    """,
    unsafe_allow_html=True,
)

html = r"""
<div class="viewport">
  <div class="glass-card">
    <h2 class="title">Login</h2>

    <div class="field">
      <span class="chip">Username</span>
      <input type="text" placeholder="Ludiflex" />
      <span class="suffix" aria-hidden="true">&#128269;</span>
    </div>

    <div class="field">
      <span class="chip">Password</span>
      <input id="pwd" type="password" placeholder="••••••••" />
      <button class="eye" type="button" onclick="
        const p = document.getElementById('pwd');
        p.type = p.type === 'password' ? 'text' : 'password';
      ">👁</button>
    </div>

    <div class="row">
      <label class="remember"><input type="checkbox"/> Remember me</label>
      <a class="link" href="#">Forgot password?</a>
    </div>

    <button class="login" type="button" onclick="alert('Demo only');">Login</button>

    <p class="cta">Don't have an account? <a class="link" href="#">Register</a></p>
  </div>
</div>

<style>
:root{ --glass: rgba(17,25,40,.45); --border: rgba(255,255,255,.25); --fg:#fff; }
*{ box-sizing: border-box; }
body{ margin:0; font-family: system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, 'Helvetica Neue', Arial, 'Apple Color Emoji','Segoe UI Emoji'; }
/* avoid clipping; let layout decide */
html, body{ overflow-x: visible; }

/* Center the card perfectly both vertically and horizontally relative to the real viewport */
.viewport{
  position: fixed; inset: 0;
  display: grid; place-items: center;
  z-index: 3; padding: 0;
}

.glass-card{
  width: clamp(560px, 70vw, 1200px);
  max-width:100%;
  min-height: 620px;
  padding:44px 40px 28px;
  border-radius:70px;
  border:1px solid var(--border);
  background: var(--glass);
  backdrop-filter: blur(18px) saturate(180%);
  -webkit-backdrop-filter: blur(18px) saturate(180%);
  box-shadow: 0 16px 48px rgba(0,0,0,.38), inset 0 1px 0 rgba(255,255,255,.12);
  color: var(--fg);
  margin: 0 auto;
}

.title{ margin:0 0 26px; text-align:center; font-weight:800; letter-spacing:.3px; font-size:32px; }

.field{ position:relative; margin: 20px 0; }
.field input{
  width:100%; height:64px; padding: 0 60px 0 18px;
  border-radius:20px; border:1px solid var(--border);
  background: rgba(255,255,255,.12);
  color: var(--fg); outline:none; font-size:18px;
}
.field input::placeholder{ color: rgba(255,255,255,.8); }

.chip{
  position:absolute; top:-11px; left:14px; padding:4px 10px;
  border-radius:999px; font-size:12px; font-weight:600; color:var(--fg);
  background: rgba(255,255,255,.15);
  border:1px solid var(--border);
  backdrop-filter: blur(10px) saturate(160%);
}

.suffix{ position:absolute; right:14px; top:50%; transform:translateY(-50%); opacity:.85; }
.eye{ position:absolute; right:10px; top:50%; transform:translateY(-50%);
      border:0; background:transparent; font-size:18px; cursor:pointer; color:#fff; }

.row{ display:flex; align-items:center; justify-content:space-between; margin:10px 2px 16px; gap:8px; }
.remember{ display:flex; align-items:center; gap:8px; opacity:.95; }
.link{ color:#e6e9f0; text-decoration:none; }
.link:hover{ text-decoration:underline; }

.login{
  width:100%; height:60px; border:0; border-radius:20px; cursor:pointer; font-weight:800;
  background:#fff; color:#111; box-shadow:0 8px 24px rgba(0,0,0,.28);
}
.login:hover{ filter:brightness(.95); }

.cta{ text-align:center; margin:18px 0 0; opacity:.95; }
</style>
"""

# Render via HTML component so inputs/buttons are not escaped by Markdown
components.html(html, height=900, width=1600, scrolling=False)




# --------     previous version ---------




import streamlit as st

# Page config
st.set_page_config(page_title="Login Page", page_icon="🔒", layout="wide")

# Background and custom CSS
st.markdown("""
    <style>
        body {
            margin: 0;
            padding: 0;
        }
        .stApp {
            background: url("https://images.unsplash.com/photo-1506748686214-e9df14d4d9d0") no-repeat center center fixed;
            background-size: cover;
        }
        /* make the top Streamlit header transparent so wallpaper shows through */
        [data-testid="stHeader"], [data-testid="stToolbar"]{
            background: transparent !important;
            backdrop-filter: none !important;
        }
        /* ensure the main view container doesn't introduce a white background */
        [data-testid="stAppViewContainer"]{ background: transparent !important; }
        /* remove excess top padding to eliminate any visible white band */
        main .block-container{ padding-top: 0.5rem !important; }
        .glass-card {
            position: absolute;
            top: 12vh; /* move slightly down while staying centered */
            left: 50%;
            transform: translate(-50%, 0%);
            background: rgba(255, 255, 255, 0.15);
            border-radius: 20px;
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.3);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.25);
            padding: 3rem 2.5rem;
            width: 380px;
            height: auto;
            text-align: center;
        }
        .glass-card h2 {
            color: white;
            margin-bottom: 1.5rem;
        }
        .glass-card input, .glass-card button {
            width: 100%;
            padding: 0.8rem;
            margin: 0.6rem 0;
            border-radius: 12px;
            border: none;
            outline: none;
        }
        .glass-card button {
            background: rgba(255, 255, 255, 0.7);
            font-weight: bold;
            cursor: pointer;
            transition: 0.3s;
        }
        .glass-card button:hover {
            background: rgba(255, 255, 255, 0.9);
        }

        /* bottom footer links - fixed so it doesn't disturb layout */
        .app-footer{
            position: fixed; left: 50%; bottom: 16px; transform: translateX(-50%);
            display: flex; align-items: center; justify-content: space-evenly; gap: 0; flex-wrap: nowrap;
            width: clamp(480px, 70vw, 700px);
            padding: 10px 22px; border-radius: 999px;
            background: rgba(0,0,0,0.28);
            border: 2px solid rgba(255,255,255,0.25);
            backdrop-filter: blur(8px);
            -webkit-backdrop-filter: blur(8px);
            color: #fff; z-index: 1000;
        }
        .app-footer a{ color:#f3f6ff; text-decoration: none; font-weight: 600; position: relative; }
        .app-footer a:hover{ text-decoration: underline; }
        /* Use CSS-generated separators so spacing stays equal */
        .app-footer .dot{ display:none; }
        .app-footer a:not(:last-child)::after{
            content: "•"; color:#f3f6ff; opacity:.65; margin-left:12px;
        }
    </style>

    <div class="glass-card">
        <h2>Login</h2>
        <input type="text" placeholder="Username">
        <input type="password" placeholder="Password">
        <button>Login</button>
        <p style="color:white; margin-top:1rem;">Don't have an account? <a href="#" style="color:#eee;">Register</a></p>
    </div>

    <div class="app-footer">
        <a href="#about">About us</a>
        <a href="#contact">Contact us</a>
        <a href="#privacy">Privacy Policy</a>
        <a href="#terms">Terms of Service</a>
    </div>
""", unsafe_allow_html=True)
