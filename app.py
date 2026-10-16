import base64
from pathlib import Path

import streamlit as st

# -----------------------------------------------------------------------------
# 1. KONFIGURASI HALAMAN & TEMA
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="CoastalLab | Coastal Wave Analytics",
    page_icon="≈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BG, LINE, TEXT, MUTED, CYAN, AMBER, RED, GREEN = (
    "#07111F",
    "#20364D",
    "#E6F1FA",
    "#8CA5BA",
    "#42D9E8",
    "#F5B95F",
    "#FF7185",
    "#7EE787",
)

# Path aset dihitung relatif terhadap file app.py.
BASE_DIR = Path(__file__).resolve().parent
LOGO_PATH = BASE_DIR / "assets" / "coastal_lab_logo.png"


def html(s):
    """Hilangkan indentasi agar Markdown tidak menganggap HTML sebagai code block."""
    return "\n".join(line.strip() for line in s.strip().splitlines())


st.markdown(
    html(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
        html, body, [class*="css"] {{ font-family: Manrope, sans-serif; }}
        .stApp {{
            background:
                radial-gradient(ellipse at 82% -5%, #14395a 0%, transparent 48%),
                radial-gradient(ellipse at 5% 105%, #0c3b4a 0%, transparent 40%),
                {BG};
            color: {TEXT};
        }}
        [data-testid="stSidebar"], [data-testid="stSidebarNav"],
        [data-testid="collapsedControl"], header[data-testid="stHeader"] {{
            display: none;
        }}
        .block-container {{ max-width: 1320px; padding-top: 2rem; padding-bottom: 3rem; }}
        .eyebrow {{
            font-family: 'DM Mono', monospace;
            color: {CYAN}; letter-spacing: .14em; font-size: .72rem;
            text-transform: uppercase;
        }}
        .hero-title {{
            font-size: clamp(2.4rem, 4.6vw, 4.1rem);
            line-height: 1.04; letter-spacing: -.055em;
            margin: .7rem 0 1.1rem; color: {TEXT}; font-weight: 800;
        }}
        .grad {{
            background: linear-gradient(100deg, {CYAN}, #7aa8ff 60%, {AMBER});
            -webkit-background-clip: text; background-clip: text; color: transparent;
        }}
        .lead {{ max-width: 600px; color: {MUTED}; font-size: 1.02rem; line-height: 1.75; }}
        .chip {{
            display: inline-block; padding: 5px 12px; margin: 0 8px 8px 0;
            border: 1px solid {LINE}; border-radius: 999px; color: {MUTED};
            font-size: .76rem; background: rgba(16,43,66,.45);
        }}
        .stat {{
            padding: 16px 18px; border: 1px solid {LINE}; border-radius: 14px;
            background: linear-gradient(160deg, rgba(20,57,90,.35), rgba(8,21,35,.6));
        }}
        .stat b {{ display: block; font-size: 1.7rem; letter-spacing: -.03em; color: {TEXT}; }}
        .stat span {{ font-size: .76rem; color: {MUTED}; letter-spacing: .06em; }}
        .sec {{
            display: flex; align-items: center; gap: 10px; margin: 2.4rem 0 1.1rem;
            color: {MUTED}; font-size: .75rem; letter-spacing: .14em;
        }}
        .sec i {{ display: inline-block; width: 26px; height: 1px; background: {CYAN}; }}
        div[data-testid="stVerticalBlockBorderWrapper"] {{
            border: 1px solid {LINE} !important; border-radius: 18px !important;
            background: linear-gradient(165deg, rgba(18,50,78,.45), rgba(8,21,35,.75));
            transition: transform .25s, border-color .25s, box-shadow .25s;
        }}
        div[data-testid="stVerticalBlockBorderWrapper"]:hover {{
            transform: translateY(-4px); border-color: #2f7f94 !important;
            box-shadow: 0 14px 40px rgba(66,217,232,.10);
        }}
        .art {{
            border-radius: 13px; overflow: hidden; margin: 6px 0 14px;
            border: 1px solid {LINE}; background: linear-gradient(180deg,#0c2438,#071522);
        }}
        .art svg {{ display: block; width: 100%; height: auto; }}
        .drift {{ animation: drift 5s linear infinite; }}
        .drift2 {{ animation: drift 8s linear infinite reverse; opacity: .5; }}
        @keyframes drift {{ from {{ transform: translateX(0); }} to {{ transform: translateX(-60px); }} }}
        .card-h {{ font-size: 1.15rem; font-weight: 700; letter-spacing: -.02em; color: {TEXT}; margin: .2rem 0 .4rem; }}
        .card-p {{ color: {MUTED}; font-size: .88rem; line-height: 1.6; min-height: 3.1rem; }}
        .kv {{ display: flex; justify-content: space-between; font-size: .78rem; color: {MUTED}; padding: 6px 0; border-top: 1px solid {LINE}; }}
        .kv b {{ color: {TEXT}; font-weight: 600; }}
        .step {{ padding: 16px; border: 1px solid {LINE}; border-radius: 14px; background: rgba(8,21,35,.55); height: 100%; }}
        .step .n {{ font-family: 'DM Mono', monospace; color: {CYAN}; font-size: .72rem; letter-spacing: .12em; }}
        .step b {{ display: block; margin: .35rem 0 .25rem; color: {TEXT}; font-size: .95rem; }}
        .step span {{ color: {MUTED}; font-size: .8rem; line-height: 1.55; }}
        .stButton > button {{
            border-radius: 11px; border: 1px solid #2d6878;
            background: linear-gradient(120deg,#145064,#123447); color: #E8FBFF;
            font-weight: 700; min-height: 2.7rem; transition: .2s;
        }}
        .stButton > button:hover {{
            border-color: {CYAN}; background: linear-gradient(120deg,#1a6a83,#16465f); color: #fff;
        }}
        hr {{ border-color: {LINE}; }}
        </style>
        """
    ),
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# 2. DATA & KOMPONEN
# -----------------------------------------------------------------------------
PAGES = {
    "open": "pages/open_ocean_beach.py",
    "pocket": "pages/pocket_beach.py",
    "reef": "pages/reef_influenced_beach.py",
}


def wave_path(y, amp, wl=60, n=14):
    d = f"M0,{y}"
    for _ in range(n):
        d += f" q{wl / 4},{-amp} {wl / 2},0 t{wl / 2},0"
    return d


def art(profile, color):
    """Ilustrasi penampang kecil per tipe pantai (SVG, animasi gelombang via CSS)."""
    seabed = {
        "open": "M0,34 L300,84 L300,100 L0,100Z",
        "pocket": "M0,30 C90,36 150,52 210,66 C250,75 280,80 300,82 L300,100 L0,100Z",
        "reef": "M0,30 L70,33 L105,60 L215,60 L250,72 L300,84 L300,100 L0,100Z",
    }[profile]

    extra = ""
    if profile == "pocket":
        extra = (
        )
    if profile == "reef":
        extra = (
            f'<text x="150" y="53" fill="{GREEN}" font-size="8.5" '
            f'font-family="DM Mono" text-anchor="middle"></text>'
        )

    return html(
        f"""
        <div class="art">
          <svg viewBox="0 0 300 100" preserveAspectRatio="xMidYMid slice">
            <defs>
              <linearGradient id="w{profile}" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0" stop-color="{color}" stop-opacity=".35"/>
                <stop offset="1" stop-color="{color}" stop-opacity=".05"/>
              </linearGradient>
            </defs>
            <rect width="300" height="100" fill="url(#w{profile})"/>
            <g class="drift2"><path d="{wave_path(26, 5)}" stroke="{color}" stroke-width="1.5" fill="none"/></g>
            <g class="drift"><path d="{wave_path(22, 7)}" stroke="{color}" stroke-width="2" fill="none"/></g>
            <path d="{seabed}" fill="#C9A26A" fill-opacity=".85"/>{extra}
          </svg>
        </div>
        """
    )


def logo(size=46):
    """Tampilkan logo PNG; gunakan simbol gelombang jika file logo belum tersedia."""
    if LOGO_PATH.is_file():
        logo_base64 = base64.b64encode(LOGO_PATH.read_bytes()).decode("ascii")
        logo_markup = (
            f'<img src="data:image/png;base64,{logo_base64}" '
            f'alt="Logo CoastalLab" '
            f'style="width:{size}px;height:{size}px;object-fit:contain;flex-shrink:0;">'
        )
    else:
        logo_markup = (
            f'<div style="background:linear-gradient(145deg,{CYAN},#1A7D9C);'
            f'width:{size}px;height:{size}px;border-radius:14px;display:grid;'
            f'place-items:center;font-size:{size // 2}px;color:#07111F;font-weight:800;'
            f'flex-shrink:0;">≈</div>'
        )

    return html(
        f"""
        <div style="display:flex;align-items:center;gap:12px;min-width:0;">
            {logo_markup}
            <div style="line-height:1.35;">
                <b style="font-size:1.1rem;color:{TEXT};white-space:nowrap;">CoastalLab</b>
                <div style="font-size:.68rem;color:{MUTED};letter-spacing:.12em;white-space:nowrap;">
                    COASTAL WAVE ANALYTICS
                </div>
            </div>
        </div>
        """
    )


def go_to(key):
    st.switch_page(PAGES[key])


# -----------------------------------------------------------------------------
# 3. HERO
# -----------------------------------------------------------------------------
st.markdown(
    html(
        f"""
        <div style="display:flex;align-items:center;margin-bottom:2.6rem;">
            {logo()}
            <div style="margin-left:auto;color:{MUTED};font-size:.74rem;letter-spacing:.12em;">
                WORKSPACE / 01
            </div>
        </div>
        """
    ),
    unsafe_allow_html=True,
)

hl, hr_ = st.columns([1.25, 1], gap="large")

with hl:
    st.markdown(
        html(
            f"""
            <div class="eyebrow">COASTAL ENVIRONMENTS · SELECT A PROFILE</div>
            <div class="hero-title">Perjalanan gelombang <span class="grad">dari laut lepas hingga pantai.</span></div>
            <p class="lead">Simulasi interaktif shoaling, refraksi, breaking, dan longshore forcing untuk tiga tipe pantai. Ubah parameter gelombang, bandingkan skenario, lalu ekspor hasilnya.</p>
            <div style="margin-top:1.2rem">
                <span class="chip">Dispersi linier</span>
                <span class="chip">Snell · refraksi</span>
                <span class="chip">Kriteria breaking H ≤ γh</span>
                <span class="chip">Radiation stress</span>
                <span class="chip">Longuet-Higgins</span>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )

with hr_:
    st.markdown(
        html(
            f"""
            <div class="art" style="border-radius:20px;margin-top:6px">
              <svg viewBox="0 0 400 250" preserveAspectRatio="xMidYMid slice">
                <defs>
                  <linearGradient id="hw" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0" stop-color="{CYAN}" stop-opacity=".45"/>
                    <stop offset="1" stop-color="{CYAN}" stop-opacity=".03"/>
                  </linearGradient>
                </defs>
                <rect width="400" height="250" fill="url(#hw)"/>
                <g class="drift2"><path d="{wave_path(95, 9, 80, 8)}" stroke="{CYAN}" stroke-width="1.5" fill="none"/></g>
                <g class="drift"><path d="{wave_path(80, 14, 80, 8)}" stroke="{CYAN}" stroke-width="2.5" fill="none"/></g>
                <g class="drift2"><path d="{wave_path(118, 6, 80, 8)}" stroke="#7aa8ff" stroke-width="1.2" fill="none"/></g>
                <path d="M0,90 L120,100 L170,170 L300,175 L400,225 L400,250 L0,250Z" fill="#C9A26A" fill-opacity=".85"/>
                <line x1="0" y1="80" x2="400" y2="80" stroke="{MUTED}" stroke-dasharray="3 5" stroke-width="1"/>
                <text x="16" y="72" fill="{MUTED}" font-size="10" font-family="DM Mono"> </text>
                <text x="236" y="165" fill="#07111F" font-size="10" font-family="DM Mono" font-weight="500"> </text>
              </svg>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )

s1, s2, s3, s4 = st.columns(4)
for c, (n, label) in zip(
    (s1, s2, s3, s4),
    [
        ("3", "PROFIL PANTAI"),
        ("9", "SKENARIO SIMULASI"),
        ("6–7", "TAB ANALISIS PER PROFIL"),
        ("1 m", "RESOLUSI GRID"),
    ],
):
    c.markdown(
        f'<div class="stat"><b>{n}</b><span>{label}</span></div>',
        unsafe_allow_html=True,
    )

# -----------------------------------------------------------------------------
# 4. KARTU PROFIL
# -----------------------------------------------------------------------------
st.markdown(
    '<div class="sec"><i></i>SELECT COASTAL PROFILE</div>',
    unsafe_allow_html=True,
)

CARDS = [
    (
        "open",
        "KASUS A",
        "open_ocean_beach",
        "Pantai terbuka dengan paparan langsung terhadap gelombang laut lepas.",
        CYAN,
        [
            ("Skenario", "A1 · A2 · A3"),
            ("Variabel diuji", "H₀ dan slope"),
            ("Fokus", "Shoaling & breaking"),
        ],
    ),
    (
        "pocket",
        "KASUS B",
        "pocket_beach",
        "Pantai kecil yang diapit tanjung; fokus pada refraksi dan longshore forcing.",
        AMBER,
        [
            ("Skenario", "B1 · B2 · B3"),
            ("Variabel diuji", "θ₀ dan slope"),
            ("Fokus", "Refraksi & longshore"),
        ],
    ),
    (
        "reef",
        "KASUS C",
        "reef_influenced_beach",
        "Pantai di belakang terumbu; reef flat mengubah energi dan lokasi gelombang pecah.",
        GREEN,
        [
            ("Skenario", "C1 · C2 · C3"),
            ("Variabel diuji", "Kedalaman reef"),
            ("Fokus", "Disipasi energi"),
        ],
    ),
]

cols = st.columns(3, gap="large")
for i, (col, (key, case, label, desc, color, kv)) in enumerate(zip(cols, CARDS), 1):
    with col, st.container(border=True):
        st.markdown(
            html(
                f"""
                <div class="eyebrow" style="color:{color}">{case} · PROFILE / 0{i}</div>
                {art(key, color)}
                <div class="card-h">{label}</div>
                <div class="card-p">{desc}</div>
                {''.join(f'<div class="kv"><span>{k}</span><b>{v}</b></div>' for k, v in kv)}
                <div style="height:12px"></div>
                """
            ),
            unsafe_allow_html=True,
        )
        st.button(
            "Open workspace  ↗",
            key=f"open_{key}",
            use_container_width=True,
            on_click=go_to,
            args=(key,),
        )

# -----------------------------------------------------------------------------
# 5. ALUR METODE
# -----------------------------------------------------------------------------
st.markdown('<div class="sec"><i></i>ALUR PEMODELAN</div>', unsafe_allow_html=True)

steps = [
    (
        "01",
        "Dispersi",
        "Panjang gelombang L, celerity C, dan Cg dari relasi dispersi linier di tiap kedalaman.",
    ),
    (
        "02",
        "Shoaling & refraksi",
        "Hukum Snell untuk θ(x), kekekalan fluks energi untuk H(x).",
    ),
    (
        "03",
        "Breaking",
        "Gelombang pecah saat H/h mencapai γ; H dibatasi H = γh di surf zone.",
    ),
    (
        "04",
        "Longshore forcing",
        "Radiation stress Sxy, indikator Pl, dan estimasi arus longshore V.",
    ),
]

for c, (n, title, description) in zip(st.columns(4, gap="medium"), steps):
    c.markdown(
        f'<div class="step"><div class="n">STEP {n}</div><b>{title}</b><span>{description}</span></div>',
        unsafe_allow_html=True,
    )

# -----------------------------------------------------------------------------
# 6. FOOTER
# -----------------------------------------------------------------------------
st.markdown(
    html(
        f"""
        <div style="margin-top:2.4rem;padding-top:1.1rem;border-top:1px solid {LINE};
                    display:flex;flex-wrap:wrap;gap:2rem;justify-content:space-between;
                    color:{MUTED};font-size:.82rem;line-height:1.7;">
            <div>
                <b style="color:{TEXT}">Team Development</b><br>
                Pandu Malik Aulia · 12923075<br>
                Ikhwan Tsuni Nusantara · 12923098<br>
                Raditya Ramadhan · 12923029<br>
                Ardika Fauzianto · 12923100
            </div>
            <div style="max-width:420px;text-align:right">
                <b style="color:{TEXT}">Catatan model</b><br>
                Model linier 1D cross-shore. Tidak memuat wave setup penuh, gesekan dasar,
                atau arus balik; arus cross-shore pada sketsa bersifat skematik.
            </div>
        </div>
        """
    ),
    unsafe_allow_html=True,
)
