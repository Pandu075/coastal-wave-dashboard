import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# -----------------------------------------------------------------------------
# 0. SETUP & TEMA (mengikuti app.py)
# -----------------------------------------------------------------------------
st.set_page_config(page_title='Pocket Beach - Case B', page_icon='⌁', layout='wide', initial_sidebar_state='expanded')

BG, TEXT, MUTED, LINE = '#07111F', '#E6F1FA', '#8CA5BA', '#20364D'
CYAN, AMBER, RED, GREEN = '#42D9E8', '#F5B95F', '#FF7185', '#7EE787'
SCEN_COLORS = {'B1': CYAN, 'B2': AMBER, 'B3': GREEN}

st.markdown(f'''<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
html,body,[class*="css"]{{font-family:Manrope,sans-serif}}
.stApp{{background:radial-gradient(ellipse at 78% 0%,#102b42 0%,{BG} 44%);color:{TEXT}}}
[data-testid="stSidebar"]{{background:#081523;border-right:1px solid {LINE}}}
[data-testid="stSidebarNav"]{{display:none}}
.block-container{{max-width:1600px;padding-top:1.4rem;padding-bottom:2rem}}
.stButton>button,.stDownloadButton>button{{border-radius:10px;border:1px solid #2d6878;background:linear-gradient(120deg,#145064,#123447);color:#E8FBFF;font-weight:700}}
.stTabs [data-baseweb="tab"]{{padding:8px 14px;font-weight:600}}
.eyebrow{{font-family:'DM Mono',monospace;color:{CYAN};letter-spacing:.13em;font-size:.72rem}}
.note{{color:{MUTED};font-size:.88rem;margin:.1rem 0 .6rem}}
[data-testid="stMetricValue"]{{font-size:1.1rem}}
[data-testid="stMetricLabel"]{{font-size:.72rem}}
</style>''', unsafe_allow_html=True)

BASE = dict(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Manrope,sans-serif', color=TEXT, size=12), margin=dict(l=10, r=10, t=40, b=10),
            hoverlabel=dict(bgcolor='#102338', bordercolor='#2B536D', font_color=TEXT),
            legend=dict(orientation='h', y=1.08, x=0))


def style(fig, height, hover='x unified'):
    fig.update_layout(**BASE, height=height, hovermode=hover)
    fig.update_xaxes(gridcolor=LINE, zerolinecolor=LINE)
    fig.update_yaxes(gridcolor=LINE, zerolinecolor=LINE)
    return fig


# -----------------------------------------------------------------------------
# 1. MODEL (di-cache per kombinasi parameter)
# -----------------------------------------------------------------------------
RHO, G = 1025.0, 9.81
X_B12 = np.array([0, 50, 100, 150, 200, 300, 400, 500, 750, 1000, 1100], dtype=float)
H_B12 = np.array([0, 0.8, 1.7, 2.6, 3.6, 5.5, 7.4, 9.2, 14.0, 20.0, 22.0])
X_B3 = np.array([0, 50, 100, 150, 200, 300, 400, 500, 750, 1000, 1500, 2000, 2500, 3143], dtype=float)
H_B3 = np.array([0, 0.35, 0.70, 1.05, 1.40, 2.10, 2.80, 3.50, 5.25, 7.00, 10.50, 14.00, 17.50, 22.00])
DEPTHS = [10, 5, 2]


def sm(a, w=9):
    return np.convolve(np.nan_to_num(a), np.ones(w) / w, mode='same')


@st.cache_data
def simulate_all(H0, T, gamma, th_base, th_obl):
    omega = 2 * np.pi / T
    profiles = {'B1 Baseline': (X_B12, H_B12, th_base), 'B2 Oblique Wave': (X_B12, H_B12, th_obl),
                'B3 Flatter Beach': (X_B3, H_B3, th_base)}

    def dispersion(h):
        hs = np.maximum(h, 1e-3)
        k = omega ** 2 / G + omega / np.sqrt(G * hs)
        for _ in range(60):  # Newton, mulai dari sisi kanan akar
            t = np.tanh(k * hs)
            k = k - (G * k * t - omega ** 2) / (G * t + G * k * hs / np.cosh(k * hs) ** 2)
        resid = np.abs(omega ** 2 - G * k * np.tanh(k * hs)) / omega ** 2
        k = np.where(h > 0, k, np.nan)
        C = omega / k
        kh = k * h
        n = .5 * (1 + 2 * kh / np.sinh(2 * kh))
        return k, 2 * np.pi / k, C, n, n * C, resid[h > 0].max()

    out = {}
    for name, (xs, hs, th0d) in profiles.items():
        x = np.arange(0, xs[-1] + .5, 1.0)
        h = np.interp(x, xs, hs)
        k, L, C, n, Cg, resid = dispersion(h)
        th0 = np.deg2rad(th0d)
        th = np.arcsin(np.clip(C / C[-1] * np.sin(th0), -1, 1))
        A = Cg * np.cos(th)
        Hpre = H0 * np.sqrt(A[-1] / A)  # shoaling + refraksi tanpa breaking
        cap = gamma * h
        N = len(x)
        H = np.zeros(N); F = np.zeros(N); D = np.zeros(N); iscap = np.zeros(N, bool)
        H[-1] = H0
        F[-1] = RHO * G / 8 * H0 ** 2 * A[-1]
        for i in range(N - 2, 0, -1):
            Ht = np.sqrt(max(0., 8 * F[i + 1] / (RHO * G * A[i])))
            H[i] = min(Ht, cap[i])
            iscap[i] = Ht > cap[i] + 1e-9
            F[i] = RHO * G / 8 * H[i] ** 2 * A[i]
            D[i] = max(0., F[i + 1] - F[i])
        E = RHO * G * H ** 2 / 8
        Sxx = E * (n * (1 + np.cos(th) ** 2) - .5)
        Sxy = -E * n * np.sin(th) * np.cos(th)
        Fx = np.full(N, np.nan); Fy = np.full(N, np.nan)
        Fx[1:] = -np.gradient(Sxx[1:], x[1:], edge_order=2)
        Fy[1:] = -np.gradient(Sxy[1:], x[1:], edge_order=2)
        eta = np.full(N, np.nan); eta[-1] = 0.0
        for j in range(N - 2, 0, -1):
            if h[j] < .05: break
            eta[j] = eta[j + 1] - (Sxx[j] - Sxx[j + 1]) / (RHO * G * .5 * (h[j] + h[j + 1]))
        Pl = E * Cg * np.sin(th) * np.cos(th)

        brk = None
        caps = np.where(iscap)[0]
        if len(caps):
            i0 = caps.max(); i1 = i0 + 1
            f0, f1 = Hpre[i0] - cap[i0], Hpre[i1] - cap[i1]
            xb = x[i1] + (-f1) / (f0 - f1) * (x[i0] - x[i1])
            hb = float(np.interp(xb, x, h)); Hb = gamma * hb
            thb = float(np.interp(xb, x, np.rad2deg(th)))
            Cgb = float(np.interp(xb, x, Cg))
            Plb = RHO * G * Hb ** 2 / 8 * Cgb * np.sin(np.deg2rad(thb)) * np.cos(np.deg2rad(thb))
            m = float(np.interp(xb, x, np.gradient(h, x)))
            V = 20.7 * m * np.sqrt(G * Hb) * np.sin(np.deg2rad(thb)) * np.cos(np.deg2rad(thb))
            brk = dict(xb=float(xb), hb=hb, Hb=Hb, thb=thb, Plb=Plb, Sxyb=float(np.interp(xb, x, Sxy)), V=V, m=m)
        out[name] = dict(x=x, h=h, k=k, L=L, C=C, Cg=Cg, n=n, theta=np.rad2deg(th), Hpre=Hpre, cap=cap, H=H, F=F,
                         D=D, Dsm=sm(D), E=E, Sxx=Sxx, Sxy=Sxy, Fx=Fx, Fy=sm(Fy, 11), eta=eta, Pl=Pl, brk=brk,
                         th0=th0d, resid=float(resid), L0=float(L[-1]), C0=float(C[-1]), Cg0=float(Cg[-1]))
    return out


def at_depth(r, key, d):
    return float(np.interp(d, r['h'][1:], np.asarray(r[key])[1:]))


# -----------------------------------------------------------------------------
# 2. SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown(f'<div style="display:flex;align-items:center;gap:12px;margin-bottom:14px"><div style="background:linear-gradient(145deg,{CYAN},#1A7D9C);width:42px;height:42px;border-radius:13px;display:grid;place-items:center;font-size:23px;color:#07111F">⌁</div><div><b style="font-size:1.05rem;color:{TEXT}">CoastalLab</b><div style="font-size:.7rem;color:{MUTED};letter-spacing:.08em">POCKET BEACH</div></div></div>', unsafe_allow_html=True)
    st.button('← Kembali ke Beranda', on_click=lambda: st.switch_page('app.py'), use_container_width=True)
    st.markdown('---')
    st.markdown('<div class="eyebrow">PARAMETER GELOMBANG</div>', unsafe_allow_html=True)
    H0 = st.slider('H₀ offshore (m)', 0.5, 4.0, 1.8, 0.1)
    T = st.slider('Periode T (s)', 4.0, 16.0, 9.0, 0.5)
    th_base = st.slider('θ₀ B1 & B3 (°)', 0, 60, 20)
    th_obl = st.slider('θ₀ B2 oblique (°)', 0, 60, 35)
    gamma = st.slider('Breaker index γ', 0.6, 1.0, 0.78, 0.01)
    st.markdown('<div class="eyebrow" style="margin-top:14px">TAMPILAN</div>', unsafe_allow_html=True)
    results = simulate_all(H0, T, gamma, th_base, th_obl)
    sel = st.multiselect('Skenario', list(results.keys()), default=list(results.keys()))
    xr = st.slider('Zoom x (m)', 0, 3143, (0, 1200), step=50)

if not sel:
    st.warning('Pilih minimal satu skenario di sidebar.')
    st.stop()
R = {n: results[n] for n in sel}
col_of = lambda n: SCEN_COLORS[n[:2]]


def mark_breakers(fig, rows=1):
    for n, r in R.items():
        if r['brk']:
            for rr in range(1, rows + 1):
                fig.add_vline(x=r['brk']['xb'], line_color=col_of(n), line_dash='dot', line_width=1.2, opacity=.7, row=rr, col=1)


# -----------------------------------------------------------------------------
# 3. HEADER + KPI
# -----------------------------------------------------------------------------
st.markdown(f'<div class="eyebrow">KASUS B / POCKET BEACH · REFRACTION CHALLENGE</div><h2 style="margin:.2rem 0 .6rem;letter-spacing:-.03em;color:{TEXT}">Refraksi dan Longshore Forcing di Pocket Beach</h2>', unsafe_allow_html=True)
kc = st.columns(len(R))
for c, (n, r) in zip(kc, R.items()):
    b = r['brk']
    with c, st.container(border=True):
        st.markdown(f'<b style="color:{col_of(n)}">{n}</b> <span style="color:{MUTED};font-size:.8rem">· θ₀ {r["th0"]:.0f}°</span>', unsafe_allow_html=True)
        a = st.columns(4)
        a[0].metric('xb', f'{b["xb"]:.0f} m' if b else '–')
        a[1].metric('Hb', f'{b["Hb"]:.2f} m' if b else '–')
        a[2].metric('θb', f'{b["thb"]:.1f}°' if b else '–')
        a[3].metric('Pl,b', f'{b["Plb"] / 1000:.1f} kW/m' if b else '–')

tabs = st.tabs(['① Batimetri & Offshore', '② Transformasi', '③ Energi & Setup', '④ Longshore Forcing', '⑤ Perbandingan & Q1–Q4', '⑥ Sketsa Penampang', '⑦ Data'])

# ---------------------------- ① BATIMETRI & OFFSHORE ------------------------
with tabs[0]:
    left, right = st.columns([3, 2])
    fig = go.Figure()
    for n, r in R.items():
        src = (X_B3, H_B3) if n.startswith('B3') else (X_B12, H_B12)
        fig.add_trace(go.Scatter(x=r['x'], y=-r['h'], name=n, line=dict(color=col_of(n), width=2.5, dash='dot' if n.startswith('B2') else 'solid')))
        fig.add_trace(go.Scatter(x=src[0], y=-src[1], mode='markers', showlegend=False, marker=dict(color=col_of(n), size=6), hoverinfo='skip'))
    style(fig, 400).update_layout(xaxis_title='x (m), positif ke offshore', yaxis_title='Elevasi dasar (m)')
    fig.update_xaxes(range=list(xr))
    left.plotly_chart(fig, use_container_width=True)
    right.dataframe(pd.DataFrame([{'Skenario': n, 'θ₀ (°)': r['th0'], 'h₀ (m)': r['h'][-1], 'L₀ (m)': round(r['L0'], 1), 'C₀ (m/s)': round(r['C0'], 2),
                                   'Cg₀ (m/s)': round(r['Cg0'], 2), 'Domain x (m)': r['x'][-1]} for n, r in R.items()]), hide_index=True, use_container_width=True)
    right.caption('Titik = data Lampiran A.3. B1 dan B2 memakai profil sama (domain 1100 m); B3 lebih landai (domain 3143 m). Kemiringan dihitung dari tabel A.3, bukan dari label 1:50 / 1:70 pada tabel 5.3.')

# ---------------------------- ② TRANSFORMASI --------------------------------
with tabs[1]:
    o1, o2, o3 = st.columns([2, 1, 1])
    mode = o1.radio('Sumbu horizontal', ['Jarak x (m)', 'Kedalaman h (m)'], horizontal=True)
    show_lin = o2.toggle('H linear (tanpa breaking)', True)
    show_cap = o3.toggle('Batas γh', True)
    byx = mode.startswith('Jarak')
    rows = 3 if byx else 2
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                        row_heights=[.2, .5, .3] if byx else [.6, .4],
                        subplot_titles=(['Batimetri (m)'] if byx else []) + ['Tinggi gelombang H (m)', 'Sudut gelombang θ (°)'])
    rH, rT = (2, 3) if byx else (1, 2)
    for n, r in R.items():
        c = col_of(n)
        sl = slice(1, None)
        xv = r['x'][sl] if byx else r['h'][sl]
        if byx: fig.add_trace(go.Scatter(x=xv, y=-r['h'][sl], name=n, legendgroup=n, line=dict(color=c, width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=xv, y=r['H'][sl], name=f'H {n[:2]}', legendgroup=n, showlegend=not byx, line=dict(color=c, width=3)), row=rH, col=1)
        if show_lin: fig.add_trace(go.Scatter(x=xv, y=r['Hpre'][sl], legendgroup=n, showlegend=False, name=f'H linear {n[:2]}', line=dict(color=c, width=1.5, dash='dot')), row=rH, col=1)
        if show_cap: fig.add_trace(go.Scatter(x=xv, y=r['cap'][sl], legendgroup=n, showlegend=False, name=f'γh {n[:2]}', line=dict(color=c, width=1, dash='dash'), opacity=.55), row=rH, col=1)
        fig.add_trace(go.Scatter(x=xv, y=r['theta'][sl], legendgroup=n, showlegend=False, name=f'θ {n[:2]}', line=dict(color=c, width=2.5)), row=rT, col=1)
        if r['brk']:
            b = r['brk']
            fig.add_trace(go.Scatter(x=[b['xb'] if byx else b['hb']], y=[b['Hb']], mode='markers', legendgroup=n, showlegend=False,
                                     marker=dict(color='white', size=10, line=dict(color=c, width=2)),
                                     hovertemplate=f'Breaking {n[:2]}: xb={b["xb"]:.0f} m, hb={b["hb"]:.2f} m, Hb={b["Hb"]:.2f} m<extra></extra>'), row=rH, col=1)
    fig.update_yaxes(range=[0, max(H0 * 1.6, 2.5)], row=rH, col=1)
    style(fig, 720 if byx else 600)
    if byx:
        fig.update_xaxes(range=list(xr)); fig.update_xaxes(title_text='x (m), positif ke offshore', row=rows, col=1)
    else:
        fig.update_xaxes(range=[22, 0]); fig.update_xaxes(title_text='Kedalaman h (m) — offshore ke kiri, pantai ke kanan', row=rows, col=1)
    st.plotly_chart(fig, use_container_width=True)
    st.caption('Titik putih = breaking pertama dari offshore. Garis titik-titik = H tanpa pembatas breaking; garis putus-putus = batas γh.')

# ---------------------------- ③ ENERGI & SETUP ------------------------------
with tabs[2]:
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                        subplot_titles=['Fluks energi normal-pantai F = E·Cg·cosθ (W/m)', 'Disipasi energi D (W/m²)', 'Setup/setdown linear η (m) — diagnostik'])
    for n, r in R.items():
        c = col_of(n)
        fig.add_trace(go.Scatter(x=r['x'][1:], y=r['F'][1:], name=n, legendgroup=n, line=dict(color=c, width=2.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=r['x'][1:], y=r['Dsm'][1:], showlegend=False, legendgroup=n, name=f'D {n[:2]}', line=dict(color=c, width=2.5), fill='tozeroy', fillcolor='rgba(255,255,255,.03)'), row=2, col=1)
        fig.add_trace(go.Scatter(x=r['x'][1:], y=r['eta'][1:], showlegend=False, legendgroup=n, name=f'η {n[:2]}', line=dict(color=c, width=2.5)), row=3, col=1)
    mark_breakers(fig, 3)
    style(fig, 700)
    fig.update_xaxes(range=list(xr)); fig.update_xaxes(title_text='x (m), positif ke offshore', row=3, col=1)
    st.plotly_chart(fig, use_container_width=True)
    st.caption('Garis titik-titik vertikal = lokasi breaking. Puncak D menunjukkan di mana energi hilang paling kuat. Setup η hanya diagnostik linear (tanpa gesekan dasar dan arus balik).')

# ---------------------------- ④ LONGSHORE FORCING ---------------------------
with tabs[3]:
    left, right = st.columns([3, 2])
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                        subplot_titles=['Indikator Pl ∝ E·Cg·sinθ·cosθ (W/m)', 'Radiation stress silang Sxy (N/m)', 'Gaya longshore Fy = −dSxy/dx (N/m²)'])
    for n, r in R.items():
        c = col_of(n)
        fig.add_trace(go.Scatter(x=r['x'][1:], y=r['Pl'][1:], name=n, legendgroup=n, line=dict(color=c, width=2.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=r['x'][1:], y=r['Sxy'][1:], showlegend=False, legendgroup=n, name=f'Sxy {n[:2]}', line=dict(color=c, width=2.5)), row=2, col=1)
        fig.add_trace(go.Scatter(x=r['x'][1:], y=r['Fy'][1:], showlegend=False, legendgroup=n, name=f'Fy {n[:2]}', line=dict(color=c, width=2.5)), row=3, col=1)
    mark_breakers(fig, 3)
    style(fig, 700)
    fig.update_xaxes(range=list(xr)); fig.update_xaxes(title_text='x (m), positif ke offshore', row=3, col=1)
    left.plotly_chart(fig, use_container_width=True)

    bn = [n for n in R if R[n]['brk']]
    fig2 = make_subplots(rows=2, cols=1, vertical_spacing=0.18, subplot_titles=['Pl di titik breaking (kW/m)', 'Arus longshore V (m/s), Longuet-Higgins'])
    fig2.add_trace(go.Bar(x=[n[:2] for n in bn], y=[R[n]['brk']['Plb'] / 1000 for n in bn], marker_color=[col_of(n) for n in bn],
                          text=[f'{R[n]["brk"]["Plb"] / 1000:.1f}' for n in bn], textposition='outside'), row=1, col=1)
    fig2.add_trace(go.Bar(x=[n[:2] for n in bn], y=[R[n]['brk']['V'] for n in bn], marker_color=[col_of(n) for n in bn],
                          text=[f'{R[n]["brk"]["V"]:.2f}' for n in bn], textposition='outside'), row=2, col=1)
    style(fig2, 700, 'closest').update_layout(showlegend=False)
    right.plotly_chart(fig2, use_container_width=True)
    st.caption('V ≈ 20,7·m·√(g·Hb)·sinθb·cosθb, m = kemiringan dasar di titik breaking. Pl sendiri adalah indikator fluks energi longshore, bukan arus.')

# ---------------------------- ⑤ PERBANDINGAN & Q1–Q4 ------------------------
with tabs[4]:
    left, right = st.columns(2)
    fig = go.Figure()
    for n, r in R.items():
        fig.add_trace(go.Bar(x=[f'h={d} m' for d in DEPTHS], y=[at_depth(r, 'H', d) for d in DEPTHS], name=n, marker_color=col_of(n),
                             text=[f'{at_depth(r, "H", d):.2f}' for d in DEPTHS], textposition='outside'))
    style(fig, 360, 'closest').update_layout(barmode='group', title='H pada kedalaman sama (m)')
    left.plotly_chart(fig, use_container_width=True)
    fig = go.Figure()
    for n, r in R.items():
        fig.add_trace(go.Bar(x=[f'h={d} m' for d in DEPTHS], y=[at_depth(r, 'theta', d) for d in DEPTHS], name=n, marker_color=col_of(n),
                             text=[f'{at_depth(r, "theta", d):.1f}' for d in DEPTHS], textposition='outside'))
    style(fig, 360, 'closest').update_layout(barmode='group', title='Sudut θ pada kedalaman sama (°)')
    right.plotly_chart(fig, use_container_width=True)

    rows = []
    for n, r in R.items():
        b = r['brk']
        rows.append({'Skenario': n, 'θ₀ (°)': r['th0'], 'xb (m)': round(b['xb']) if b else None, 'hb (m)': round(b['hb'], 2) if b else None,
                     'Hb (m)': round(b['Hb'], 2) if b else None, 'θb (°)': round(b['thb'], 1) if b else None,
                     'Pl,b (kW/m)': round(b['Plb'] / 1000, 2) if b else None, 'V (m/s)': round(b['V'], 2) if b else None,
                     **{f'H@{d}m': round(at_depth(r, 'H', d), 2) for d in DEPTHS}})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    if all(k in results and results[k]['brk'] for k in ('B1 Baseline', 'B2 Oblique Wave', 'B3 Flatter Beach')):
        b1, b2, b3 = (results[k]['brk'] for k in ('B1 Baseline', 'B2 Oblique Wave', 'B3 Flatter Beach'))
        s1, s2, s3 = (results[k] for k in ('B1 Baseline', 'B2 Oblique Wave', 'B3 Flatter Beach'))
        with st.container(border=True):
            st.markdown('**Ringkasan hasil untuk pertanyaan 5.5** (angka dari model, interpretasi tetap perlu ditulis sendiri)')
            st.markdown(f"**Q1 — Pengaruh θ₀ terhadap θ di breaker:** θ₀ {s1['th0']:.0f}° → {s2['th0']:.0f}° (B1 → B2) mengubah θb dari {b1['thb']:.1f}° menjadi {b2['thb']:.1f}°, "
                        f"dan Hb dari {b1['Hb']:.2f} m menjadi {b2['Hb']:.2f} m. Refraksi memutar crest sehingga θb jauh lebih kecil dari θ₀.")
            ratio = b2['Plb'] / b1['Plb'] if b1['Plb'] else np.nan
            verdict = 'lebih besar' if ratio > 1 else 'tidak lebih besar'
            st.markdown(f"**Q2 — Sudut lebih besar = forcing lebih besar?** Pl,b B2 = {b2['Plb'] / 1000:.1f} kW/m vs B1 = {b1['Plb'] / 1000:.1f} kW/m (rasio {ratio:.2f}), jadi forcing B2 {verdict} pada kasus ini. "
                        f"Hubungannya tidak linear karena sinθ·cosθ maksimum di 45° dan Hb ikut berubah. V longshore: B1 {b1['V']:.2f} m/s, B2 {b2['V']:.2f} m/s.")
            st.markdown(f"**Q3 — Slope lebih landai (B3 vs B1):** breaking bergeser dari xb = {b1['xb']:.0f} m ke {b3['xb']:.0f} m (hb {b1['hb']:.2f} → {b3['hb']:.2f} m), "
                        f"θb {b1['thb']:.1f}° → {b3['thb']:.1f}°, Pl,b {b1['Plb'] / 1000:.1f} → {b3['Plb'] / 1000:.1f} kW/m. Pada kedalaman yang sama H dan θ identik (Snell dan shoaling hanya bergantung pada h), jadi slope mengubah lokasi dan lebar surf zone, bukan nilai θb dan Hb. Gaya longshore Fy tersebar pada jarak lebih panjang, sehingga puncaknya lebih kecil.")
            st.markdown('**Q4 — Banding dengan Kasus A:** butuh hasil Kasus A dari halaman Open Ocean Beach. Bandingkan xb, Hb, θb, dan Pl,b dengan tabel di atas.')
    else:
        st.info('Aktifkan ketiga skenario B1, B2, B3 di sidebar untuk menampilkan ringkasan Q1–Q3.')


# ---------------------------- ⑥ SKETSA PENAMPANG ----------------------------
with tabs[5]:
    c1, c2 = st.columns([1, 2])

    sc = c1.selectbox('Skenario', list(R.keys()))
    r = R[sc]
    b = r['brk']
    col = col_of(sc)
    th0 = float(r['th0'])

    xmax = int(r['x'][-1])
    lo, hi = c2.slider(
        'Rentang x ditampilkan (m)',
        0, xmax,
        (0, min(xmax, 600)),
        step=50,
        key=f'sk_{sc}'
    )

    # Jaga rentang tetap valid
    hi = min(max(hi, lo + 100), xmax)
    span = max(hi - lo, 1)

    m = (r['x'] >= lo) & (r['x'] <= hi)
    xs = r['x'][m]
    hh = r['h'][m]
    Hh = r['H'][m]

    if xs.size == 0:
        idx = int(np.argmin(np.abs(r['x'] - lo)))
        xs = r['x'][idx:idx + 1]
        hh = r['h'][idx:idx + 1]
        Hh = r['H'][idx:idx + 1]

    ymax = float(np.nanmax(hh)) * 1.1 + 1
    ytop = max(float(np.nanmax(Hh)) / 2, 1) + 2.2

    left, right = st.columns([2, 1])

    # Status breaking
    breaking_visible = (
        b is not None
        and lo <= float(b['xb']) <= hi
    )

    # ==========================================================
    # A. PENAMPANG CROSS-SHORE
    # ==========================================================
    fig = go.Figure()

    # Dasar pengisi area laut
    fig.add_trace(go.Scatter(
        x=xs,
        y=np.full(xs.size, -ymax),
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))

    # Dasar laut
    fig.add_trace(go.Scatter(
        x=xs,
        y=-hh,
        name='Dasar laut',
        fill='tonexty',
        fillcolor='rgba(214,176,120,0.55)',
        line=dict(color='#D6B078', width=2),
        hovertemplate=(
            'x = %{x:.0f} m'
            '<br>Dasar = %{y:.2f} m'
            '<extra></extra>'
        )
    ))

    # Muka air diam
    fig.add_trace(go.Scatter(
        x=xs,
        y=np.zeros(xs.size),
        name='Muka air diam',
        fill='tonexty',
        fillcolor='rgba(66,217,232,0.10)',
        line=dict(color=MUTED, width=1, dash='dot'),
        hovertemplate='x = %{x:.0f} m<br>z = 0 m<extra></extra>'
    ))

    # Batas bawah envelope gelombang
    fig.add_trace(go.Scatter(
        x=xs,
        y=-Hh / 2,
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))

    # Envelope tinggi gelombang
    fig.add_trace(go.Scatter(
        x=xs,
        y=Hh / 2,
        name='Envelope H (±H/2)',
        fill='tonexty',
        fillcolor='rgba(66,217,232,0.30)',
        line=dict(color=col, width=2),
        customdata=Hh,
        hovertemplate=(
            'x = %{x:.0f} m'
            '<br>H = %{customdata:.2f} m'
            '<extra></extra>'
        )
    ))

    # Area dan garis titik breaking
    if breaking_visible:
        xb = float(b['xb'])

        fig.add_vrect(
            x0=lo,
            x1=xb,
            fillcolor=RED,
            opacity=0.07,
            line_width=0,
            layer='below'
        )

        fig.add_vline(
            x=xb,
            line_color=RED,
            line_dash='dash',
            line_width=2
        )

        fig.add_trace(go.Scatter(
            x=[xb],
            y=[float(b['Hb']) / 2],
            mode='markers',
            name=f'Breaking xb = {xb:.0f} m',
            marker=dict(
                color='white',
                size=10,
                line=dict(color=RED, width=2)
            ),
            hovertemplate=(
                f'Breaking pada x = {xb:.1f} m'
                f'<br>hb = {float(b["hb"]):.2f} m'
                f'<br>Hb = {float(b["Hb"]):.2f} m'
                f'<extra></extra>'
            )
        ))

        # Panah ilustratif pada bagian domain yang terlihat
        xa_visible = max(float(lo), 0.0)
        xb_visible = min(float(hi), xb)

        if xb_visible - xa_visible > 0.05 * span:
            for f in (0.25, 0.50, 0.75):
                x0 = xa_visible + f * (xb_visible - xa_visible)
                yy = -float(np.interp(x0, r['x'], r['h'])) * 0.75
                x1 = min(x0 + 0.025 * span, hi)

                if x1 > x0:
                    fig.add_annotation(
                        x=x1,
                        y=yy,
                        ax=x0,
                        ay=yy,
                        xref='x',
                        yref='y',
                        axref='x',
                        ayref='y',
                        showarrow=True,
                        arrowhead=3,
                        arrowsize=1,
                        arrowwidth=2,
                        arrowcolor=GREEN,
                        text=''
                    )

    # Informasi utama diletakkan di header panel
    if b is None:
        info_breaking = 'Tidak ada breaking terdeteksi'
    elif breaking_visible:
        info_breaking = (
            f"Breaking: xb = {float(b['xb']):.0f} m"
            f" | hb = {float(b['hb']):.2f} m"
            f" | Hb = {float(b['Hb']):.2f} m"
            f" | θb = {float(b['thb']):.1f}°"
        )
    else:
        info_breaking = (
            f"Breaking di luar rentang"
            f" (xb = {float(b['xb']):.0f} m)"
        )

    fig.add_annotation(
        x=0.02,
        y=0.99,
        xref='paper',
        yref='paper',
        text=f'θ₀ = {th0:.1f}° | {info_breaking}',
        showarrow=False,
        xanchor='left',
        yanchor='top',
        font=dict(color='white', size=10),
        bgcolor='rgba(10,20,32,0.94)',
        bordercolor='rgba(150,170,190,0.55)',
        borderwidth=1,
        borderpad=5
    )

    # Label shoreline pada sudut bawah
    if lo == 0:
        fig.add_annotation(
            x=0.01,
            y=0.02,
            xref='paper',
            yref='paper',
            text='SHORELINE | x = 0',
            showarrow=False,
            xanchor='left',
            yanchor='bottom',
            font=dict(color=MUTED, size=10)
        )

    style(fig, 500, 'closest').update_layout(
        title=dict(
            text='Penampang cross-shore',
            x=0.5,
            xanchor='center'
        ),
        xaxis_title='x (m) — offshore ke kanan',
        yaxis_title='Elevasi relatif muka air (m)',
        xaxis_range=[lo, hi],
        yaxis_range=[-ymax, ytop],
        margin=dict(t=105, r=25, b=65, l=65),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='left',
            x=0,
            font=dict(size=9)
        ),
        hovermode='closest'
    )

    left.plotly_chart(
        fig,
        use_container_width=True,
        config={'displaylogo': False, 'responsive': True},
        key=f'cross_section_{sc}'
    )

    # ==========================================================
    # B. TAMPAK ATAS: REFRAKSI CREST
    # ==========================================================
    Ly = max(span * 0.6, 100.0)
    xlo = lo - 0.08 * span if lo == 0 else lo

    # Hilangkan titik awal NaN dari data sudut dan panjang gelombang
    xt = np.asarray(r['x'][1:], dtype=float)
    tht = np.asarray(r['theta'][1:], dtype=float)
    Lt = np.asarray(r['L'][1:], dtype=float)

    valid = np.isfinite(xt) & np.isfinite(tht) & np.isfinite(Lt)
    xt, tht, Lt = xt[valid], tht[valid], Lt[valid]

    fp = go.Figure()

    # Daratan dan garis pantai
    if lo == 0:
        fp.add_vrect(
            x0=xlo,
            x1=0,
            fillcolor='#B8793A',
            opacity=0.45,
            line_width=0,
            layer='below'
        )

        fp.add_vline(
            x=0,
            line_color='#D6B078',
            line_width=3
        )

    # Gambar crest jika data valid tersedia
    if xt.size >= 2:
        Lnom = max(float(np.interp(hi, xt, Lt)), 40.0)
        ys = np.arange(0, Ly + 1, 5.0)
        dy = 5.0

        # Batasi jumlah crest agar render tetap ringan
        starts = np.arange(0, hi + Ly + Lnom, Lnom)[:150]

        for x_start in starts:
            xc = [float(x_start)]

            for _ in ys[1:]:
                x_eval = np.clip(xc[-1], xt[0], xt[-1])

                theta_local = float(
                    np.interp(x_eval, xt, tht)
                )

                theta_local = np.clip(
                    theta_local, -89.0, 89.0
                )

                dx_dy = -np.tan(np.deg2rad(theta_local))
                xc.append(xc[-1] + dx_dy * dy)

            xc = np.asarray(xc)

            ok = (
                (xc >= max(lo, 1))
                & (xc <= hi)
            )

            if np.count_nonzero(ok) >= 2:
                fp.add_trace(go.Scatter(
                    x=xc[ok],
                    y=ys[ok],
                    mode='lines',
                    line=dict(color=col, width=2),
                    showlegend=False,
                    hoverinfo='skip'
                ))

    # Garis breaking pada tampak atas
    if breaking_visible:
        xb = float(b['xb'])

        fp.add_vline(
            x=xb,
            line_color=RED,
            line_dash='dash',
            line_width=2
        )

        # Label breaking pada sisi atas, tidak di tengah crest
        fp.add_annotation(
            x=xb,
            y=0.89,
            xref='x',
            yref='paper',
            text='BREAKING',
            showarrow=False,
            xanchor='left',
            yanchor='bottom',
            font=dict(color=RED, size=10),
            bgcolor='rgba(10,20,32,0.88)',
            borderpad=3
        )

        # Panah longshore ditempatkan di area bawah panel
        x_long = lo + 0.68 * span
        y_long = Ly * 0.16
        y_arrow_end = min(Ly * 0.28, Ly * 0.95)

        fp.add_annotation(
            x=x_long,
            y=y_arrow_end,
            ax=x_long,
            ay=y_long,
            xref='x',
            yref='y',
            axref='x',
            ayref='y',
            showarrow=True,
            arrowhead=3,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor=AMBER,
            text=''
        )

        fp.add_annotation(
            x=x_long,
            y=y_arrow_end + Ly * 0.025,
            xref='x',
            yref='y',
            text='Longshore (skematik)',
            showarrow=False,
            xanchor='left',
            yanchor='bottom',
            font=dict(color=AMBER, size=9),
            bgcolor='rgba(10,20,32,0.82)',
            borderpad=3
        )

    # Informasi sudut datang diletakkan terpisah di header
    fp.add_annotation(
        x=0.02,
        y=0.99,
        xref='paper',
        yref='paper',
        text=f'θ₀ = {th0:.1f}° | Refraksi crest',
        showarrow=False,
        xanchor='left',
        yanchor='top',
        font=dict(color='white', size=10),
        bgcolor='rgba(10,20,32,0.94)',
        bordercolor='rgba(150,170,190,0.55)',
        borderwidth=1,
        borderpad=5
    )

    style(fp, 500, 'closest').update_layout(
        title=dict(
            text='Tampak atas (refraksi crest)',
            x=0.5,
            xanchor='center'
        ),
        xaxis_title='x (m)',
        yaxis_title='y longshore (m)',
        showlegend=False,
        xaxis_range=[xlo, hi],
        yaxis_range=[0, Ly],
        margin=dict(t=90, r=25, b=65, l=65),
        hovermode='closest'
    )

    right.plotly_chart(
        fp,
        use_container_width=True,
        config={'displaylogo': False, 'responsive': True},
        key=f'top_view_{sc}'
    )

    # ==========================================================
    # C. KETERANGAN DI BAWAH GRAFIK
    # ==========================================================
    if b is None:
        txt = 'Tidak ada breaking terdeteksi.'
    elif breaking_visible:
        txt = (
            f"Breaking pada x = {float(b['xb']):.1f} m, "
            f"hb = {float(b['hb']):.2f} m, "
            f"Hb = {float(b['Hb']):.2f} m, "
            f"θb = {float(b['thb']):.1f}°."
        )
    else:
        txt = (
            f"Breaking berada di x = {float(b['xb']):.1f} m, "
            "di luar rentang penampang yang ditampilkan."
        )

    st.caption(
        txt
        + ' Panah cross-shore dan longshore hanya ilustrasi, '
        + 'bukan hasil perhitungan arus dari model. '
        + 'Garis crest dibentuk menggunakan distribusi θ(x).'
    )

# ---------------------------- ⑦ DATA ----------------------------------------
with tabs[6]:
    sc = st.selectbox('Skenario', list(R.keys()), key='data_sc')
    r = R[sc]

    def frame(rr):
        return pd.DataFrame({'x_m': rr['x'], 'h_m': rr['h'], 'k': rr['k'], 'L_m': rr['L'], 'C_m_s': rr['C'], 'Cg_m_s': rr['Cg'], 'theta_deg': rr['theta'],
                             'H_linear_m': rr['Hpre'], 'H_m': rr['H'], 'F_W_m': rr['F'], 'D_W_m2': rr['D'], 'Sxx_N_m': rr['Sxx'], 'Sxy_N_m': rr['Sxy'],
                             'Fy_N_m2': rr['Fy'], 'setup_m': rr['eta'], 'Pl_W_m': rr['Pl']})
    df = frame(r)
    st.dataframe(df.iloc[::10].round(3), use_container_width=True, hide_index=True, height=300)
    st.caption(f'Validasi: residu maksimum relasi dispersi = {r["resid"]:.1e}; fluks energi normal-pantai F konstan sebelum breaking karena H dihitung dari kekekalan F.')
    d1, d2 = st.columns(2)
    d1.download_button('📥 CSV (skenario ini, 1 m)', df.to_csv(index=False).encode(), f'Kasus_B_{sc[:2]}.csv', 'text/csv', use_container_width=True)
    try:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine='openpyxl') as w:
            pd.DataFrame([{'Skenario': n, 'xb': (rr['brk'] or {}).get('xb'), 'hb': (rr['brk'] or {}).get('hb'), 'Hb': (rr['brk'] or {}).get('Hb'),
                           'thetab': (rr['brk'] or {}).get('thb'), 'Pl_b': (rr['brk'] or {}).get('Plb'), 'V': (rr['brk'] or {}).get('V')} for n, rr in R.items()]).to_excel(w, sheet_name='Breaking', index=False)
            for n, rr in R.items(): frame(rr).to_excel(w, sheet_name=f'Detail_{n[:2]}', index=False)
        d2.download_button('📥 Excel (semua skenario)', buf.getvalue(), 'Hasil_Transformasi_Gelombang_Kasus_B.xlsx',
                           'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', use_container_width=True)
    except ImportError:
        d2.caption('Tambahkan `openpyxl` ke requirements.txt untuk ekspor Excel.')