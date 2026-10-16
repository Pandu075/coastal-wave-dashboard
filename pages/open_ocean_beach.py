import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# -----------------------------------------------------------------------------
# 0. SETUP & TEMA (mengikuti app.py)
# -----------------------------------------------------------------------------
st.set_page_config(page_title='Open Ocean Beach - Case A', page_icon='〰', layout='wide', initial_sidebar_state='expanded')

BG, TEXT, MUTED, LINE = '#07111F', '#E6F1FA', '#8CA5BA', '#20364D'
CYAN, AMBER, RED, GREEN = '#42D9E8', '#F5B95F', '#FF7185', '#7EE787'
SCEN_COLORS = {'A1': CYAN, 'A2': AMBER, 'A3': RED}

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
# 1. MODEL (mengikuti notebook Kasus A, di-cache per kombinasi parameter)
# -----------------------------------------------------------------------------
RHO, G = 1025.0, 9.81
DEPTHS = [10, 5, 2]


def sm(a, w=9):
    return np.convolve(np.nan_to_num(a), np.ones(w) / w, mode='same')


@st.cache_data
def read_bathymetry(raw):
    """Baca xlsx batimetri (sheet 'A_Open_Beach' atau sheet pertama). Kolom: x (m), Depth h (m)."""
    xl = pd.ExcelFile(io.BytesIO(raw))
    df = xl.parse('A_Open_Beach' if 'A_Open_Beach' in xl.sheet_names else xl.sheet_names[0])
    cx = next(c for c in df.columns if str(c).strip().lower().startswith('x'))
    ch = next(c for c in df.columns if 'depth' in str(c).lower() or str(c).strip().lower().startswith('h'))
    d = df[[cx, ch]].dropna().sort_values(cx)
    return d[cx].to_numpy(float), d[ch].to_numpy(float)


@st.cache_data
def simulate_all(H0_a, H0_a2, T, th0, gamma, N12, N3, ref, up_x, up_h):
    omega = 2 * np.pi / T
    if up_x is not None:
        x12 = np.arange(0, up_x.max() + .5, 1.0); h12 = np.interp(x12, up_x, up_h)
    else:
        x12 = np.arange(0, 1001, 1.0); h12 = x12 / N12
    x3 = np.arange(0, 1001, 1.0); h3 = x3 / N3
    sc = {'A1 Baseline': (H0_a, x12, h12, N12), 'A2 High Wave': (H0_a2, x12, h12, N12), 'A3 Steep Beach': (H0_a, x3, h3, N3)}

    def dispersion(h):
        hs = np.maximum(h, 1e-3)
        k = omega ** 2 / G + omega / np.sqrt(G * hs)
        for _ in range(60):
            t = np.tanh(k * hs)
            k = k - (G * k * t - omega ** 2) / (G * t + G * k * hs / np.cosh(k * hs) ** 2)
        k = np.where(h > 0, k, np.nan)
        C = omega / k
        n = .5 * (1 + 2 * k * h / np.sinh(2 * k * h))
        return k, 2 * np.pi / k, C, n, n * C

    out = {}
    for name, (H0, x, h, slope_N) in sc.items():
        k, L, C, n, Cg = dispersion(h)
        th0r = np.deg2rad(th0)
        if ref == 'Laut dalam (notebook)':
            C0 = G * T / (2 * np.pi); Cg0 = .5 * C0
        else:
            C0, Cg0 = C[-1], Cg[-1]
        th = np.arcsin(np.clip(C / C0 * np.sin(th0r), -1, 1))
        Ks = np.sqrt(Cg0 / Cg); Kr = np.sqrt(np.cos(th0r) / np.cos(th))
        Hpre = H0 * Ks * Kr
        cap = gamma * h
        H = np.minimum(Hpre, cap); H[0] = 0
        E = RHO * G * H ** 2 / 8
        F = np.nan_to_num(E * Cg * np.cos(th))
        D = sm(np.maximum(np.gradient(F, x), 0))
        Sxy = -E * n * np.sin(th) * np.cos(th)
        Fy = sm(-np.gradient(np.nan_to_num(Sxy), x), 11)
        Pl = E * Cg * np.sin(th) * np.cos(th)
        brk = None
        f = Hpre - cap
        pos = np.where(f[1:] >= 0)[0] + 1
        if len(pos):
            i0 = pos.max()
            if i0 == len(x) - 1:
                xb = x[-1]
            else:
                i1 = i0 + 1
                xb = x[i1] + (-f[i1]) / (f[i0] - f[i1]) * (x[i0] - x[i1])
            hb = float(np.interp(xb, x, h)); Hb = gamma * hb
            thb = float(np.interp(xb, x, np.rad2deg(th)))
            Cgb = float(np.interp(xb, x, Cg)); Lb = float(np.interp(xb, x, L)); Cb = float(np.interp(xb, x, C))
            Ksb = float(np.interp(xb, x, Ks)); Krb = float(np.interp(xb, x, Kr))
            s, c = np.sin(np.deg2rad(thb)), np.cos(np.deg2rad(thb))
            m = float(np.interp(xb, x, np.gradient(h, x)))
            brk = dict(xb=float(xb), hb=hb, Hb=Hb, thb=thb, Lb=Lb, Cb=Cb, Cgb=Cgb, Ksb=Ksb, Krb=Krb,
                       Plb=RHO * G * Hb ** 2 / 8 * Cgb * s * c, V=20.7 * m * np.sqrt(G * Hb) * s * c, m=m,
                       Sxyb=float(np.interp(xb, x, Sxy)))
        out[name] = dict(x=x, h=h, k=k, L=L, C=C, n=n, Cg=Cg, theta=np.rad2deg(th), Ks=Ks, Kr=Kr, Hpre=Hpre, cap=cap, H=H,
                         E=E, F=F, D=D, Sxy=Sxy, Fy=Fy, Pl=Pl, brk=brk, H0=H0, th0=th0, slope=slope_N)
    return out


def at_depth(r, key, d):
    return float(np.interp(d, r['h'][1:], np.asarray(r[key])[1:]))


# -----------------------------------------------------------------------------
# 2. SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown(f'<div style="display:flex;align-items:center;gap:12px;margin-bottom:14px"><div style="background:linear-gradient(145deg,{CYAN},#1A7D9C);width:42px;height:42px;border-radius:13px;display:grid;place-items:center;font-size:23px;color:#07111F">〰</div><div><b style="font-size:1.05rem;color:{TEXT}">CoastalLab</b><div style="font-size:.7rem;color:{MUTED};letter-spacing:.08em">OPEN OCEAN BEACH</div></div></div>', unsafe_allow_html=True)
    st.button('← Kembali ke Beranda', on_click=lambda: st.switch_page('app.py'), use_container_width=True)
    st.markdown('---')
    st.markdown('<div class="eyebrow">PARAMETER GELOMBANG</div>', unsafe_allow_html=True)
    H0_a = st.slider('H₀ A1 & A3 (m)', 0.5, 4.0, 1.5, 0.05)
    H0_a2 = st.slider('H₀ A2 high wave (m)', 0.5, 4.0, 2.25, 0.05)
    T = st.slider('Periode T (s)', 4.0, 16.0, 8.0, 0.5)
    th0 = st.slider('Sudut datang θ₀ (°)', 0, 60, 15)
    gamma = st.slider('Breaker index γ', 0.6, 1.0, 0.78, 0.01)
    st.markdown('<div class="eyebrow" style="margin-top:14px">BATIMETRI</div>', unsafe_allow_html=True)
    N12 = st.slider('Slope A1 & A2 (1:N)', 20, 100, 50)
    N3 = st.slider('Slope A3 (1:N)', 15, 100, 35)
    up = st.file_uploader('Opsional: Excel batimetri A1/A2', type=['xlsx'], help="Sheet 'A_Open_Beach' dengan kolom 'x (m)' dan 'Depth h (m)'. Jika kosong dipakai profil linier 1:N.")
    ref = st.radio('Acuan gelombang awal (H₀, θ₀)', ['Laut dalam (notebook)', 'Ujung profil'], help='Notebook memakai Cg₀ = 0,5·C₀ (laut dalam). "Ujung profil" memakai kedalaman terluar sebagai acuan.')
    up_x = up_h = None
    if up is not None:
        try:
            up_x, up_h = read_bathymetry(up.getvalue())
        except Exception as e:
            st.error(f'File tidak bisa dibaca ({e}); memakai profil linier.')
    st.markdown('<div class="eyebrow" style="margin-top:14px">TAMPILAN</div>', unsafe_allow_html=True)
    results = simulate_all(H0_a, H0_a2, T, th0, gamma, N12, N3, ref, up_x, up_h)
    sel = st.multiselect('Skenario', list(results.keys()), default=list(results.keys()))
    xr = st.slider('Zoom x (m)', 0, 1000, (0, 1000), step=50)

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
st.markdown(f'<div class="eyebrow">KASUS A / OPEN OCEAN BEACH</div><h2 style="margin:.2rem 0 .6rem;letter-spacing:-.03em;color:{TEXT}">Shoaling, Refraksi, dan Breaking di Pantai Terbuka</h2>', unsafe_allow_html=True)
kc = st.columns(len(R))
for c, (n, r) in zip(kc, R.items()):
    b = r['brk']
    with c, st.container(border=True):
        st.markdown(f'<b style="color:{col_of(n)}">{n}</b> <span style="color:{MUTED};font-size:.8rem">· H₀ {r["H0"]:.2f} m · 1:{r["slope"]}</span>', unsafe_allow_html=True)
        a = st.columns(4)
        a[0].metric('xb', f'{b["xb"]:.0f} m' if b else '–')
        a[1].metric('hb', f'{b["hb"]:.2f} m' if b else '–')
        a[2].metric('Hb', f'{b["Hb"]:.2f} m' if b else '–')
        a[3].metric('θb', f'{b["thb"]:.1f}°' if b else '–')

tabs = st.tabs(['① Batimetri', '② Transformasi', '③ Energi & Longshore', '④ Perbandingan A1–A3', '⑤ Sketsa Penampang', '⑥ Data'])

# ---------------------------- ① BATIMETRI -----------------------------------
with tabs[0]:
    left, right = st.columns([3, 2])
    fig = go.Figure()
    for n, r in R.items():
        fig.add_trace(go.Scatter(x=r['x'], y=-r['h'], name=f'{n} (1:{r["slope"]})' if up_x is None or n.startswith('A3') else n,
                                 line=dict(color=col_of(n), width=2.5, dash='dot' if n.startswith('A2') else 'solid')))
    fig.add_hline(y=0, line_color=MUTED, line_dash='dash', annotation_text='MSL')
    style(fig, 380).update_layout(xaxis_title='x (m) dari shoreline, positif ke offshore', yaxis_title='Elevasi dasar (m)')
    fig.update_xaxes(range=list(xr))
    left.plotly_chart(fig, use_container_width=True)
    right.dataframe(pd.DataFrame([{'Skenario': n, 'H₀ (m)': r['H0'], 'T (s)': T, 'θ₀ (°)': r['th0'], 'h maks (m)': round(r['h'][-1], 1),
                                   'L₀ (m)': round(9.81 * T ** 2 / (2 * np.pi), 1)} for n, r in R.items()]), hide_index=True, use_container_width=True)
    right.caption('A1 dan A2 memakai profil yang sama; A3 lebih curam. Jika Excel diunggah, profil A1/A2 diganti dengan data tersebut (interpolasi tiap 1 m).')

# ---------------------------- ② TRANSFORMASI --------------------------------
with tabs[1]:
    o1, o2, o3, o4 = st.columns(4)
    byx = o1.radio('Sumbu horizontal', ['Jarak x (m)', 'Kedalaman h (m)'], horizontal=True).startswith('Jarak')
    show_lin = o2.toggle('H linear (tanpa breaking)', True)
    show_cap = o3.toggle('Batas γh', True)
    show_k = o4.toggle('Koefisien Ks & Kr', False)
    rows = (3 if byx else 2) + (1 if show_k else 0)
    titles = (['Batimetri (m)'] if byx else []) + ['Tinggi gelombang H (m)', 'Sudut gelombang θ (°)'] + (['Koefisien shoaling Ks (solid) dan refraksi Kr (putus)'] if show_k else [])
    base_h = [.2, .45, .35] if byx else [.6, .4]
    heights = (base_h + [.3]) if show_k else base_h
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=heights, subplot_titles=titles)
    rH, rT = (2, 3) if byx else (1, 2)
    rK = rT + 1
    sl = slice(1, None)
    for n, r in R.items():
        c = col_of(n)
        xv = r['x'][sl] if byx else r['h'][sl]
        if byx: fig.add_trace(go.Scatter(x=xv, y=-r['h'][sl], name=n, legendgroup=n, line=dict(color=c, width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=xv, y=r['H'][sl], name=f'H {n[:2]}', legendgroup=n, showlegend=not byx, line=dict(color=c, width=3)), row=rH, col=1)
        if show_lin: fig.add_trace(go.Scatter(x=xv, y=r['Hpre'][sl], legendgroup=n, showlegend=False, name=f'H linear {n[:2]}', line=dict(color=c, width=1.5, dash='dot')), row=rH, col=1)
        if show_cap: fig.add_trace(go.Scatter(x=xv, y=r['cap'][sl], legendgroup=n, showlegend=False, name=f'γh {n[:2]}', line=dict(color=c, width=1, dash='dash'), opacity=.55), row=rH, col=1)
        fig.add_trace(go.Scatter(x=xv, y=r['theta'][sl], legendgroup=n, showlegend=False, name=f'θ {n[:2]}', line=dict(color=c, width=2.5)), row=rT, col=1)
        if show_k:
            fig.add_trace(go.Scatter(x=xv, y=r['Ks'][sl], legendgroup=n, showlegend=False, name=f'Ks {n[:2]}', line=dict(color=c, width=2.2)), row=rK, col=1)
            fig.add_trace(go.Scatter(x=xv, y=r['Kr'][sl], legendgroup=n, showlegend=False, name=f'Kr {n[:2]}', line=dict(color=c, width=1.5, dash='dash')), row=rK, col=1)
        b = r['brk']
        if b:
            fig.add_trace(go.Scatter(x=[b['xb'] if byx else b['hb']], y=[b['Hb']], mode='markers', legendgroup=n, showlegend=False,
                                     marker=dict(color='white', size=10, line=dict(color=c, width=2)),
                                     hovertemplate=f'Breaking {n[:2]}: xb={b["xb"]:.0f} m, hb={b["hb"]:.2f} m, Hb={b["Hb"]:.2f} m<extra></extra>'), row=rH, col=1)
    fig.add_hline(y=th0, line_dash='dash', line_color=MUTED, annotation_text=f'θ₀ = {th0}°', row=rT, col=1)
    fig.update_yaxes(range=[0, max(H0_a2 * 1.8, 3)], row=rH, col=1)
    style(fig, 560 + 140 * (rows - 2))
    if byx:
        fig.update_xaxes(range=list(xr)); fig.update_xaxes(title_text='x (m), positif ke offshore', row=rows, col=1)
    else:
        fig.update_xaxes(range=[max(r['h'][-1] for r in R.values()), 0]); fig.update_xaxes(title_text='Kedalaman h (m) — offshore ke kiri, pantai ke kanan', row=rows, col=1)
    st.plotly_chart(fig, use_container_width=True)
    st.caption('Titik putih = breaking pertama dari offshore (H/h ≥ γ). Setelah breaking, H dibatasi H = γh. A1 dan A2 memiliki θ(x) identik karena Snell hanya bergantung pada kedalaman dan θ₀.')

# ---------------------------- ③ ENERGI & LONGSHORE --------------------------
with tabs[2]:
    left, right = st.columns([3, 2])
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        subplot_titles=['Fluks energi normal-pantai F (W/m)', 'Disipasi energi D (W/m²)', 'Radiation stress silang Sxy (N/m)', 'Longshore forcing Fy = −dSxy/dx (N/m²)'])
    for n, r in R.items():
        c = col_of(n)
        fig.add_trace(go.Scatter(x=r['x'][1:], y=r['F'][1:], name=n, legendgroup=n, line=dict(color=c, width=2.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=r['x'][1:], y=r['D'][1:], showlegend=False, legendgroup=n, name=f'D {n[:2]}', line=dict(color=c, width=2.5), fill='tozeroy', fillcolor='rgba(255,255,255,.03)'), row=2, col=1)
        fig.add_trace(go.Scatter(x=r['x'][1:], y=r['Sxy'][1:], showlegend=False, legendgroup=n, name=f'Sxy {n[:2]}', line=dict(color=c, width=2.5)), row=3, col=1)
        fig.add_trace(go.Scatter(x=r['x'][1:], y=r['Fy'][1:], showlegend=False, legendgroup=n, name=f'Fy {n[:2]}', line=dict(color=c, width=2.5)), row=4, col=1)
    mark_breakers(fig, 4)
    style(fig, 780)
    fig.update_xaxes(range=list(xr)); fig.update_xaxes(title_text='x (m), positif ke offshore', row=4, col=1)
    left.plotly_chart(fig, use_container_width=True)

    bn = [n for n in R if R[n]['brk']]
    fig2 = make_subplots(rows=2, cols=1, vertical_spacing=0.18, subplot_titles=['Pl di titik breaking (kW/m)', 'Arus longshore V (m/s), Longuet-Higgins'])
    fig2.add_trace(go.Bar(x=[n[:2] for n in bn], y=[R[n]['brk']['Plb'] / 1000 for n in bn], marker_color=[col_of(n) for n in bn],
                         text=[f'{R[n]["brk"]["Plb"] / 1000:.1f}' for n in bn], textposition='outside'), row=1, col=1)
    fig2.add_trace(go.Bar(x=[n[:2] for n in bn], y=[R[n]['brk']['V'] for n in bn], marker_color=[col_of(n) for n in bn],
                         text=[f'{R[n]["brk"]["V"]:.2f}' for n in bn], textposition='outside'), row=2, col=1)
    style(fig2, 780, 'closest').update_layout(showlegend=False)
    right.plotly_chart(fig2, use_container_width=True)
    st.caption('Fy adalah indikator gradien radiation stress, bukan kecepatan arus. Tanda mengikuti x positif ke offshore. V ≈ 20,7·m·√(g·Hb)·sinθb·cosθb, m = kemiringan dasar di titik breaking.')

# ---------------------------- ④ PERBANDINGAN --------------------------------
with tabs[3]:
    left, right = st.columns(2)
    fig = go.Figure()
    for n, r in R.items():
        fig.add_trace(go.Bar(x=[f'h={d} m' for d in DEPTHS], y=[at_depth(r, 'H', d) for d in DEPTHS], name=n, marker_color=col_of(n),
                             text=[f'{at_depth(r, "H", d):.2f}' for d in DEPTHS], textposition='outside'))
    style(fig, 360, 'closest').update_layout(barmode='group', title='H pada kedalaman sama (m)')
    left.plotly_chart(fig, use_container_width=True)
    bn = [n for n in R if R[n]['brk']]
    fig = make_subplots(rows=1, cols=2, subplot_titles=['Lokasi breaking xb (m)', 'Tinggi breaking Hb (m)'])
    fig.add_trace(go.Bar(x=[n[:2] for n in bn], y=[R[n]['brk']['xb'] for n in bn], marker_color=[col_of(n) for n in bn], text=[f'{R[n]["brk"]["xb"]:.0f}' for n in bn], textposition='outside'), row=1, col=1)
    fig.add_trace(go.Bar(x=[n[:2] for n in bn], y=[R[n]['brk']['Hb'] for n in bn], marker_color=[col_of(n) for n in bn], text=[f'{R[n]["brk"]["Hb"]:.2f}' for n in bn], textposition='outside'), row=1, col=2)
    style(fig, 360, 'closest').update_layout(showlegend=False)
    right.plotly_chart(fig, use_container_width=True)

    rows = []
    for n, r in R.items():
        b = r['brk']
        rows.append({'Skenario': n, 'H₀ (m)': r['H0'], 'slope': f'1:{r["slope"]}', 'xb (m)': round(b['xb'], 1) if b else None, 'hb (m)': round(b['hb'], 2) if b else None,
                     'Hb (m)': round(b['Hb'], 2) if b else None, 'θb (°)': round(b['thb'], 2) if b else None, 'Ks': round(b['Ksb'], 3) if b else None,
                     'Kr': round(b['Krb'], 3) if b else None, 'Pl,b (kW/m)': round(b['Plb'] / 1000, 2) if b else None, 'V (m/s)': round(b['V'], 2) if b else None})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    keys = ('A1 Baseline', 'A2 High Wave', 'A3 Steep Beach')
    if all(k in results and results[k]['brk'] for k in keys):
        a1, a2, a3 = (results[k]['brk'] for k in keys)
        with st.container(border=True):
            st.markdown('**Ringkasan isolasi variabel** (angka dari model, interpretasi tetap perlu ditulis sendiri)')
            st.markdown(f"**A1 vs A2 (efek H₀):** H₀ {results[keys[0]]['H0']:.2f} → {results[keys[1]]['H0']:.2f} m menggeser xb dari {a1['xb']:.0f} ke {a2['xb']:.0f} m (hb {a1['hb']:.2f} → {a2['hb']:.2f} m) "
                        f"dan menaikkan Hb dari {a1['Hb']:.2f} ke {a2['Hb']:.2f} m. θb sama ({a1['thb']:.1f}° vs {a2['thb']:.1f}°) karena refraksi tidak bergantung pada H₀; Pl,b naik {a2['Plb'] / a1['Plb']:.2f}×.")
            st.markdown(f"**A1 vs A3 (efek slope):** slope 1:{N12} → 1:{N3} menggeser xb dari {a1['xb']:.0f} ke {a3['xb']:.0f} m. Hb {a1['Hb']:.2f} → {a3['Hb']:.2f} m dan θb {a1['thb']:.1f}° → {a3['thb']:.1f}°. "
                        f"Pada kedalaman sama nilai H dan θ identik; slope mengubah lokasi dan lebar surf zone, serta nilai V ({a1['V']:.2f} → {a3['V']:.2f} m/s) lewat suku kemiringan m.")
    else:
        st.info('Aktifkan A1, A2, dan A3 di sidebar untuk melihat ringkasan isolasi variabel.')


# ---------------------------- ⑤ SKETSA PENAMPANG ----------------------------
with tabs[4]:
    c1, c2 = st.columns([1, 2])

    sc = c1.selectbox('Skenario', list(R.keys()))
    r = R[sc]
    b = r['brk']
    col = col_of(sc)

    xmax = int(r['x'][-1])
    lo, hi = c2.slider(
        'Rentang x ditampilkan (m)',
        0, xmax,
        (0, min(xmax, 500)),
        step=50,
        key=f'sk_{sc}'
    )

    # Pastikan rentang minimum tetap valid
    hi = max(hi, min(lo + 100, xmax))
    span = max(hi - lo, 1)

    m = (r['x'] >= lo) & (r['x'] <= hi)
    xs = r['x'][m]
    hh = r['h'][m]
    Hh = r['H'][m]

    # Fallback jika rentang slider tidak memiliki titik grid
    if xs.size == 0:
        idx = np.where((r['x'] >= lo) & (r['x'] <= hi))[0]
        if idx.size == 0:
            idx = np.array([np.argmin(np.abs(r['x'] - lo))])
        xs = r['x'][idx]
        hh = r['h'][idx]
        Hh = r['H'][idx]

    ymax = float(np.nanmax(hh)) * 1.1 + 1
    ytop = max(float(np.nanmax(Hh)) / 2, 1) + 2.2

    left, right = st.columns([2, 1])

    # ==========================================================
    # A. PENAMPANG CROSS-SHORE
    # ==========================================================
    fig = go.Figure()

    # Area dasar laut
    fig.add_trace(go.Scatter(
        x=xs,
        y=np.full(xs.size, -ymax),
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))

    fig.add_trace(go.Scatter(
        x=xs,
        y=-hh,
        name='Dasar laut',
        fill='tonexty',
        fillcolor='rgba(214,176,120,0.55)',
        line=dict(color='#D6B078', width=2),
        hovertemplate='x=%{x:.0f} m<br>Dasar=%{y:.2f} m<extra></extra>'
    ))

    # Muka air diam
    fig.add_trace(go.Scatter(
        x=xs,
        y=np.zeros(xs.size),
        name='Muka air diam',
        fill='tonexty',
        fillcolor='rgba(66,217,232,0.10)',
        line=dict(color=MUTED, width=1, dash='dot'),
        hovertemplate='x=%{x:.0f} m<br>Muka air=0 m<extra></extra>'
    ))

    # Batas bawah envelope gelombang
    fig.add_trace(go.Scatter(
        x=xs,
        y=-Hh / 2,
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))

    # Envelope gelombang ±H/2
    fig.add_trace(go.Scatter(
        x=xs,
        y=Hh / 2,
        name='Envelope H (±H/2)',
        fill='tonexty',
        fillcolor='rgba(66,217,232,0.30)',
        line=dict(color=col, width=2),
        customdata=Hh,
        hovertemplate=(
            'x=%{x:.0f} m'
            '<br>H=%{customdata:.2f} m'
            '<extra></extra>'
        )
    ))

    # Posisi breaking hanya jika berada dalam rentang tampilan
    breaking_visible = (
        b is not None
        and lo <= float(b['xb']) <= hi
    )

    if breaking_visible:
        xb = float(b['xb'])

        # Area sebelum titik breaking
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

        # Marker breaking
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
            customdata=[[
                float(b['hb']),
                float(b['Hb']),
                float(b['thb'])
            ]],
            hovertemplate=(
                'Breaking'
                '<br>x=%{x:.1f} m'
                '<br>hb=%{customdata[0]:.2f} m'
                '<br>Hb=%{customdata[1]:.2f} m'
                '<br>θb=%{customdata[2]:.1f}°'
                '<extra></extra>'
            )
        ))

        # Panah cross-shore: hanya pada area yang terlihat
        xa_visible = max(float(lo), 0.0)
        xb_visible = min(float(hi), xb)

        if xb_visible - xa_visible > 0.05 * span:
            for f in (0.25, 0.50, 0.75):
                x0 = xa_visible + f * (xb_visible - xa_visible)
                yy = -float(np.interp(x0, r['x'], r['h'])) * 0.75

                # Panah dibuat pendek agar tidak mengganggu profil
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

    # Informasi utama di bagian atas panel, menggunakan koordinat panel
    fig.add_annotation(
        x=0.02,
        y=0.98,
        xref='paper',
        yref='paper',
        text=(
            f"θ₀ = {th0}°"
            + (
                f" | Breaking: xb = {float(b['xb']):.0f} m"
                f" | hb = {float(b['hb']):.2f} m"
                f" | Hb = {float(b['Hb']):.2f} m"
                f" | θb = {float(b['thb']):.1f}°"
                if breaking_visible
                else (
                    " | Breaking di luar rentang"
                    if b is not None
                    else " | Tidak ada breaking terdeteksi"
                )
            )
        ),
        showarrow=False,
        xanchor='left',
        yanchor='top',
        font=dict(color='white', size=10),
        bgcolor='rgba(10,20,32,0.92)',
        bordercolor='rgba(150,170,190,0.5)',
        borderwidth=1,
        borderpad=5
    )

    # Penanda shoreline
    if lo == 0:
        fig.add_annotation(
            x=0.01,
            y=0.04,
            xref='paper',
            yref='paper',
            text='SHORELINE  x = 0',
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
            font=dict(size=10)
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
    fp = go.Figure()

    xlo = lo - 0.08 * span if lo == 0 else lo

    # Zona daratan di shoreline
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

    # Panjang gelombang lokal di ujung rentang
    Lnom = float(np.interp(hi, r['x'], r['L']))
    Lnom = max(Lnom, 1.0)

    # Jarak antargaris crest disesuaikan dengan panjang gelombang
    dx_crest = max(Lnom, 40.0)

    # Bangun crest dengan integrasi sederhana:
    # dx/dy = -tan(theta)
    y_step = 5.0
    ys = np.arange(0, Ly + y_step, y_step)

    # Batasi jumlah crest agar grafik tetap ringan
    n_crest = min(
        int(np.ceil((hi + Ly + Lnom) / dx_crest)) + 1,
        150
    )

    for x_start in np.arange(
        0,
        hi + Ly + Lnom,
        dx_crest
    )[:n_crest]:

        xc = [float(x_start)]

        for _ in ys[1:]:
            theta_local = float(
                np.interp(
                    np.clip(xc[-1], r['x'][0], r['x'][-1]),
                    r['x'],
                    r['theta']
                )
            )

            # Hindari nilai ekstrem akibat data sudut yang tidak valid
            theta_local = np.clip(theta_local, -89.0, 89.0)
            dx_dy = -np.tan(np.deg2rad(theta_local))

            xc.append(xc[-1] + dx_dy * y_step)

        xc = np.asarray(xc)

        # Tampilkan hanya bagian crest dalam rentang x
        ok = (xc >= max(lo, 0)) & (xc <= hi)

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

        fp.add_annotation(
            x=xb,
            y=0.90,
            xref='x',
            yref='paper',
            text='Breaking',
            showarrow=False,
            xanchor='left',
            yanchor='bottom',
            font=dict(color=RED, size=10),
            bgcolor='rgba(10,20,32,0.85)',
            borderpad=3
        )

        # Arah longshore digambarkan di bagian tengah panel
        fp.add_annotation(
            x=lo + 0.72 * span,
            y=Ly * 0.22,
            ax=lo + 0.72 * span,
            ay=Ly * 0.10,
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
            x=lo + 0.74 * span,
            y=Ly * 0.24,
            xref='x',
            yref='y',
            text='Longshore',
            showarrow=False,
            xanchor='left',
            yanchor='bottom',
            font=dict(color=AMBER, size=10),
            bgcolor='rgba(10,20,32,0.8)',
            borderpad=3
        )

    # Label informasi tetap di bagian atas panel
    fp.add_annotation(
        x=0.02,
        y=0.98,
        xref='paper',
        yref='paper',
        text=f"θ₀ = {th0}° | Tampak atas — refraksi crest",
        showarrow=False,
        xanchor='left',
        yanchor='top',
        font=dict(color='white', size=10),
        bgcolor='rgba(10,20,32,0.92)',
        bordercolor='rgba(150,170,190,0.5)',
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
    # C. KETERANGAN
    # ==========================================================
    if b is None:
        txt = (
            'Tidak ada breaking terdeteksi pada skenario ini. '
            'Pastikan kondisi gelombang dan kedalaman mencakup '
            'kemungkinan terjadinya breaking.'
        )
    elif not breaking_visible:
        txt = (
            f"Breaking berada di x = {float(b['xb']):.1f} m, "
            'di luar rentang penampang yang sedang ditampilkan.'
        )
    else:
        txt = (
            f"Breaking pada x = {float(b['xb']):.1f} m; "
            f"kedalaman hb = {float(b['hb']):.2f} m; "
            f"tinggi Hb = {float(b['Hb']):.2f} m; "
            f"sudut θb = {float(b['thb']):.1f}°."
        )

    st.caption(
        txt
        + ' Panah cross-shore dan longshore bersifat ilustratif. '
        + 'Garis crest menunjukkan arah berdasarkan distribusi θ(x).'
    )

# ---------------------------- ⑥ DATA ----------------------------------------
with tabs[5]:
    sc = st.selectbox('Skenario', list(R.keys()), key='data_sc')

    def frame(rr):
        return pd.DataFrame({'x (m)': rr['x'], 'h (m)': rr['h'], 'L (m)': rr['L'], 'k (1/m)': rr['k'], 'C (m/s)': rr['C'], 'n': rr['n'], 'Cg (m/s)': rr['Cg'],
                             'Theta (deg)': rr['theta'], 'Ks': rr['Ks'], 'Kr': rr['Kr'], 'H linear (m)': rr['Hpre'], 'H (m)': rr['H'], 'F (W/m)': rr['F'],
                             'Sxy (N/m)': rr['Sxy'], 'Fy (N/m2)': rr['Fy'], 'Pl (W/m)': rr['Pl']})
    df = frame(R[sc])
    st.dataframe(df.iloc[::10].round(4), use_container_width=True, hide_index=True, height=300)
    d1, d2 = st.columns(2)
    d1.download_button('📥 CSV (skenario ini, 1 m)', df.to_csv(index=False).encode(), f'Hasil_Transformasi_{sc[:2]}.csv', 'text/csv', use_container_width=True)
    try:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine='openpyxl') as w:
            pd.DataFrame([{'Skenario': n, **{k: v for k, v in (rr['brk'] or {}).items()}} for n, rr in R.items()]).to_excel(w, sheet_name='Breaking', index=False)
            for n, rr in R.items(): frame(rr).to_excel(w, sheet_name=f'Detail_{n[:2]}', index=False)
        d2.download_button('📥 Excel (semua skenario)', buf.getvalue(), 'Hasil_Transformasi_Kasus_A.xlsx',
                           'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', use_container_width=True)
    except ImportError:
        d2.caption('Tambahkan `openpyxl` ke requirements.txt untuk ekspor Excel.')