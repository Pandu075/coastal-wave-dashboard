import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# -----------------------------------------------------------------------------
# 0. SETUP & TEMA (mengikuti app.py)
# -----------------------------------------------------------------------------
st.set_page_config(page_title='Reef-Influenced Beach', page_icon='🪸', layout='wide', initial_sidebar_state='expanded')

BG, TEXT, MUTED, LINE = '#07111F', '#E6F1FA', '#8CA5BA', '#20364D'
CYAN, AMBER, RED, GREEN = '#42D9E8', '#F5B95F', '#FF7185', '#7EE787'
SCEN_COLORS = {'C1': CYAN, 'C2': AMBER, 'C3': RED, 'C*': GREEN}

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
[data-testid="stMetricValue"]{{font-size:1.35rem}}
</style>''', unsafe_allow_html=True)

BASE = dict(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Manrope,sans-serif', color=TEXT, size=12), margin=dict(l=10, r=10, t=40, b=10),
            hoverlabel=dict(bgcolor='#102338', bordercolor='#2B536D', font_color=TEXT),
            legend=dict(orientation='h', y=1.08, x=0))


def style(fig, height):
    fig.update_layout(**BASE, height=height, hovermode='x unified')
    fig.update_xaxes(gridcolor=LINE, zerolinecolor=LINE)
    fig.update_yaxes(gridcolor=LINE, zerolinecolor=LINE)
    return fig


# -----------------------------------------------------------------------------
# 1. MODEL (di-cache per kombinasi parameter)
# -----------------------------------------------------------------------------
RHO, G = 1025.0, 9.81
X_NODES = np.array([0, 25, 50, 75, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 700, 850, 1000], dtype=float)
H_NODES = {
    'C1 Normal Reef': np.array([0, .50, 1, 1.50, 2, 3, 3, 3, 3, 3, 3, 3, 4, 5.50, 7, 10, 15, 20]),
    'C2 Shallow Reef': np.array([0, .33, .67, 1, 1.33, 2, 2, 2, 2, 2, 2, 2, 3.50, 5.20, 6.80, 10, 15, 20]),
    'C3 Deep Reef': np.array([0, .67, 1.33, 2, 2.67, 4, 4, 4, 4, 4, 4, 4, 4.70, 5.80, 7.30, 10, 15, 20]),
}
X = np.linspace(0, 1000, 1001)  # indeks = meter
ZONES = [('Beach', 0, 150), ('Reef flat', 150, 450), ('Reef slope', 450, 600), ('Offshore', 600, 1000)]
SAMPLE_X = {'Offshore': 1000, 'Reef slope': 525, 'Reef flat': 300, 'Beach': 75}
FLUX_ZONES = [('Reef slope', 600, 450), ('Reef flat', 450, 150), ('Beach', 150, 1)]


def custom_profile(d):
    """Profil linier: pantai 0->d, flat d, slope d->7 m (x=450..600), lalu sama dengan tabel."""
    h = np.empty_like(X_NODES)
    for i, xn in enumerate(X_NODES):
        if xn <= 150: h[i] = d * xn / 150
        elif xn <= 450: h[i] = d
        elif xn <= 600: h[i] = d + (7 - d) * (xn - 450) / 150
        else: h[i] = [10, 15, 20][[700, 850, 1000].index(xn)]
    return h


@st.cache_data
def simulate_all(H0, T, theta0, gamma, custom_d):
    omega = 2 * np.pi / T
    nodes = dict(H_NODES)
    if custom_d:
        nodes['C* Custom'] = custom_profile(custom_d)

    def dispersion(h):
        hs = np.maximum(h, 1e-3)
        k = omega ** 2 / G + omega / np.sqrt(G * hs)  # tebakan awal >= akar, Newton konvergen
        for _ in range(60):
            t = np.tanh(k * hs)
            f = G * k * t - omega ** 2
            df = G * t + G * k * hs / np.cosh(k * hs) ** 2
            k = k - f / df
        k = np.where(h > 0, k, np.nan)
        C = omega / k
        kh = k * h
        n = .5 * (1 + 2 * kh / np.sinh(2 * kh))
        return k, 2 * np.pi / k, C, n * C

    out = {}
    for name, hn in nodes.items():
        h = np.interp(X, X_NODES, hn)
        k, L, C, Cg = dispersion(h)
        th = np.arcsin(np.clip(np.sin(np.deg2rad(theta0)) * C / C[-1], -1, 1))
        Hlin = H0 * np.sqrt(Cg[-1] * np.cos(np.deg2rad(theta0)) / (Cg * np.cos(th)))
        cap = gamma * h
        H = np.full_like(h, np.nan)
        H[-1] = min(H0, cap[-1])
        for j in range(len(X) - 2, -1, -1):
            if h[j] <= 0:
                H[j] = 0
                continue
            H[j] = min(H[j + 1] * np.sqrt(Cg[j + 1] * np.cos(th[j + 1]) / (Cg[j] * np.cos(th[j]))), cap[j])
        limited = (H < Hlin - 1e-6) & (h > 0)
        E = RHO * G * H ** 2 / 8
        F = np.nan_to_num(E * Cg * np.cos(th))
        D = np.convolve(np.gradient(F, X), np.ones(15) / 15, mode='same')  # disipasi dF/dx (W/m²)
        out[name] = dict(h=h, k=k, L=L, C=C, Cg=Cg, theta=np.rad2deg(th), Hlin=Hlin, H=H, cap=cap,
                         limited=limited, E=E, F=F, D=D, depth=float(h[300]))
    return out


def breaking(r):
    idx = np.where(r['limited'])[0]
    if not len(idx): return None
    j = idx[-1]
    return dict(j=j, xb=X[j], hb=r['h'][j], Hb=r['H'][j], thb=r['theta'][j])


def flux_loss(r):
    return {z: (r['F'][a] - r['F'][b]) / r['F'][600] * 100 for z, a, b in FLUX_ZONES}


def longshore(r, b):
    m = r['h'][150] / 150  # kemiringan muka pantai
    th = np.deg2rad(b['thb'])
    return 20.7 * m * np.sqrt(G * b['Hb']) * np.sin(th) * np.cos(th)


def add_zones(fig, **kw):
    for z, a, b in ZONES:
        fig.add_vrect(x0=a, x1=b, fillcolor='white', opacity=0.03, line_width=0, **kw)


def zone_labels(fig, row=1):
    for z, a, b in ZONES:
        fig.add_annotation(x=(a + b) / 2, y=1.0, yref='y domain' if row == 1 else f'y{row} domain', xref='x',
                           text=z, showarrow=False, font=dict(color=MUTED, size=10), yanchor='bottom')


# -----------------------------------------------------------------------------
# 2. SIDEBAR: navigasi + parameter interaktif
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown(f'<div style="display:flex;align-items:center;gap:12px;margin-bottom:14px"><div style="background:linear-gradient(145deg,{CYAN},#1A7D9C);width:42px;height:42px;border-radius:13px;display:grid;place-items:center;font-size:23px;color:#07111F">🪸</div><div><b style="font-size:1.05rem;color:{TEXT}">CoastalLab</b><div style="font-size:.7rem;color:{MUTED};letter-spacing:.08em">REEF-INFLUENCED BEACH</div></div></div>', unsafe_allow_html=True)
    
    # Tombol kembali ke Beranda (Sesuaikan jika nama file utama kamu berbeda)
    st.button('← Kembali ke Beranda', on_click=lambda: st.switch_page('app.py'), use_container_width=True)
    st.markdown('---')
    
    st.markdown(f'<div class="eyebrow">PARAMETER GELOMBANG</div>', unsafe_allow_html=True)
    H0 = st.slider('H₀ offshore (m)', 0.5, 4.0, 1.8, 0.1)
    T = st.slider('Periode T (s)', 4.0, 16.0, 9.0, 0.5)
    theta0 = st.slider('Sudut datang θ₀ (°)', 0, 45, 15)
    gamma = st.slider('Breaker index γ', 0.6, 1.0, 0.78, 0.01)
    
    st.markdown(f'<div class="eyebrow" style="margin-top:14px">SKENARIO</div>', unsafe_allow_html=True)
    use_custom = st.toggle('Tambah skenario custom (C*)', value=False)
    custom_d = st.slider('Kedalaman reef flat C* (m)', 1.0, 6.0, 2.5, 0.1) if use_custom else 0
    
    results = simulate_all(H0, T, theta0, gamma, custom_d)
    sel = st.multiselect('Tampilkan Skenario', list(results.keys()), default=list(H_NODES.keys()))

if not sel:
    st.warning('Pilih minimal satu skenario di sidebar.')
    st.stop()
    
R = {n: results[n] for n in sel}
col_of = lambda n: SCEN_COLORS.get(n[:2], GREEN)

# -----------------------------------------------------------------------------
# 3. HEADER + KPI
# -----------------------------------------------------------------------------
st.markdown(f'<div class="eyebrow">KASUS C / REEF-INFLUENCED BEACH</div><h2 style="margin:.2rem 0 .6rem;letter-spacing:-.03em;color:{TEXT}">Transformasi Gelombang di Pantai Berterumbu</h2>', unsafe_allow_html=True)

kc = st.columns(len(R))
for c, (n, r) in zip(kc, R.items()):
    b = breaking(r)
    with c, st.container(border=True):
        st.markdown(f'<b style="color:{col_of(n)}">{n}</b> <span style="color:{MUTED};font-size:.8rem">· reef flat {r["depth"]:.1f} m</span>', unsafe_allow_html=True)
        a1, a2, a3 = st.columns(3)
        a1.metric('H pantai (x=75)', f'{r["H"][75]:.2f} m')
        a2.metric('Breaking xb', f'{b["xb"]:.0f} m' if b else '–')
        a3.metric('Hb', f'{b["Hb"]:.2f} m' if b else '–')

tabs = st.tabs(['① Batimetri', '② Transformasi', '③ Energi', '④ Perbandingan', '⑤ Sketsa Penampang', '⑥ Data'])

# ---------------------------- ① BATIMETRI -----------------------------------
with tabs[0]:
    st.markdown("<p class='note'>Profil dasar laut: Beach 0–150 m, Reef flat 150–450 m, Reef slope 450–600 m, Offshore > 600 m.</p>", unsafe_allow_html=True)
    fig = go.Figure()
    for n, r in R.items():
        fig.add_trace(go.Scatter(x=X, y=-r['h'], name=n, line=dict(color=col_of(n), width=2.5)))
    add_zones(fig)
    zone_labels(fig)
    style(fig, 430).update_layout(xaxis_title='Jarak dari shoreline x (m)', yaxis_title='Elevasi dasar (m)')
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------- ② TRANSFORMASI --------------------------------
with tabs[1]:
    o1, o2, o3 = st.columns(3)
    show_lin = o1.toggle('Tampilkan H linear (tanpa breaking)', True)
    show_cap = o2.toggle('Tampilkan batas γh', True)
    show_hn = o3.toggle('Tampilkan sudut θ', True)
    
    rows = 3 if show_hn else 2
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        row_heights=[.22, .48, .30][:rows] if show_hn else [.3, .7],
                        subplot_titles=['Batimetri (m)', 'Tinggi gelombang H (m)'] + (['Sudut gelombang θ (°)'] if show_hn else []))
    
    for n, r in R.items():
        c = col_of(n)
        fig.add_trace(go.Scatter(x=X, y=-r['h'], name=n, legendgroup=n, line=dict(color=c, width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=X, y=r['H'], name=f'H {n[:2]}', legendgroup=n, showlegend=False, line=dict(color=c, width=3)), row=2, col=1)
        if show_lin:
            fig.add_trace(go.Scatter(x=X, y=r['Hlin'], name=f'H linear {n[:2]}', legendgroup=n, showlegend=False, line=dict(color=c, width=1.5, dash='dot')), row=2, col=1)
        if show_cap:
            fig.add_trace(go.Scatter(x=X, y=r['cap'], name=f'γh {n[:2]}', legendgroup=n, showlegend=False, line=dict(color=c, width=1, dash='dash'), opacity=.55), row=2, col=1)
        
        b = breaking(r)
        if b:
            fig.add_trace(go.Scatter(x=[b['xb']], y=[b['Hb']], mode='markers+text', text=[f'xb={b["xb"]:.0f} m'], textposition='top center',
                                     marker=dict(color='white', size=10, line=dict(color=c, width=2)), showlegend=False, legendgroup=n,
                                     hovertemplate='Breaking: x=%{x:.0f} m, Hb=%{y:.2f} m<extra></extra>'), row=2, col=1)
        if show_hn:
            fig.add_trace(go.Scatter(x=X, y=r['theta'], showlegend=False, legendgroup=n, line=dict(color=c, width=2.5)), row=3, col=1)
            
    if show_hn:
        fig.add_hline(y=theta0, line_dash='dash', line_color=MUTED, annotation_text=f'θ₀ = {theta0}°', row=3, col=1)
        
    for rr in range(1, rows + 1): 
        add_zones(fig, row=rr, col=1)
    zone_labels(fig)
    
    fig.update_yaxes(range=[0, max(H0 * 1.6, 2.5)], row=2, col=1)
    style(fig, 720 if show_hn else 560).update_layout(xaxis_title=None)
    fig.update_xaxes(title_text='Jarak dari shoreline x (m)', row=rows, col=1)
    st.plotly_chart(fig, use_container_width=True)
    st.caption('Titik putih = breaking pertama dari offshore. Garis titik-titik = H tanpa pembatas breaking; garis putus-putus = batas γh.')

# ---------------------------- ③ ENERGI --------------------------------------
with tabs[2]:
    left, right = st.columns([3, 2])
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=['Fluks energi F (W/m)', 'Disipasi dF/dx (W/m²) — puncak = perubahan energi terkuat'])
    
    for n, r in R.items():
        c = col_of(n)
        fig.add_trace(go.Scatter(x=X, y=r['F'], name=n, legendgroup=n, line=dict(color=c, width=2.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=X, y=r['D'], name=n, legendgroup=n, showlegend=False, line=dict(color=c, width=2.5), fill='tozeroy', fillcolor='rgba(255,255,255,0.03)'), row=2, col=1)
        
    for rr in (1, 2): 
        add_zones(fig, row=rr, col=1)
    zone_labels(fig)
    style(fig, 560).update_layout(xaxis2_title='Jarak dari shoreline x (m)')
    left.plotly_chart(fig, use_container_width=True)

    zl = {n: flux_loss(r) for n, r in R.items()}
    fig2 = go.Figure()
    for n in R:
        fig2.add_trace(go.Bar(x=[z for z, _, _ in FLUX_ZONES], y=[zl[n][z] for z, _, _ in FLUX_ZONES], name=n, marker_color=col_of(n),
                              text=[f'{zl[n][z]:.0f}%' for z, _, _ in FLUX_ZONES], textposition='outside'))
    style(fig2, 560).update_layout(barmode='group', title='Kehilangan fluks per zona (% dari F di x=600 m)', yaxis_title='% F offshore', hovermode='closest')
    right.plotly_chart(fig2, use_container_width=True)
    
    for n in R:
        z = max(zl[n], key=zl[n].get)
        st.markdown(f'**{n[:2]}** — perubahan energi terkuat di **{z}** ({zl[n][z]:.0f}% dari fluks offshore).')

# ---------------------------- ④ PERBANDINGAN --------------------------------
with tabs[3]:
    left, right = st.columns(2)
    zn = list(SAMPLE_X.keys())
    fig = go.Figure()
    for n, r in R.items():
        fig.add_trace(go.Bar(x=zn, y=[r['H'][SAMPLE_X[z]] for z in zn], name=n, marker_color=col_of(n),
                             text=[f'{r["H"][SAMPLE_X[z]]:.2f}' for z in zn], textposition='outside'))
    style(fig, 400).update_layout(barmode='group', title='H per zona (m)', hovermode='closest')
    left.plotly_chart(fig, use_container_width=True)

    pts = sorted([(r['depth'], n, breaking(r)) for n, r in R.items() if breaking(r)], key=lambda t: t[0])
    fig = make_subplots(rows=1, cols=2, subplot_titles=['Lokasi breaking xb (m)', 'Tinggi breaking Hb (m)'])
    if pts:
        dd = [p[0] for p in pts]
        fig.add_trace(go.Scatter(x=dd, y=[p[2]['xb'] for p in pts], mode='lines+markers+text', text=[p[1][:2] for p in pts], textposition='top center', line=dict(color=CYAN), marker=dict(size=11)), row=1, col=1)
        fig.add_trace(go.Scatter(x=dd, y=[p[2]['Hb'] for p in pts], mode='lines+markers+text', text=[p[1][:2] for p in pts], textposition='top center', line=dict(color=AMBER), marker=dict(size=11)), row=1, col=2)
    style(fig, 400).update_layout(showlegend=False, hovermode='closest')
    fig.update_xaxes(title_text='Kedalaman reef flat (m)')
    right.plotly_chart(fig, use_container_width=True)

    rows = []
    for n, r in R.items():
        b = breaking(r)
        rows.append({'Skenario': n, 'Reef flat (m)': round(r['depth'], 1),
                     **{f'H {z} (m)': round(r['H'][SAMPLE_X[z]], 2) for z in zn},
                     'xb (m)': round(b['xb']) if b else None, 'Hb (m)': round(b['Hb'], 2) if b else None,
                     'θ di pantai (°)': round(r['theta'][75], 1)})
                     
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    order = sorted(R.items(), key=lambda kv: kv[1]['depth'])
    st.info('**H di pantai (x=75 m), urut dari reef dangkal → dalam:** ' + ' → '.join(f'{n[:2]} ({r["depth"]:.1f} m): {r["H"][75]:.2f} m' for n, r in order)
            + '. Bandingkan urutan ini dengan pertanyaan 6.5 no. 1: reef lebih dangkal belum tentu menghasilkan gelombang lebih kecil di pantai.')


# ---------------------------- ⑤ SKETSA PENAMPANG ----------------------------
with tabs[4]:
    c1, c2 = st.columns([1, 2])

    sc = c1.selectbox('Skenario', list(R.keys()))
    lo, hi = c2.slider(
        'Rentang x ditampilkan (m)',
        0, 1000, (0, 700),
        step=50,
        key=f'sk_{sc}'
    )

    hi = min(max(hi, lo + 100), 1000)
    span = max(hi - lo, 1)

    r = R[sc]
    b = breaking(r)
    col = col_of(sc)

    m = (X >= lo) & (X <= hi)
    xs, hh, Hh = X[m], r['h'][m], r['H'][m]

    if xs.size == 0:
        idx = int(np.argmin(np.abs(X - lo)))
        xs, hh, Hh = X[idx:idx+1], r['h'][idx:idx+1], r['H'][idx:idx+1]

    ymax = float(np.nanmax(hh)) * 1.1 + 1
    ytop = max(float(np.nanmax(Hh)) / 2, 1) + 2.2

    left, right = st.columns([2, 1])

    breaking_visible = (
        b is not None
        and lo <= float(b['xb']) <= hi
    )

    V = longshore(r, b) if breaking_visible else None

    # ==========================================================
    # A. PENAMPANG CROSS-SHORE
    # ==========================================================
    fig = go.Figure()

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

    fig.add_trace(go.Scatter(
        x=xs,
        y=np.zeros(xs.size),
        name='Muka air diam',
        fill='tonexty',
        fillcolor='rgba(66,217,232,0.10)',
        line=dict(color=MUTED, width=1, dash='dot'),
        hovertemplate='x=%{x:.0f} m<br>z=0 m<extra></extra>'
    ))

    fig.add_trace(go.Scatter(
        x=xs,
        y=-Hh / 2,
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))

    fig.add_trace(go.Scatter(
        x=xs,
        y=Hh / 2,
        name='Envelope H (±H/2)',
        fill='tonexty',
        fillcolor='rgba(66,217,232,0.30)',
        line=dict(color=col, width=2),
        customdata=Hh,
        hovertemplate='x=%{x:.0f} m<br>H=%{customdata:.2f} m<extra></extra>'
    ))

    # Zona morfologi
    add_zones(fig)

    for z, a, bb in ZONES:
        if bb > lo and a < hi:
            xa_zone = (max(a, lo) + min(bb, hi)) / 2
            fig.add_annotation(
                x=xa_zone,
                y=0.91,
                xref='x',
                yref='paper',
                text=z,
                showarrow=False,
                font=dict(color=MUTED, size=9),
                bgcolor='rgba(10,20,32,0.75)',
                borderpad=2
            )

    # Titik breaking
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
            name=f'Breaking xb={xb:.0f} m',
            marker=dict(
                color='white',
                size=10,
                line=dict(color=RED, width=2)
            ),
            hovertemplate=(
                f"xb={xb:.1f} m"
                f"<br>hb={float(b['hb']):.2f} m"
                f"<br>Hb={float(b['Hb']):.2f} m"
                f"<br>θb={float(b['thb']):.1f}°"
                "<extra></extra>"
            )
        ))

        # Panah cross-shore hanya di area breaking yang terlihat
        xa_visible = max(float(lo), 0.0)
        xb_visible = min(float(hi), xb)

        if xb_visible > xa_visible:
            for f in (0.25, 0.50, 0.75):
                x0 = xa_visible + f * (xb_visible - xa_visible)
                yy = -float(np.interp(x0, X, r['h'])) * 0.75
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

    # Arah gelombang: posisi tetap relatif terhadap panel
    fig.add_annotation(
        x=0.98,
        y=0.98,
        xref='paper',
        yref='paper',
        text=f'Arah gelombang θ₀ = {theta0:.1f}°',
        showarrow=False,
        xanchor='right',
        yanchor='top',
        font=dict(color=AMBER, size=10),
        bgcolor='rgba(10,20,32,0.92)',
        bordercolor=AMBER,
        borderwidth=1,
        borderpad=4
    )

    # Informasi breaking dipisahkan dari envelope gelombang
    if b is None:
        info = 'Tidak ada breaking terdeteksi'
    elif breaking_visible:
        info = (
            f"BREAKING | xb={float(b['xb']):.0f} m"
            f" | hb={float(b['hb']):.2f} m"
            f" | Hb={float(b['Hb']):.2f} m"
            f" | θb={float(b['thb']):.1f}°"
        )
    else:
        info = f"Breaking di luar rentang | xb={float(b['xb']):.0f} m"

    fig.add_annotation(
        x=0.02,
        y=0.98,
        xref='paper',
        yref='paper',
        text=info,
        showarrow=False,
        xanchor='left',
        yanchor='top',
        font=dict(color='white', size=9),
        bgcolor='rgba(10,20,32,0.92)',
        bordercolor='rgba(150,170,190,0.55)',
        borderwidth=1,
        borderpad=4
    )

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
            font=dict(color=MUTED, size=9)
        )

    style(fig, 500).update_layout(
        title='Penampang cross-shore',
        xaxis_title='x (m) — offshore ke kanan',
        yaxis_title='Elevasi (m)',
        hovermode='closest',
        yaxis_range=[-ymax, ytop + 0.6],
        xaxis_range=[lo, hi],
        margin=dict(t=110, r=25, b=60, l=65),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='left',
            x=0,
            font=dict(size=9)
        )
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

    xt = np.asarray(X[1:], dtype=float)
    tht = np.asarray(r['theta'][1:], dtype=float)
    Lt = np.asarray(r['L'][1:], dtype=float)

    # Filter nilai tidak valid, termasuk NaN
    valid = np.isfinite(xt) & np.isfinite(tht) & np.isfinite(Lt)
    xt, tht, Lt = xt[valid], tht[valid], Lt[valid]

    fp = go.Figure()

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

    # Crest gelombang
    if xt.size >= 2:
        Lnom = max(float(np.interp(hi, xt, Lt)), 40.0)
        ys = np.arange(0, Ly + 1, 5.0)
        dy = 5.0

        starts = np.arange(0, hi + Ly + Lnom, Lnom)[:150]

        for x_start in starts:
            xc = [float(x_start)]

            for _ in ys[1:]:
                x_eval = np.clip(xc[-1], xt[0], xt[-1])
                theta_local = float(np.interp(x_eval, xt, tht))
                theta_local = np.clip(theta_local, -89.0, 89.0)

                dx_dy = -np.tan(np.deg2rad(theta_local))
                xc.append(xc[-1] + dx_dy * dy)

            xc = np.asarray(xc)
            ok = (xc >= max(lo, 1)) & (xc <= hi)

            if np.count_nonzero(ok) >= 2:
                fp.add_trace(go.Scatter(
                    x=xc[ok],
                    y=ys[ok],
                    mode='lines',
                    line=dict(color=col, width=2),
                    showlegend=False,
                    hoverinfo='skip'
                ))

    # Breaking line
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
            text='BREAKING',
            showarrow=False,
            xanchor='left',
            yanchor='bottom',
            font=dict(color=RED, size=9),
            bgcolor='rgba(10,20,32,0.88)',
            borderpad=3
        )

        # Panah longshore di bagian bawah agar tidak menutupi crest
        x_long = lo + 0.62 * span
        y_start = Ly * 0.12
        y_end = Ly * 0.24

        fp.add_annotation(
            x=x_long,
            y=y_end,
            ax=x_long,
            ay=y_start,
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
            x=x_long + 0.015 * span,
            y=y_end,
            xref='x',
            yref='y',
            text='Longshore (skematik)',
            showarrow=False,
            xanchor='left',
            yanchor='middle',
            font=dict(color=AMBER, size=9),
            bgcolor='rgba(10,20,32,0.85)',
            borderpad=3
        )

    # Label header tetap
    fp.add_annotation(
        x=0.02,
        y=0.98,
        xref='paper',
        yref='paper',
        text=f'θ₀ = {theta0:.1f}° | Refraksi crest',
        showarrow=False,
        xanchor='left',
        yanchor='top',
        font=dict(color='white', size=10),
        bgcolor='rgba(10,20,32,0.92)',
        bordercolor='rgba(150,170,190,0.55)',
        borderwidth=1,
        borderpad=4
    )

    # Zona morfologi
    add_zones(fp)

    style(fp, 500).update_layout(
        title='Tampak atas (refraksi crest)',
        xaxis_title='x (m)',
        yaxis_title='y longshore (m)',
        showlegend=False,
        hovermode='closest',
        xaxis_range=[xlo, hi],
        yaxis_range=[0, Ly],
        margin=dict(t=90, r=25, b=60, l=65)
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
        txt = 'Tidak ada breaking terdeteksi.'
    elif breaking_visible:
        txt = (
            f"Breaking pada x={float(b['xb']):.1f} m, "
            f"h={float(b['hb']):.2f} m, "
            f"Hb={float(b['Hb']):.2f} m, "
            f"θb={float(b['thb']):.1f}°."
        )
    else:
        txt = (
            f"Breaking berada pada x={float(b['xb']):.1f} m, "
            "di luar rentang yang ditampilkan."
        )

    st.caption(
        txt
        + (
            f" Kecepatan longshore hasil fungsi model: V={V:.2f} m/s."
            if V is not None
            else ""
        )
        + " Panah cross-shore dan longshore bersifat skematik; "
        + "arah crest mengikuti θ(x)."
    )

# ---------------------------- ⑥ DATA ----------------------------------------
with tabs[5]:
    sc = st.selectbox('Skenario', list(R.keys()), key='data_sc')
    r = R[sc]
    df = pd.DataFrame({'x (m)': X, 'h (m)': r['h'], 'H (m)': r['H'], 'H linear (m)': r['Hlin'], 'L (m)': r['L'], 'C (m/s)': r['C'],
                       'Cg (m/s)': r['Cg'], 'theta (deg)': r['theta'], 'E (J/m2)': r['E'], 'F (W/m)': r['F'], 'dF/dx (W/m2)': r['D']})
    st.dataframe(df.iloc[::10].round(3), use_container_width=True, hide_index=True, height=320)
    
    d1, d2 = st.columns(2)
    d1.download_button('📥 CSV (skenario ini, 1 m)', df.to_csv(index=False).encode(), f'Kasus_C_{sc[:2]}.csv', 'text/csv', use_container_width=True)
    try:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine='openpyxl') as w:
            for n, rr in R.items():
                pd.DataFrame({'x (m)': X, 'h (m)': rr['h'], 'H (m)': rr['H'], 'C (m/s)': rr['C'], 'theta (deg)': rr['theta'], 'F (W/m)': rr['F']}).to_excel(w, sheet_name=n[:30], index=False)
        d2.download_button('📥 Excel (semua skenario)', buf.getvalue(), 'Data_Transformasi_Kasus_C.xlsx',
                           'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', use_container_width=True)
    except ImportError:
        d2.caption('Tambahkan `openpyxl` ke requirements.txt untuk ekspor Excel.')