import simpy
import random
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import math

st.set_page_config(layout="wide", page_title="Simulador Colas M/M/1 y M/M/c (Final)")

# ----------------------------
# FUNCIONES ANALÍTICAS
# ----------------------------
def mm1_analitico(lmbda, mu):
    if lmbda >= mu:
        return {"estable": False}
    rho = lmbda / mu
    L = rho / (1 - rho)
    Lq = rho**2 / (1 - rho)
    W = 1 / (mu - lmbda)
    Wq = lmbda / (mu * (mu - lmbda))
    return {"estable": True, "rho": rho, "L": L, "Lq": Lq, "W": W, "Wq": Wq}

def mmc_analitico(lmbda, mu, c):
    if c < 1:
        raise ValueError("c debe ser >= 1")
    if lmbda >= c * mu:
        return {'estable': False}

    a = lmbda / mu
    rho = lmbda / (c * mu)

    sum_terms = sum((a**n) / math.factorial(n) for n in range(c))
    last = (a**c) / (math.factorial(c) * (1 - rho))
    P0 = 1.0 / (sum_terms + last)

    ErlangC = last * P0 * (1 / (1 - rho))

    Lq = (ErlangC * lmbda) / (c * mu - lmbda)
    Wq = Lq / lmbda
    W = Wq + 1/mu
    L = lmbda * W

    return {
        'estable': True, 'a': a, 'rho': rho, 'P0': P0,
        'ErlangC': ErlangC, 'Lq': Lq, 'Wq': Wq, 'W': W, 'L': L
    }

# ----------------------------
# SIMULACIÓN (SimPy)
# ----------------------------
def customer(env, server, mu, stats):
    arrival = env.now
    with server.request() as req:
        yield req
        wait = env.now - arrival
        stats['waits'].append(wait)
        service_time = random.expovariate(mu)
        yield env.timeout(service_time)
        stats['sojourns'].append(env.now - arrival)

def arrival_generator(env, server, lmbda, mu, stats, tiempo_max):
    while env.now < tiempo_max:
        inter = random.expovariate(lmbda)
        yield env.timeout(inter)
        env.process(customer(env, server, mu, stats))

def run_once(lmbda, mu, c, tiempo_max, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    env = simpy.Environment()
    server = simpy.Resource(env, capacity=c)
    stats = {'waits': [], 'sojourns': []}
    env.process(arrival_generator(env, server, lmbda, mu, stats, tiempo_max))
    env.run(until=tiempo_max)
    return stats

def run_replicas(lmbda, mu, c, tiempo_max, replications=3, base_seed=100):
    results = []
    for r in range(replications):
        stats = run_once(lmbda, mu, c, tiempo_max, seed=base_seed + r)

        avg_wait = float(np.mean(stats['waits'])) if stats['waits'] else 0.0
        avg_sojourn = float(np.mean(stats['sojourns'])) if stats['sojourns'] else 0.0
        n_served = len(stats['sojourns'])

        results.append({
            'rep': r+1, 'avg_wait': avg_wait, 'avg_sojourn': avg_sojourn,
            'n_served': n_served, 'raw': stats
        })

    df = pd.DataFrame([
        {'replica': r['rep'], 'Wq_emp': r['avg_wait'], 'W_emp': r['avg_sojourn'], 'clientes_atendidos': r['n_served']}
        for r in results
    ])

    summary = {
        'Wq_mean': df['Wq_emp'].mean() if not df.empty else 0.0,
        'Wq_std': df['Wq_emp'].std(ddof=1) if len(df) > 1 else 0.0,
        'W_mean': df['W_emp'].mean() if not df.empty else 0.0,
        'W_std': df['W_emp'].std(ddof=1) if len(df) > 1 else 0.0,
        'total_served': df['clientes_atendidos'].sum() if not df.empty else 0,
        'df': df,
        'raw_results': results
    }

    return summary

# ----------------------------
# INTERFAZ STREAMLIT
# ----------------------------
st.title("Simulación REAL de Colas M/M/1 y M/M/c — Comparación Analítica vs Simulada")

tab_model, tab_sim, tab_interp, tab_export = st.tabs(["Modelo", "Simulación", "Interpretación Didáctica", "Exportar"])

# ----------------------------
# TAB 1 — MODELO
# ----------------------------
with tab_model:
    st.header("Parámetros del Modelo")

    c1, c2, c3 = st.columns(3)
    with c1:
        lmbda = st.number_input("λ — Tasa de llegada", min_value=0.01, value=0.9, step=0.01)
        mu = st.number_input("μ — Tasa de servicio por servidor", min_value=0.01, value=1.0, step=0.01)
    with c2:
        c = st.slider("Número de servidores (c)", min_value=1, max_value=20, value=2)
    with c3:
        tiempo_max = st.number_input("Tiempo de simulación", min_value=100, value=5000, step=100)
        replications = st.number_input("Replicaciones", min_value=1, value=3)

    st.subheader("Resultados Analíticos")

    ana = mm1_analitico(lmbda, mu) if c == 1 else mmc_analitico(lmbda, mu, c)

    if not ana["estable"]:
        st.error("⚠ El sistema es inestable (λ ≥ c·μ).")
    else:
        st.json(ana)

# ----------------------------
# TAB 2 — SIMULACIÓN
# ----------------------------
with tab_sim:
    st.header("Simulación REAL")

    if st.button("Correr Simulación"):
        with st.spinner("Simulando..."):
            sim = run_replicas(lmbda, mu, c, tiempo_max, replications=int(replications))

        st.success("Simulación completada")

        st.subheader("Resumen agregado por réplicas")
        st.dataframe(sim["df"])

        st.markdown("""
### ¿Qué significa este resumen?

- **Wq_emp** = tiempo promedio de espera en cola medido en la simulación  
- **W_emp** = tiempo total en sistema medido (cola + servicio)  
- **Variabilidad entre réplicas:**  
    - Si la desviación estándar (std) es alta, significa que la simulación presenta mucha variación → aumentar tiempo o nº de réplicas  
- **Clientes atendidos:** determina el tamaño muestral real  
""")

        st.write(f"**Promedio Wq (simulado):** {sim['Wq_mean']:.6f}")
        st.write(f"**Desviación estándar Wq:** {sim['Wq_std']:.6f}")
        st.write(f"**Promedio W  (simulado):** {sim['W_mean']:.6f}")
        st.write(f"**Desviación estándar W:** {sim['W_std']:.6f}")
        st.write(f"**Clientes atendidos:** {sim['total_served']}")

        # Histograma
        first_raw = sim["raw_results"][0]["raw"]
        if first_raw["waits"]:
            fig = px.histogram(first_raw["waits"], nbins=40,
                               title="Histograma de tiempos de espera — Réplica 1")
            st.plotly_chart(fig)

        # ----------------------------
        # COMPARACIÓN ANALÍTICA VS SIMULADA
        # ----------------------------
        st.subheader("Comparación Analítica vs Simulada")

        if not ana["estable"]:
            st.warning("La teoría indica que el sistema es inestable, pero la simulación puede mostrar colas crecientes.")
        else:
            comp = pd.DataFrame([
                {"métrica": "W (sistema total)", "analítico": ana["W"], "simulado_promedio": sim["W_mean"], "simulado_std": sim["W_std"]},
                {"métrica": "Wq (cola)", "analítico": ana["Wq"], "simulado_promedio": sim["Wq_mean"], "simulado_std": sim["Wq_std"]}
            ])
            st.table(comp)

            def pct(a, b):
                try:
                    return 100*(b - a)/a
                except:
                    return None

            st.write(f"**Diferencia porcentual W:** {pct(ana['W'], sim['W_mean']):.2f}%")
            st.write(f"**Diferencia porcentual Wq:** {pct(ana['Wq'], sim['Wq_mean']):.2f}%")

        st.session_state["sim"] = sim
        st.session_state["ana"] = ana

# ----------------------------
# TAB 3 — INTERPRETACIÓN DIDÁCTICA (COMPLETA)
# ----------------------------
with tab_interp:
    st.header("Interpretación didáctica y guías de trabajo")

    st.markdown("""
### 🔷 Representaciones
En la simulación trabajamos con varios registros de representación (Duval):

- **Simbólico**: Fórmulas analíticas del modelo (M/M/1 o M/M/c).
- **Numérico/Tabular**: Resultados empíricos generados con SimPy.
- **Gráfico**: Histogramas y comparaciones visuales entre teoría y simulación.

Esto permite al estudiante *coordinar registros*: comprender cómo la teoría se expresa en datos reales.

---

### 🔷 Aproximación al límite (ρ → 1)
Cuando la **utilización** ρ = λ / (c·μ) se acerca a 1:

- El tiempo en cola crece rápidamente.
- Un pequeño aumento en λ produce grandes aumentos en Wq.
- La simulación muestra alta variabilidad e inestabilidad.
- El sistema tarda mucho en recuperar equilibrio.

Esto permite visualizar fenómenos que solo con fórmulas serían difíciles de comprender (Medina).

---

### 🔷 Preguntas guiadas para actividad
1. Fija **μ** y aumenta **λ** poco a poco.  
   ¿Cómo cambian W y Wq? ¿Cómo se refleja en la gráfica?
2. Compara **1 servidor vs 3 servidores** con la misma λ.  
   ¿Qué mejora observas en W?
3. Si el sistema está inestable (λ ≥ c·μ), ¿cómo se comporta la simulación?
4. ¿Qué estrategias podrían estabilizar el sistema?
   - ¿Aumentar servidores?  
   - ¿Aumentar μ (velocidad)?  
   - ¿Reducir λ?

---

### 🔷 Notas metodológicas
- La simulación presenta **variabilidad natural** → por eso usamos **replicaciones**.
- A mayor tiempo de simulación → menor varianza en los promedios.
- Cuando λ ≥ c·μ:
  - La teoría marca **divergencia**.
  - La simulación muestra colas que **crecen indefinidamente**.

Estas notas ayudan al estudiante a distinguir entre:
- el comportamiento **ideal** (modelo analítico), y  
- el comportamiento **real** (simulación con variabilidad).

""")

# ----------------------------
# TAB 4 — EXPORTAR
# ----------------------------
with tab_export:
    st.header("Exportar resultados")

    if "sim" not in st.session_state:
        st.info("Primero corre una simulación.")
    else:
        sim = st.session_state["sim"]

        st.subheader("Exportar resumen por réplica")
        csv_reps = sim["df"].to_csv(index=False)
        st.download_button("Descargar resumen (CSV)", csv_reps, "resumen_replicas.csv")

        st.subheader("Exportar todas las réplicas (datos completos)")
        all_rows = []
        for r in sim["raw_results"]:
            rep = r["rep"]
            waits = r["raw"]["waits"]
            soj = r["raw"]["sojourns"]
            length = min(len(waits), len(soj))
            for i in range(length):
                all_rows.append({"replica": rep, "wait_time": waits[i], "sojourn_time": soj[i]})

        if len(all_rows) == 0:
            st.info("No hay datos suficientes para exportar (aumenta el tiempo de simulación).")
        else:
            df_all = pd.DataFrame(all_rows)
            st.download_button("Descargar TODAS las réplicas (CSV)", df_all.to_csv(index=False), "todas_las_replicas.csv")