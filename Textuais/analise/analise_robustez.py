"""
Análise de robustez e melhorias econométricas para o TCC.
Executa independentemente do notebook. 4 melhorias:
  1. Categorização por bloc (H3-H7) + tabela resumo por hipótese
  2. Modelos sem espectro político (econ bruto vs econ ajustado)
  3. Average Marginal Effects (AME) para os itens mais relevantes
  4. Bootstrap CIs (B=1000) para os 4 efeitos econ nominalmente significativos
"""

import os, re, warnings
import numpy as np
import pandas as pd
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.stats.multitest import multipletests
warnings.filterwarnings('ignore')

# ── 0. Reprodução do pré-processamento ──────────────────────────────────────

url = os.environ['GSHEET_URL']
df = pd.read_csv(url)

form_econ_col = 'Qual seu nível de formação em Ciências Econômicas? '
econ_map = {'Não tenho formação em Economia':0,'Graduação em Economia':0,
            'Mestrado em Economia':1,'Doutorado em Economia':1}
econ_series = df[form_econ_col].map(econ_map).astype('Int64')

clt_map = {'Empregado com carteira assinada (CLT)':1,'Empregado sem carteira assinada':0,
           'Autônomo':2,'Empresário':3,'Desempregado':-1,'Estudante':-2,
           'Aposentado':-3,'Servidor público':-4,'Outro':4}
df['Qual é o seu vínculo empregatício? '] = df['Qual é o seu vínculo empregatício? '].map(clt_map).astype('Int64')

ideol_map = {'Extrema esquerda':-2,'Esquerda':-1,'Centro':0,'Direita':1,
             'Extrema direita':2,'Independente':3,'Sem opinião':0}
ESPECTRO = 'Com qual espectro político você mais se identifica? '
df[ESPECTRO] = df[ESPECTRO].map(ideol_map).astype('Int64')

variaveis_controle = [
    form_econ_col,'Você é homem? ','Com qual grupo racial/étnico você mais se identifica? ',
    'Qual seu nível de escolaridade? ','Qual é a sua faixa etária? ',
    'Qual é o seu vínculo empregatício? ','Você se considera uma pessoa politicamente engajada? ',
    ESPECTRO
]
ctrl_sem_form = [c for c in variaveis_controle if c != form_econ_col]
df_controle_dummies = pd.get_dummies(df[ctrl_sem_form].copy(), drop_first=True, dtype=float)
df_controle_dummies['econ'] = econ_series.astype(float)
df_controle_dummies[ESPECTRO] = df[ESPECTRO].astype(float)
df_controle_dummies['Qual é o seu vínculo empregatício? '] = df['Qual é o seu vínculo empregatício? '].astype(float)
control_cols_full = df_controle_dummies.columns.tolist()
control_cols_sem_esp = [c for c in control_cols_full if c != ESPECTRO]

def detect_number(x):
    if pd.isna(x): return np.nan
    m = re.findall(r'[0-9]', str(x))
    return min(map(int, m)) if m else np.nan

def make_acronym(colname):
    col = re.sub(r'[^\w\s]', ' ', colname, flags=re.UNICODE)
    stop = {'de','da','do','das','dos','a','o','as','os','em','na','no','nas','nos',
            'para','por','com','ao','à','às','e','ou','que','se','é','uma','você','qual','sua','são','sobre','tem'}
    toks = [t.lower() for t in col.split() if t.lower() not in stop]
    return '_'.join(toks[:3]) if len(toks) >= 2 else (toks[0] if toks else col.lower())

colunas_excl = ['Carimbo de data/hora',
                'Ao clicar em "Aceito", você declara que compreendeu as informações e consente em participar.'
                ] + variaveis_controle
df_dep = df.drop(columns=colunas_excl).apply(lambda s: s.apply(detect_number)).astype('Int64')
rename_dep = {c: make_acronym(c) for c in df_dep.columns}
df_dep.rename(columns=rename_dep, inplace=True)
seen = {}; new_cols = []
for col in df_dep.columns:
    if col not in seen: seen[col] = 1; new_cols.append(col)
    else: seen[col] += 1; new_cols.append(f"{col}_{seen[col]}")
df_dep.columns = new_cols
df_dep = df_dep.drop(columns=['região_brasil_reside','trabalha_economia_finanças',
                               'trabalha_política_áreas','costuma_acompanhar_notícias'])
df_final = pd.concat([df_dep, df_controle_dummies], axis=1)
dvs = list(df_dep.columns)


# ── 1. Mapeamento por bloc hipótese ─────────────────────────────────────────

BLOCO = {
    # H3 — Viés Antimercado
    'empresas_lucram_demais': 'H3-Antimercado',
    'altos_executivos_ganham': 'H3-Antimercado',
    'governo_regulamenta_muito': 'H3-Antimercado',
    'há_deduções_demais': 'H3-Antimercado',
    'privatização_estatais_benéfica': 'H3-Antimercado',
    'competição_entre_empresas': 'H3-Antimercado',
    'lucros_empresariais_ocorrem': 'H3-Antimercado',
    'governo_deve_intervir': 'H3-Antimercado',
    'mulheres_minorias_têm': 'H3-Antimercado',
    # H4 — Viés Antiestrangeiro
    'temos_imigrantes_demais': 'H4-Antiestrangeiro',
    'gasto_ajuda_externa': 'H4-Antiestrangeiro',
    'acordos_comerciais_outros': 'H4-Antiestrangeiro',
    'acha_acordos_comerciais': 'H4-Antiestrangeiro',
    'entrada_estrangeiros_mercado': 'H4-Antiestrangeiro',
    'produtos_importados_benéficos': 'H4-Antiestrangeiro',
    'indústria_nacional_deve': 'H4-Antiestrangeiro',
    'brasil_deveria_adotar': 'H4-Antiestrangeiro',
    'brasil_deveria_priorizar': 'H4-Antiestrangeiro',
    # H5 — Viés Antitrabalho
    'tecnologia_causa_demissões': 'H5-Antitrabalho',
    'empresas_estão_enviando': 'H5-Antitrabalho',
    'empresas_estão_reduzindo': 'H5-Antitrabalho',
    'empresas_não_investem': 'H5-Antitrabalho',
    'mais_mulheres_força': 'H5-Antitrabalho',
    'aumento_uso_tecnologia': 'H5-Antitrabalho',
    'redução_recente_postos': 'H5-Antitrabalho',
    'automação_prejudica_mercado': 'H5-Antitrabalho',
    'acha_novos_postos': 'H5-Antitrabalho',
    'pessoas_não_dão': 'H5-Antitrabalho',
    # H6 — Viés Pessimista
    'últimos_20_anos': 'H6-Pessimista',
    'pensando_apenas_salários': 'H6-Pessimista',
    'próximos_cinco_anos': 'H6-Pessimista',
    'espera_geração_seus': 'H6-Pessimista',
    'filhos_menos_30': 'H6-Pessimista',
    'desigualdade_entre_ricos': 'H6-Pessimista',
    'produtividade_está_aumentando': 'H6-Pessimista',
    'pessoas_não_poupam': 'H6-Pessimista',
    'brasil_chance_virar': 'H6-Pessimista',
    # H7 — Influência Ideológica / Fiscal-Institucional
    'impostos_muito_altos': 'H7-Ideologia',
    'déficit_federal_grande': 'H7-Ideologia',
    'corte_impostos': 'H7-Ideologia',
    'seguridade_social_previdência': 'H7-Ideologia',
    'reforma_previdência_necessária': 'H7-Ideologia',
    'reforma_trabalhista_necessária': 'H7-Ideologia',
    'reforma_tributária_necessária': 'H7-Ideologia',
    'taxa_juros_selic': 'H7-Ideologia',
    'governo_atual_sabe': 'H7-Ideologia',
    'corrupção_principal_causa': 'H7-Ideologia',
    'educação_qualificação_profissional': 'H7-Ideologia',
    # Percepções específicas
    'acha_preços_combustíveis': 'Percepção-Específica',
    'acha_presidente_pode': 'Percepção-Específica',
    'quem_considera_maior': 'Percepção-Específica',
    'algumas_pessoas_dizem': 'H6-Pessimista',
    'algumas_pessoas_dizem_2': 'H7-Ideologia',
}

# ── 2. Função fit ─────────────────────────────────────────────────────────

def fit_ol(y, X, methods=('lbfgs', 'bfgs', 'powell')):
    model = OrderedModel(y, X, distr='logit', hasconst=False)
    for m in methods:
        try:
            r = model.fit(method=m, disp=False)
            return r
        except Exception:
            pass
    return None

def fit_ol_fast(y, X):
    """Versão rápida para bootstrap — só lbfgs, sem fallback."""
    try:
        model = OrderedModel(y, X, distr='logit', hasconst=False)
        return model.fit(method='lbfgs', disp=False)
    except Exception:
        return None

def beta_p(result, col):
    if result is None or col not in result.params.index:
        return np.nan, np.nan
    return float(result.params[col]), float(result.pvalues[col])

def is_singular(result):
    if result is None: return True
    return bool(result.pvalues.isna().any())


# ── 3. Loop principal: modelos COM e SEM espectro ────────────────────────────

import sys
print("Rodando modelos COM e SEM espectro... (pode demorar ~60s)", flush=True)
rows = []
for dv in dvs:
    df_m = df_final[[dv] + control_cols_full].dropna()
    n = len(df_m)
    y  = pd.to_numeric(df_m[dv], errors='coerce').astype(int)
    if y.nunique() < 2:
        rows.append({'DV':dv}); continue

    classes = np.sort(y.unique())

    # --- COM espectro
    X_full = df_m[control_cols_full].apply(pd.to_numeric, errors='coerce').astype(float)
    const_like = X_full.columns[X_full.nunique() <= 1].tolist()
    X_full = X_full.drop(columns=const_like, errors='ignore')
    r_full  = fit_ol(y, X_full)
    b_econ_full,  p_econ_full  = beta_p(r_full, 'econ')
    b_esp_full,   p_esp_full   = beta_p(r_full, ESPECTRO)
    sing_full = is_singular(r_full)

    # --- SEM espectro
    X_no_esp = df_m[[c for c in control_cols_sem_esp if c in df_m.columns]].apply(pd.to_numeric, errors='coerce').astype(float)
    const_like2 = X_no_esp.columns[X_no_esp.nunique() <= 1].tolist()
    X_no_esp = X_no_esp.drop(columns=const_like2, errors='ignore')
    r_no_esp = fit_ol(y, X_no_esp)
    b_econ_no_esp, p_econ_no_esp = beta_p(r_no_esp, 'econ')
    sing_no_esp = is_singular(r_no_esp)

    # --- Contrafactual (com espectro, para consistência com notebook)
    media_e0 = media_e1 = media_cf = np.nan
    if not sing_full and r_full is not None and 'econ' in X_full.columns:
        mask_e0 = (X_full['econ'] == 0)
        mask_e1 = (X_full['econ'] == 1)
        media_e0 = float(y[mask_e0].mean()) if mask_e0.any() else np.nan
        media_e1 = float(y[mask_e1].mean()) if mask_e1.any() else np.nan
        if mask_e1.any():
            X_cf = X_full.loc[mask_e1].copy()
            X_cf['econ'] = 0.0
            probs = r_full.predict(X_cf)
            media_cf = float(np.mean(probs @ classes.astype(float)))

    rows.append({
        'DV': dv,
        'Bloc': BLOCO.get(dv, 'Outro'),
        'N': n,
        # COM espectro
        'Econ_β_full':   round(b_econ_full, 4) if not np.isnan(b_econ_full) else np.nan,
        'Econ_p_full':   round(p_econ_full, 4) if not np.isnan(p_econ_full) else np.nan,
        'Esp_β':         round(b_esp_full, 4)  if not np.isnan(b_esp_full)  else np.nan,
        'Esp_p':         round(p_esp_full, 4)  if not np.isnan(p_esp_full)  else np.nan,
        'Singular_full': sing_full,
        # SEM espectro
        'Econ_β_no_esp': round(b_econ_no_esp, 4) if not np.isnan(b_econ_no_esp) else np.nan,
        'Econ_p_no_esp': round(p_econ_no_esp, 4) if not np.isnan(p_econ_no_esp) else np.nan,
        'Singular_no_esp': sing_no_esp,
        # Médias
        'Media_e0': round(media_e0, 3) if not np.isnan(media_e0) else np.nan,
        'Media_e1': round(media_e1, 3) if not np.isnan(media_e1) else np.nan,
        'Media_CF': round(media_cf, 3) if not np.isnan(media_cf) else np.nan,
    })

df_rob = pd.DataFrame(rows)
print(f"Loop concluído. Itens: {len(df_rob)}", flush=True)


# ── 4. BH sobre os dois conjuntos de p-valores ───────────────────────────────

def apply_bh(series, alpha=0.05):
    valid = pd.to_numeric(series, errors='coerce')
    mask  = valid.notna()
    out   = pd.Series(np.nan, index=series.index)
    if mask.sum() == 0: return out
    _, p_bh, _, _ = multipletests(valid[mask].values, alpha=alpha, method='fdr_bh')
    out[mask] = p_bh
    return out.round(4)

df_rob['Econ_p_BH_full']   = apply_bh(df_rob['Econ_p_full'])
df_rob['Econ_p_BH_no_esp'] = apply_bh(df_rob['Econ_p_no_esp'])
df_rob['Esp_p_BH']         = apply_bh(df_rob['Esp_p'])


# ── 5. Resumo por bloc ───────────────────────────────────────────────────────

def sig_count(df, col_p, alpha=0.05):
    return int(pd.to_numeric(df[col_p], errors='coerce').lt(alpha).sum())

print("\n=== RESUMO POR BLOC ===", flush=True)
for bloc in sorted(df_rob['Bloc'].unique()):
    sub = df_rob[df_rob['Bloc'] == bloc]
    n_items   = len(sub)
    n_sing    = sub['Singular_full'].sum()
    n_valid   = n_items - n_sing
    esp_nom   = sig_count(sub[~sub['Singular_full']], 'Esp_p')
    esp_bh    = sig_count(sub[~sub['Singular_full']], 'Esp_p_BH')
    econ_nom  = sig_count(sub[~sub['Singular_full']], 'Econ_p_full')
    econ_bh   = sig_count(sub[~sub['Singular_full']], 'Econ_p_BH_full')
    econ_raw  = sig_count(sub[~sub['Singular_no_esp']], 'Econ_p_no_esp')
    print(f"  {bloc:<25} itens={n_items:2d} (sing={n_sing}) | "
          f"Esp nom={esp_nom}/{n_valid} BH={esp_bh}/{n_valid} | "
          f"Econ adj nom={econ_nom}/{n_valid} BH={econ_bh}/{n_valid} | "
          f"Econ bruto (sem esp) nom={econ_raw}")

print("\n=== TOTAIS ===")
valid_all = df_rob[~df_rob['Singular_full']]
print(f"  Espectro p<0.05 nominal: {sig_count(valid_all,'Esp_p')} | BH: {sig_count(valid_all,'Esp_p_BH')}")
print(f"  Econ p<0.05 nominal (adj): {sig_count(valid_all,'Econ_p_full')} | BH: {sig_count(valid_all,'Econ_p_BH_full')}")
valid_no = df_rob[~df_rob['Singular_no_esp']]
print(f"  Econ p<0.05 nominal (bruto, sem esp): {sig_count(valid_no,'Econ_p_no_esp')} | BH: {sig_count(df_rob,'Econ_p_BH_no_esp')}")


# ── 6. AME para os itens-chave ───────────────────────────────────────────────

print("\n=== AME (Average Marginal Effect) de econ ===", flush=True)
KEY_ITEMS = ['déficit_federal_grande','gasto_ajuda_externa',
             'produtividade_está_aumentando','produtos_importados_benéficos']

def compute_ame(dv, control_cols):
    df_m = df_final[[dv] + control_cols].dropna()
    y    = pd.to_numeric(df_m[dv], errors='coerce').astype(int)
    if y.nunique() < 2: return None
    classes = np.sort(y.unique())
    X = df_m[control_cols].apply(pd.to_numeric, errors='coerce').astype(float)
    const_like = X.columns[X.nunique() <= 1].tolist()
    X = X.drop(columns=const_like, errors='ignore')
    if 'econ' not in X.columns: return None
    result = fit_ol(y, X)
    if result is None or is_singular(result): return None

    # AME via diferenças finitas: prever com econ=0 e econ=1
    X1 = X.copy(); X1['econ'] = 1.0
    X0 = X.copy(); X0['econ'] = 0.0
    p1 = result.predict(X1)   # (N, J)
    p0 = result.predict(X0)
    dp = (p1 - p0).mean(axis=0)   # AME por categoria
    E1 = float(np.mean(p1 @ classes.astype(float)))  # E[Y|econ=1]
    E0 = float(np.mean(p0 @ classes.astype(float)))  # E[Y|econ=0]
    return {'categories': classes, 'ame_per_cat': dp, 'E_econ1': E1, 'E_econ0': E0}

ame_rows = []
for dv in KEY_ITEMS:
    res = compute_ame(dv, control_cols_full)
    if res is None:
        print(f"  {dv}: não estimável")
        continue
    cats   = res['categories']
    dp     = res['ame_per_cat']
    E1, E0 = res['E_econ1'], res['E_econ0']
    print(f"\n  DV: {dv}")
    print(f"    E[Y|econ=1]={E1:.3f}  E[Y|econ=0]={E0:.3f}  Δ={E1-E0:.3f}")
    for j, (cat, d) in enumerate(zip(cats, dp)):
        print(f"    AME P(Y={cat}): {d:+.4f}  ({d*100:+.1f} pp)")
    ame_rows.append({'DV': dv, 'E_econ0': round(E0,3), 'E_econ1': round(E1,3),
                     **{f'AME_cat{c}': round(d,4) for c,d in zip(cats, dp)}})

df_ame = pd.DataFrame(ame_rows)


# ── 7. Bootstrap CIs para os 4 itens ─────────────────────────────────────────

print("\n=== Bootstrap (B=200) para econ β — 4 itens-chave ===", flush=True)
np.random.seed(42)
B = 200  # 200 suficiente para IC percentil com n≈180; 1000 causava timeout

boot_rows = []
for dv in KEY_ITEMS:
    df_m = df_final[[dv] + control_cols_full].dropna()
    y  = pd.to_numeric(df_m[dv], errors='coerce').astype(int)
    if y.nunique() < 2: continue
    X  = df_m[control_cols_full].apply(pd.to_numeric, errors='coerce').astype(float)
    const_like = X.columns[X.nunique() <= 1].tolist()
    X  = X.drop(columns=const_like, errors='ignore')
    if 'econ' not in X.columns: continue
    n  = len(y)

    betas = []
    for b in range(B):
        idx = np.random.choice(n, n, replace=True)
        yb  = y.iloc[idx].reset_index(drop=True)
        Xb  = X.iloc[idx].reset_index(drop=True)
        if yb.nunique() < 2 or Xb['econ'].nunique() < 2:
            continue
        rb = fit_ol_fast(yb, Xb)  # rápido: só lbfgs, sem fallback
        if rb is not None and not is_singular(rb) and 'econ' in rb.params.index:
            betas.append(float(rb.params['econ']))

    if not betas:
        print(f"  {dv}: bootstrap falhou")
        continue

    betas = np.array(betas)
    lo, hi = np.percentile(betas, [2.5, 97.5])
    med = np.median(betas)
    prop_neg = (betas < 0).mean()
    prop_pos = (betas > 0).mean()

    # p-value via proporção de zeros cruzados
    p_boot = 2 * min(prop_neg, prop_pos)

    print(f"\n  DV: {dv}")
    print(f"    Bootstrap B={len(betas)} amostras válidas de {B}", flush=True)
    print(f"    β_econ mediana={med:+.4f}  IC95=[{lo:+.4f}, {hi:+.4f}]")
    print(f"    Fração β<0: {prop_neg:.2%}  β>0: {prop_pos:.2%}  p_boot={p_boot:.4f}")

    boot_rows.append({
        'DV': dv, 'B_validos': len(betas),
        'beta_mediana': round(med,4), 'IC95_lo': round(lo,4), 'IC95_hi': round(hi,4),
        'frac_neg': round(prop_neg,3), 'frac_pos': round(prop_pos,3), 'p_boot': round(p_boot,4)
    })

df_boot = pd.DataFrame(boot_rows)


# ── 8. Salvar outputs ────────────────────────────────────────────────────────

out_dir = '/home/bruno_tcc/tcc_esag/Textuais/analise'
df_rob.to_csv(f'{out_dir}/tabela_robustez.csv', index=False, sep=';')
df_ame.to_csv(f'{out_dir}/tabela_ame.csv', index=False, sep=';')
df_boot.to_csv(f'{out_dir}/tabela_bootstrap.csv', index=False, sep=';')

print(f"\n=== Arquivos salvos em {out_dir} ===")
print("  tabela_robustez.csv  — modelos COM e SEM espectro + BH + médias")
print("  tabela_ame.csv       — AME por categoria para 4 itens-chave")
print("  tabela_bootstrap.csv — Bootstrap CIs para 4 itens-chave")

# ── 9. Tabela-resumo por bloc (para o relatório) ─────────────────────────────

print("\n=== TABELA RESUMO POR BLOC (para defesa) ===")
print(f"{'Hipótese':<25} {'N':>4} {'Sing':>4} | {'Esp nom':>7} {'Esp BH':>6} | {'Econ adj nom':>12} {'Econ adj BH':>11} | {'Econ bruto nom':>14}")
print("-" * 95)
for bloc in sorted(df_rob['Bloc'].unique()):
    sub = df_rob[df_rob['Bloc'] == bloc]
    n_items  = len(sub)
    n_sing   = int(sub['Singular_full'].sum())
    n_valid  = n_items - n_sing
    if n_valid == 0:
        print(f"  {bloc:<25} {n_items:>4} {n_sing:>4} | {'—':>7} {'—':>6} | {'—':>12} {'—':>11} | {'—':>14}")
        continue
    sub_v = sub[~sub['Singular_full']]
    sub_nv = sub[~sub['Singular_no_esp']]
    esp_n = sig_count(sub_v, 'Esp_p')
    esp_b = sig_count(sub_v, 'Esp_p_BH')
    ec_n  = sig_count(sub_v, 'Econ_p_full')
    ec_b  = sig_count(sub_v, 'Econ_p_BH_full')
    ec_raw= sig_count(sub_nv, 'Econ_p_no_esp')
    print(f"  {bloc:<25} {n_items:>4} {n_sing:>4} | {esp_n:>3}/{n_valid:<3} {esp_b:>2}/{n_valid:<3} | {ec_n:>5}/{n_valid:<6} {ec_b:>4}/{n_valid:<6} | {ec_raw:>6}/{len(sub_nv):<7}")
