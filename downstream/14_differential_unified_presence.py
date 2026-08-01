import os, sys, glob, time, warnings, pickle, json, threading, traceback
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from collections import defaultdict
from itertools import combinations
from scipy.stats import fisher_exact
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

warnings.filterwarnings("ignore", category=FutureWarning)

# ================================================================
#  CONFIGURATION
# ================================================================

BASE_DIR = "/path/to/data"

PREFIXES = [
    "Dopaminergic", "MSN", "FOXP2", "PVALB", "VIP", "CNGA",
    "CRC01","CRC02","CRC03","CRC04","CRC05","CRC06","CRC07",
    "CRC08","CRC10","CRC11","CRC12","CRC13","CRC14","CRC15",
]

DATA_SOURCES = [
    {"name":"enhancer", "suffix":"_sorted_scBAMs_enhancer_csv",
     "pattern":"*_overlaps_enhancer_summary.csv","tag":"enhancer","is_bed":False,"type_col":None},
    {"name":"gtf",      "suffix":"_sorted_scBAMs_GTF_csv",
     "pattern":"*_overlaps_GTF_summary.csv",     "tag":"gene_body","is_bed":False,"type_col":None},
    {"name":"promoter", "suffix":"_sorted_scBAMs_PROMOTER_csv",
     "pattern":"*_overlaps_PROMOTER_summary.csv","tag":"promoter","is_bed":False,"type_col":None},
    {"name":"regulome", "suffix":"_sorted_scBAMs_REGULOME_csv",
     "pattern":"*_overlaps_REGULOME_summary.csv","tag":None,"is_bed":False,"type_col":"Regulome_type"},
    {"name":"unknown",  "suffix":"_sorted_scBAMs_UNKNOWN_TFBS",
     "pattern":"*_BD_uncharacterized.bed",       "tag":"uncharacterized","is_bed":True,"type_col":None},
]
SRC_BY_NAME = {s["name"]: s for s in DATA_SOURCES}

MIN_CELLS_GLOBAL = 10
PSEUDOCOUNT      = 0.1
FDR_CUTS         = (0.05, 0.01, 0.001)
TOP_N_HEATMAP    = 25
N_WORKERS        = min(96, mp.cpu_count())

MONITOR_LOG = os.path.join(BASE_DIR, "14_unified_monitor.log")
GLOBAL_CKPT = os.path.join(BASE_DIR, "14_unified_global_checkpoint.json")

TYPE_COLORS = {
    "enhancer":"#3498db","silencer":"#2ecc71","promoter":"#f39c12",
    "gene_body":"#9b59b6","uncharacterized":"#e74c3c",
    "insulator":"#1abc9c","regulome":"#34495e",
}
def _tc(rt):
    for k,v in TYPE_COLORS.items():
        if k in rt: return v
    return "#95a5a6"

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.size":10,"axes.labelsize":12,"axes.titlesize":14,
    "xtick.labelsize":9,"ytick.labelsize":9,"legend.fontsize":9,
    "figure.dpi":150,"savefig.dpi":300,"savefig.bbox":"tight",
    "pdf.fonttype":42,"ps.fonttype":42,
})

# ================================================================
#  MONITORING
# ================================================================

def _mem_gb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])/(1024*1024)
    except: pass
    return 0.0

def _total_mem_gb():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1])/(1024*1024)
    except: pass
    return 0.0

_monitor_state = {"prefix":"","step":"idle","detail":"","stop":False}

def _monitor_thread():
    while not _monitor_state["stop"]:
        try:
            mem = _mem_gb()
            avail = _total_mem_gb()
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            line = (f"[{ts}] prefix={_monitor_state['prefix']} "
                    f"step={_monitor_state['step']} "
                    f"detail={_monitor_state['detail']} "
                    f"RSS={mem:.1f}GB avail={avail:.1f}GB\n")
            with open(MONITOR_LOG,"a") as f:
                f.write(line)
        except: pass
        time.sleep(120)

mon = threading.Thread(target=_monitor_thread, daemon=True)
mon.start()

def log_step(prefix, step, detail=""):
    _monitor_state["prefix"] = prefix
    _monitor_state["step"]   = step
    _monitor_state["detail"] = detail
    mem = _mem_gb(); avail = _total_mem_gb()
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    msg = (f"[{ts}] [{prefix}] STEP={step} {detail} "
           f"| RSS={mem:.1f}GB avail={avail:.1f}GB")
    print(msg, flush=True)
    with open(MONITOR_LOG,"a") as f:
        f.write(msg+"\n")

# ================================================================
#  CHECKPOINT / RESUME
# ================================================================

def load_global_ckpt():
    if os.path.exists(GLOBAL_CKPT):
        with open(GLOBAL_CKPT) as f:
            return json.load(f)
    return {}

def save_global_ckpt(data):
    with open(GLOBAL_CKPT,"w") as f:
        json.dump(data, f, indent=2)

def ckpt_path(out_dir, step):
    return os.path.join(out_dir, f".ckpt_{step}.pkl")

def step_done(out_dir, step):
    return os.path.exists(ckpt_path(out_dir, step))

def save_step(out_dir, step, data):
    with open(ckpt_path(out_dir, step),"wb") as f:
        pickle.dump(data, f, protocol=4)

def load_step(out_dir, step):
    with open(ckpt_path(out_dir, step),"rb") as f:
        return pickle.load(f)

# ================================================================
#  HELPERS
# ================================================================

def cell_id(fp):
    b = os.path.splitext(os.path.basename(fp))[0]
    i = b.find("_BD_")
    return b[:i] if i>0 else b

def rlabel(c,s,e):  return f"{c}:{s}-{e}"
def rlabels(df):     return df["Chromosome"].astype(str)+":"+df["Start"].astype(str)+"-"+df["End"].astype(str)

def _simplify_type(raw_type):
    """Map 283 compound region types to ~7 major categories for clean plotting."""
    rt = raw_type.lower()
    if "uncharacterized" in rt: return "uncharacterized"
    # Check insulator/enhancer_blocking BEFORE enhancer (substring overlap)
    if "insulator" in rt or "enhancer_blocking" in rt: return "insulator"
    if "enhancer" in rt:        return "enhancer"
    if "silencer" in rt:        return "silencer"
    if "promoter" in rt:        return "promoter"
    if "gene_body" in rt:       return "gene_body"
    return "other_regulatory"

# Simplified color map (only 7 categories)
SIMPLE_TYPE_COLORS = {
    "enhancer":         "#3498db",
    "silencer":         "#2ecc71",
    "promoter":         "#f39c12",
    "gene_body":        "#9b59b6",
    "uncharacterized":  "#e74c3c",
    "insulator":        "#1abc9c",
    "other_regulatory": "#95a5a6",
}
def _stc(raw_type):
    return SIMPLE_TYPE_COLORS.get(_simplify_type(raw_type), "#95a5a6")

def bh_fdr(pv):
    p=np.asarray(pv,dtype=float); n=len(p)
    if n==0: return np.array([])
    o=np.argsort(p); adj=p[o]*n/np.arange(1,n+1,dtype=float)
    adj=np.minimum.accumulate(adj[::-1])[::-1]
    adj=np.clip(adj,0.0,1.0); out=np.empty(n); out[o]=adj
    return out

def _read_csv_regions(fp):
    try:
        df=pd.read_csv(fp,usecols=["Chromosome","Start","End"],low_memory=False,dtype={"Chromosome":str})
    except: return set()
    if df.empty: return set()
    df["Start"]=pd.to_numeric(df["Start"],errors="coerce")
    df["End"]=pd.to_numeric(df["End"],errors="coerce")
    df=df.dropna(subset=["Start","End"])
    if df.empty: return set()
    return set(zip(df["Chromosome"],df["Start"].astype(int),df["End"].astype(int)))

def _read_regulome(fp):
    try: df=pd.read_csv(fp,low_memory=False,dtype={"Chromosome":str})
    except: return {}
    if df.empty or "Chromosome" not in df.columns: return {}
    ht="Regulome_type" in df.columns
    df["Start"]=pd.to_numeric(df["Start"],errors="coerce")
    df["End"]=pd.to_numeric(df["End"],errors="coerce")
    df=df.dropna(subset=["Chromosome","Start","End"])
    if df.empty: return {}
    out={}
    if ht:
        for c,s,e,t in zip(df["Chromosome"],df["Start"].astype(int),df["End"].astype(int),df["Regulome_type"].fillna("regulome")):
            ts=str(t).strip(); out[(c,s,e)]=ts if ts not in ("","nan","None") else "regulome"
    else:
        for c,s,e in zip(df["Chromosome"],df["Start"].astype(int),df["End"].astype(int)):
            out[(c,s,e)]="regulome"
    return out

def _read_bed_peaks(fp):
    try: df=pd.read_csv(fp,sep="\t",header=None,comment="#",usecols=[6,7,8],engine="c",dtype={6:str})
    except: return set()
    if df.empty: return set()
    df.columns=["Chromosome","Start","End"]
    df["Start"]=pd.to_numeric(df["Start"],errors="coerce")
    df["End"]=pd.to_numeric(df["End"],errors="coerce")
    df=df.dropna()
    if df.empty: return set()
    return set(zip(df["Chromosome"],df["Start"].astype(int),df["End"].astype(int)))

def _read_source(fp, src):
    if src["is_bed"]:   return {r:src["tag"] for r in _read_bed_peaks(fp)}
    if src["type_col"]: return _read_regulome(fp)
    return {r:src["tag"] for r in _read_csv_regions(fp)}

def _save(fig,stem):
    for ext in ("png","pdf"):
        try:
            fig.savefig(f"{stem}.{ext}",dpi=300,bbox_inches="tight")
        except ValueError as e:
            if "too large" in str(e):
                # Fallback: save at lower DPI
                try:
                    fig.savefig(f"{stem}.{ext}",dpi=100,bbox_inches="tight")
                except Exception:
                    pass
            else:
                raise
    plt.close(fig)

def _smart_legend(ax, title=None, fontsize=7, title_fontsize=8, max_per_col=6, outside=True, **kwargs):
    """Place legend below the plot in multiple columns when entries exceed max_per_col."""
    handles, labels = ax.get_legend_handles_labels()
    n = len(handles)
    if n == 0:
        return
    ncol = max(1, (n + max_per_col - 1) // max_per_col)  # ceiling division
    if outside:
        ax.legend(handles, labels, title=title, fontsize=fontsize, title_fontsize=title_fontsize,
                  ncol=ncol, loc="upper center", bbox_to_anchor=(0.5, -0.08),
                  framealpha=0.9, edgecolor="#bdc3c7", columnspacing=1.0,
                  handletextpad=0.4, borderpad=0.4, **kwargs)
    else:
        ax.legend(handles, labels, title=title, fontsize=fontsize, title_fontsize=title_fontsize,
                  ncol=ncol, loc="best", framealpha=0.9, edgecolor="#bdc3c7",
                  columnspacing=1.0, handletextpad=0.4, borderpad=0.4, **kwargs)

# --- parallel worker: process one cell across all its source files ---
def _process_cell(args):
    """Worker: read all sources for one cell, return (cell_id, unified_regions_dict)."""
    cid, src_files = args
    unified = {}  # region -> set of type tags
    for sname, fp in src_files.items():
        rmap = _read_source(fp, SRC_BY_NAME[sname])
        for reg, tag in rmap.items():
            if reg not in unified:
                unified[reg] = set()
            unified[reg].add(tag)
    return cid, unified

_GLOBAL_FILT_SET = None  # module-level shared set (avoids pickling per worker)

def _init_filt_set(fs):
    global _GLOBAL_FILT_SET
    _GLOBAL_FILT_SET = fs

def _process_cell_filtered(args):
    """Worker for pass 2: return (cell_id, set of filtered regions present)."""
    cid, src_files = args
    regs = set()
    for sname, fp in src_files.items():
        rmap = _read_source(fp, SRC_BY_NAME[sname])
        for reg in rmap:
            if reg in _GLOBAL_FILT_SET:
                regs.add(reg)
    return cid, regs

# --- parallel Fisher's exact worker ---
def _fisher_chunk(args):
    """Compute Fisher's exact for a chunk of regions."""
    chunk, na, nb, ct_a_counts, ct_b_counts, pseudo = args
    results = []
    for reg in chunk:
        ca = ct_a_counts.get(reg, 0)
        cb = ct_b_counts.get(reg, 0)
        pa = (ca/na*100) if na else 0.0
        pb = (cb/nb*100) if nb else 0.0
        table = [[ca,cb],[na-ca,nb-cb]]
        oddsratio, pval = fisher_exact(table, alternative="two-sided")
        l2fc = np.log2((pa+pseudo)/(pb+pseudo))
        results.append((reg, ca, pa, cb, pb, pa-pb, l2fc, oddsratio, pval))
    return results

# ================================================================
#  MAIN PIPELINE FOR ONE PREFIX
# ================================================================

def run_prefix(PREFIX):
    t0 = time.time()
    OUT = os.path.join(BASE_DIR, f"{PREFIX}_differential_unified_celltypes")
    PAIR = os.path.join(OUT,"pairwise")
    PLOT = os.path.join(OUT,"plots")
    EXCL = os.path.join(OUT,"exclusive_unified_per_celltype")
    SIM  = os.path.join(OUT,"Similarity")
    for d in (OUT,PAIR,PLOT,EXCL,SIM): os.makedirs(d,exist_ok=True)

    # ---- STEP 1: DISCOVER ----
    log_step(PREFIX,"discover","scanning sources")

    src_cts={}
    for src in DATA_SOURCES:
        root=os.path.join(BASE_DIR,f"{PREFIX}{src['suffix']}")
        if os.path.isdir(root):
            src_cts[src["name"]]=set(d for d in os.listdir(root) if os.path.isdir(os.path.join(root,d)))
        else:
            src_cts[src["name"]]=set()

    all_cts=sorted(set().union(*src_cts.values()))
    if len(all_cts)<2:
        log_step(PREFIX,"SKIP",f"only {len(all_cts)} celltypes")
        return

    log_step(PREFIX,"discover",f"{len(all_cts)} celltypes found")

    # ---- STEP 2: PASS 1 — global census (parallel) ----
    if not step_done(OUT,"pass1"):
        log_step(PREFIX,"pass1","building cell file map + parallel read")

        global_counts   = defaultdict(int)
        region_type_map = defaultdict(set)
        cells_per_ct    = defaultdict(set)

        for ct in all_cts:
            ct_cell_files = defaultdict(dict)
            for src in DATA_SOURCES:
                d = os.path.join(BASE_DIR,f"{PREFIX}{src['suffix']}",ct)
                if not os.path.isdir(d): continue
                for fp in glob.glob(os.path.join(d,src["pattern"])):
                    ct_cell_files[cell_id(fp)][src["name"]] = fp

            log_step(PREFIX,"pass1",f"{ct}: {len(ct_cell_files)} cells — reading")

            tasks = list(ct_cell_files.items())

            with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
                futures = {pool.submit(_process_cell, (cid,srcs)): cid for cid,srcs in tasks}
                for fut in as_completed(futures):
                    cid, unified = fut.result()
                    cells_per_ct[ct].add(cid)
                    for reg, tags in unified.items():
                        global_counts[reg] += 1
                        region_type_map[reg].update(tags)

            log_step(PREFIX,"pass1",f"{ct}: done — {len(cells_per_ct[ct])} cells")

        celltype_ncells = {ct:len(ids) for ct,ids in cells_per_ct.items()}
        celltypes = sorted(celltype_ncells.keys())

        save_step(OUT,"pass1",{
            "global_counts":dict(global_counts),
            "region_type_map":{k:v for k,v in region_type_map.items()},
            "celltype_ncells":celltype_ncells,
            "celltypes":celltypes,
        })
        log_step(PREFIX,"pass1",f"DONE — {len(global_counts):,} regions, {sum(celltype_ncells.values()):,} cells")
    else:
        log_step(PREFIX,"pass1","RESUMING from checkpoint")
        d = load_step(OUT,"pass1")
        global_counts   = defaultdict(int, d["global_counts"])
        region_type_map = defaultdict(set, d["region_type_map"])
        celltype_ncells = d["celltype_ncells"]
        celltypes       = d["celltypes"]
        all_cts         = celltypes

    # ---- STEP 3: FILTER ----
    log_step(PREFIX,"filter",f"MIN_CELLS_GLOBAL={MIN_CELLS_GLOBAL}")
    filtered = frozenset(r for r,c in global_counts.items() if c>=MIN_CELLS_GLOBAL)
    n_filt = len(filtered)
    log_step(PREFIX,"filter",f"kept {n_filt:,}, dropped {len(global_counts)-n_filt:,}")
    if n_filt == 0:
        log_step(PREFIX,"SKIP","no regions survived filter")
        return

    filt_rows=[]
    for r in sorted(filtered):
        filt_rows.append({"Chromosome":r[0],"Start":r[1],"End":r[2],
                          "Region_Type":";".join(sorted(region_type_map[r])),
                          "Global_Count":global_counts[r]})
    pd.DataFrame(filt_rows).to_csv(os.path.join(OUT,f"filtered_regions_ge{MIN_CELLS_GLOBAL}.csv"),index=False)

    # ---- STEP 4: PASS 2 — per-celltype counts (parallel) ----
    if not step_done(OUT,"pass2"):
        log_step(PREFIX,"pass2","parallel read for filtered regions")
        ct_counts = {ct:defaultdict(int) for ct in celltypes}

        filt_set = set(filtered)

        for ct in celltypes:
            ct_cell_files = defaultdict(dict)
            for src in DATA_SOURCES:
                d = os.path.join(BASE_DIR,f"{PREFIX}{src['suffix']}",ct)
                if not os.path.isdir(d): continue
                for fp in glob.glob(os.path.join(d,src["pattern"])):
                    ct_cell_files[cell_id(fp)][src["name"]] = fp

            log_step(PREFIX,"pass2",f"{ct}: {len(ct_cell_files)} cells")

            tasks = [(cid,srcs) for cid,srcs in ct_cell_files.items()]

            # Use initializer to share filt_set once per worker (not per task)
            with ProcessPoolExecutor(max_workers=N_WORKERS,
                                     initializer=_init_filt_set,
                                     initargs=(filt_set,)) as pool:
                futures = [pool.submit(_process_cell_filtered, t) for t in tasks]
                for fut in as_completed(futures):
                    cid, regs = fut.result()
                    for reg in regs:
                        ct_counts[ct][reg] += 1

            n_active = sum(1 for v in ct_counts[ct].values() if v>0)
            log_step(PREFIX,"pass2",f"{ct}: {n_active:,} active regions")

        save_step(OUT,"pass2",{ct:dict(d) for ct,d in ct_counts.items()})
        log_step(PREFIX,"pass2","DONE")
    else:
        log_step(PREFIX,"pass2","RESUMING from checkpoint")
        ct_counts_raw = load_step(OUT,"pass2")
        ct_counts = {ct:defaultdict(int,d) for ct,d in ct_counts_raw.items()}

    # ---- STEP 5: MASTER TABLE ----
    if not step_done(OUT,"master"):
        log_step(PREFIX,"master","building")
        region_list = sorted(filtered)
        rows=[]
        for reg in region_list:
            c,s,e=reg; types=sorted(region_type_map[reg])
            row={"Chromosome":c,"Start":s,"End":e,
                 "Region_Type":";".join(types),"All_Types":";".join(types)}
            for ct in celltypes:
                cnt=ct_counts[ct].get(reg,0); nc=celltype_ncells[ct]
                row[f"Count_{ct}"]=cnt; row[f"Pct_{ct}"]=(cnt/nc*100) if nc else 0.0
            rows.append(row)

        master=pd.DataFrame(rows)
        pct_cols=[f"Pct_{ct}" for ct in celltypes]
        master["Max_pct"]=master[pct_cols].max(axis=1)
        master["Min_pct"]=master[pct_cols].min(axis=1)
        master["Diff_pct"]=master["Max_pct"]-master["Min_pct"]
        master["log2FC"]=np.log2((master["Max_pct"]+PSEUDOCOUNT)/(master["Min_pct"]+PSEUDOCOUNT))
        master["Max_Celltype"]=master[pct_cols].idxmax(axis=1).str.replace("Pct_","")
        def _excl(row):
            p=[ct for ct in celltypes if row[f"Count_{ct}"]>0]
            return p[0] if len(p)==1 else ""
        master["ExclusiveCelltype"]=master.apply(_excl,axis=1)
        master.sort_values("Diff_pct",ascending=False,inplace=True)
        master.to_csv(os.path.join(OUT,"unified_master_table_celltypes.csv"),index=False)
        save_step(OUT,"master",True)
        log_step(PREFIX,"master",f"DONE — {len(master):,} regions")
    else:
        log_step(PREFIX,"master","RESUMING — reloading CSV")
        master=pd.read_csv(os.path.join(OUT,"unified_master_table_celltypes.csv"),low_memory=False)
        pct_cols=[f"Pct_{ct}" for ct in celltypes]

    region_list = sorted(filtered)

    # ---- STEP 6: PAIRWISE TESTING (serial Fisher's — avoids costly dict serialization) ----
    if not step_done(OUT,"pairwise"):
        log_step(PREFIX,"pairwise","Fisher's exact + BH FDR")
        pairs=list(combinations(celltypes,2))
        log_step(PREFIX,"pairwise",f"{len(pairs)} pairs, {n_filt:,} regions each")

        pair_results={}
        all_pvals=[]
        pair_sizes=[]

        for pi,(ct_a,ct_b) in enumerate(pairs):
            log_step(PREFIX,"pairwise",f"[{pi+1}/{len(pairs)}] {ct_a} vs {ct_b}")
            na=celltype_ncells[ct_a]; nb=celltype_ncells[ct_b]
            ct_a_d=ct_counts[ct_a]; ct_b_d=ct_counts[ct_b]

            prows=[]
            pvals=[]
            for reg in region_list:
                c,s,e=reg
                ca=ct_a_d.get(reg,0); cb=ct_b_d.get(reg,0)
                pa=(ca/na*100) if na else 0.0
                pb=(cb/nb*100) if nb else 0.0
                table=[[ca,cb],[na-ca,nb-cb]]
                odds,pval=fisher_exact(table,alternative="two-sided")
                l2fc=np.log2((pa+PSEUDOCOUNT)/(pb+PSEUDOCOUNT))
                prows.append({
                    "Chromosome":c,"Start":s,"End":e,
                    "Region_Type":";".join(sorted(region_type_map[reg])),
                    f"Count_{ct_a}":ca,f"Pct_{ct_a}":pa,
                    f"Count_{ct_b}":cb,f"Pct_{ct_b}":pb,
                    f"nCells_{ct_a}":na,f"nCells_{ct_b}":nb,
                    "Diff_pct":pa-pb,"log2FC":l2fc,
                    "odds_ratio":odds,"p_value":pval,
                })
                pvals.append(pval)

            fdr=bh_fdr(np.array(pvals))
            pdf=pd.DataFrame(prows)
            pdf["FDR"]=fdr
            pdf["neg_log10_FDR"]=-np.log10(pdf["FDR"].clip(lower=1e-300))

            all_pvals.extend(pvals)
            pair_sizes.append(len(pvals))
            pair_results[(ct_a,ct_b)]=pdf

            ns005=(fdr<0.05).sum()
            log_step(PREFIX,"pairwise",f"  {ct_a} vs {ct_b}: {ns005:,} sig FDR<0.05")

        # global FDR
        log_step(PREFIX,"pairwise","computing global FDR")
        gfdr=bh_fdr(np.array(all_pvals))
        offset=0
        for (ct_a,ct_b),sz in zip(pairs,pair_sizes):
            df=pair_results[(ct_a,ct_b)]
            df["global_FDR"]=gfdr[offset:offset+sz]
            offset+=sz
            df["sig_FDR005"]=df["FDR"]<0.05
            df["sig_FDR001"]=df["FDR"]<0.01
            df["sig_FDR0001"]=df["FDR"]<0.001
            df["direction"]="ns"
            df.loc[(df["FDR"]<0.05)&(df["log2FC"]>0),"direction"]=f"up_in_{ct_a}"
            df.loc[(df["FDR"]<0.05)&(df["log2FC"]<0),"direction"]=f"up_in_{ct_b}"
            df.sort_values("FDR",inplace=True)
            df.to_csv(os.path.join(PAIR,f"{ct_a}_vs_{ct_b}.csv"),index=False)

        save_step(OUT,"pairwise",True)
        log_step(PREFIX,"pairwise","DONE — all CSVs saved")
    else:
        log_step(PREFIX,"pairwise","RESUMING — reloading CSVs")
        pairs=list(combinations(celltypes,2))
        pair_results={}
        for ct_a,ct_b in pairs:
            p=os.path.join(PAIR,f"{ct_a}_vs_{ct_b}.csv")
            if os.path.exists(p):
                pair_results[(ct_a,ct_b)]=pd.read_csv(p,low_memory=False)

    # ---- STEP 7: PLOTS ----
    if not step_done(OUT,"plots"):
        log_step(PREFIX,"plots","generating journal-quality figures")
        pairs=list(combinations(celltypes,2))

        # 7a: Volcano per pair — simplified types, full coords, leader lines
        for (ct_a,ct_b),pdf in pair_results.items():
            fig,ax=plt.subplots(figsize=(10,8))
            pdf_plot = pdf.copy()
            pdf_plot["_stype"] = pdf_plot["Region_Type"].apply(_simplify_type)

            # non-significant (grey)
            ns=pdf_plot[pdf_plot["FDR"]>=0.05]
            ax.scatter(ns["log2FC"],ns["neg_log10_FDR"],c="#cccccc",s=6,alpha=0.25,
                       edgecolors="none",zorder=1,rasterized=True)

            # significant, colored by simplified type
            sig=pdf_plot[pdf_plot["FDR"]<0.05].copy()
            for st in ["enhancer","silencer","promoter","gene_body","uncharacterized","insulator","other_regulatory"]:
                m=sig["_stype"]==st
                if m.sum()==0: continue
                ax.scatter(sig.loc[m,"log2FC"],sig.loc[m,"neg_log10_FDR"],
                           c=SIMPLE_TYPE_COLORS[st],s=18,alpha=0.65,edgecolors="k",
                           linewidths=0.2,label=f"{st} ({m.sum():,})",zorder=2,rasterized=True)

            # threshold lines
            for fc,ls in [(0.05,"--"),(0.01,":"),(0.001,"-.")]:
                y=-np.log10(fc)
                ax.axhline(y,color="#7f8c8d",ls=ls,lw=0.7,alpha=0.6)
                ax.text(ax.get_xlim()[1]*0.98 if ax.get_xlim()[1]>0 else 1, y,
                        f" FDR={fc}",va="bottom",ha="right",fontsize=7,color="#7f8c8d")
            ax.axvline(1,color="#7f8c8d",ls="--",lw=0.7,alpha=0.5)
            ax.axvline(-1,color="#7f8c8d",ls="--",lw=0.7,alpha=0.5)

            # quadrant counts
            n_up=((pdf_plot["FDR"]<0.05)&(pdf_plot["log2FC"]>0)).sum()
            n_dn=((pdf_plot["FDR"]<0.05)&(pdf_plot["log2FC"]<0)).sum()
            ax.text(0.02,0.98,f"Up in {ct_a}: {n_up:,}",transform=ax.transAxes,
                    fontsize=9,va="top",color="#c0392b",fontweight="bold")
            ax.text(0.98,0.98,f"Up in {ct_b}: {n_dn:,}",transform=ax.transAxes,
                    fontsize=9,va="top",ha="right",color="#2980b9",fontweight="bold")

            # label top 8 with FULL coordinates + leader lines, radially offset
            if not sig.empty:
                top8=sig.nlargest(8,"neg_log10_FDR")
                angles=np.linspace(30,330,len(top8))
                for idx,(_,r) in enumerate(top8.iterrows()):
                    full_coord=rlabel(r["Chromosome"],int(r["Start"]),int(r["End"]))
                    ang=np.radians(angles[idx])
                    ox=np.cos(ang)*60; oy=np.sin(ang)*40
                    ax.annotate(full_coord,
                                xy=(r["log2FC"],r["neg_log10_FDR"]),
                                xytext=(ox,oy), textcoords="offset points",
                                fontsize=6, fontweight="bold", alpha=0.9,
                                bbox=dict(boxstyle="round,pad=0.2",fc="white",ec="#bdc3c7",alpha=0.85,lw=0.5),
                                arrowprops=dict(arrowstyle="-",lw=0.6,color="#555555",
                                                connectionstyle="arc3,rad=0.15"))

            ax.set_xlabel(r"log$_2$ Fold Change ($\leftarrow$ "+ct_b+r" | "+ct_a+r" $\rightarrow$)",fontsize=12)
            ax.set_ylabel(r"$-\log_{10}$ FDR",fontsize=12)
            ax.set_title(f"Differential Presence: {ct_a} vs {ct_b}\n"
                         f"(n={len(pdf):,} regions, Fisher's exact + BH)",fontsize=13,fontweight="bold")
            _smart_legend(ax, title="Region Type", fontsize=8, title_fontsize=9)
            ax.grid(True,alpha=0.15)
            fig.subplots_adjust(bottom=0.16)
            _save(fig,os.path.join(PLOT,f"volcano_{ct_a}_vs_{ct_b}"))
        log_step(PREFIX,"plots","volcanos done")

        # 7b: MA per pair
        for (ct_a,ct_b),pdf in pair_results.items():
            fig,ax=plt.subplots(figsize=(8,6))
            mp_val=(pdf[f"Pct_{ct_a}"]+pdf[f"Pct_{ct_b}"])/2
            ns_m=pdf["FDR"]>=0.05
            ax.scatter(mp_val[ns_m],pdf.loc[ns_m,"log2FC"],c="#cccccc",s=6,alpha=0.25,edgecolors="none",rasterized=True,zorder=1)
            up_m=(pdf["FDR"]<0.05)&(pdf["log2FC"]>0)
            dn_m=(pdf["FDR"]<0.05)&(pdf["log2FC"]<0)
            ax.scatter(mp_val[up_m],pdf.loc[up_m,"log2FC"],c="#e74c3c",s=12,alpha=0.45,edgecolors="none",rasterized=True,label=f"Up in {ct_a} ({up_m.sum():,})",zorder=2)
            ax.scatter(mp_val[dn_m],pdf.loc[dn_m,"log2FC"],c="#3498db",s=12,alpha=0.45,edgecolors="none",rasterized=True,label=f"Up in {ct_b} ({dn_m.sum():,})",zorder=2)
            ax.axhline(0,color="black",lw=0.8)
            ax.set_xlabel("Mean Presence (%)",fontsize=12)
            ax.set_ylabel(r"log$_2$ Fold Change",fontsize=12)
            ax.set_title(f"MA Plot: {ct_a} vs {ct_b}",fontsize=13,fontweight="bold")
            _smart_legend(ax, fontsize=9, title_fontsize=9)
            ax.grid(True,alpha=0.15)
            fig.subplots_adjust(bottom=0.18)
            _save(fig,os.path.join(PLOT,f"MA_{ct_a}_vs_{ct_b}"))
        log_step(PREFIX,"plots","MA plots done")

        # 7c: Top-N heatmap with significance stars
        log_step(PREFIX,"plots","heatmap with significance stars")

        # Build FDR lookup dicts per pair (vectorized — avoids O(N*M) DataFrame scans)
        pair_fdr_dicts = {}
        for (ca,cb), pdf in pair_results.items():
            fdr_dict = {}
            for c_val, s_val, e_val, fdr_val in zip(
                pdf["Chromosome"], pdf["Start"], pdf["End"], pdf["FDR"]):
                fdr_dict[(c_val, int(s_val), int(e_val))] = fdr_val
            pair_fdr_dicts[(ca,cb)] = fdr_dict
            log_step(PREFIX,"plots",f"  FDR index built for {ca} vs {cb}")

        min_fdr_per_region = {}
        for reg in region_list:
            best = 1.0
            for fdr_dict in pair_fdr_dicts.values():
                fv = fdr_dict.get(reg)
                if fv is not None and fv < best:
                    best = fv
            min_fdr_per_region[reg] = best

        log_step(PREFIX,"plots",f"  min-FDR computed for {len(min_fdr_per_region):,} regions")
        sorted_sig=sorted(min_fdr_per_region.items(),key=lambda x:x[1])
        top_regs=[r for r,_ in sorted_sig[:TOP_N_HEATMAP]]
        top_fdrs=[f for _,f in sorted_sig[:TOP_N_HEATMAP]]
        N_plot=len(top_regs)

        if N_plot>0:
            heat_data=np.zeros((N_plot,len(celltypes)))
            labels_y=[]; stypes_y=[]; stars_y=[]
            for i,reg in enumerate(top_regs):
                c,s,e=reg
                labels_y.append(rlabel(c,s,e))
                stypes_y.append(_simplify_type(";".join(sorted(region_type_map[reg]))))
                fv=top_fdrs[i]
                stars_y.append("***" if fv<0.001 else "**" if fv<0.01 else "*" if fv<0.05 else "")
                for j,ct in enumerate(celltypes):
                    cnt=ct_counts[ct].get(reg,0); nc=celltype_ncells[ct]
                    heat_data[i,j]=(cnt/nc*100) if nc else 0.0

            # Larger figure: more height per row, wider columns
            row_h = 0.55
            fig_w = max(12, len(celltypes)*1.8)
            fig_h = row_h*N_plot + 5
            fig,axes=plt.subplots(1,2,figsize=(fig_w, fig_h),
                                  gridspec_kw={"width_ratios":[1,15]})

            # type sidebar with simplified colors
            ax0=axes[0]
            for i,st in enumerate(stypes_y):
                ax0.add_patch(plt.Rectangle((0,i),1,1,color=SIMPLE_TYPE_COLORS.get(st,"#95a5a6")))
            ax0.set_xlim(0,1);ax0.set_ylim(0,N_plot);ax0.invert_yaxis()
            ax0.set_xticks([]);ax0.set_yticks([]);ax0.set_xlabel("Type",fontsize=10)

            # main heatmap
            ax1=axes[1]
            im=ax1.imshow(heat_data,cmap="YlOrRd",aspect="auto",vmin=0)
            cb=plt.colorbar(im,ax=ax1,shrink=0.5,pad=0.02)
            cb.set_label("Presence (%)",fontsize=11)
            ax1.set_xticks(np.arange(len(celltypes)))
            ax1.set_xticklabels(celltypes,rotation=45,ha="right",fontsize=11,fontweight="bold")
            ax1.xaxis.tick_top();ax1.xaxis.set_label_position("top")
            ax1.set_yticks(np.arange(N_plot))
            ax1.set_yticklabels(labels_y,fontsize=8)  # full coordinates, readable size

            # overlay percentages + stars — larger font
            max_ct_idx=np.argmax(heat_data,axis=1)
            for i in range(N_plot):
                for j in range(len(celltypes)):
                    val=heat_data[i,j]; txt=f"{val:.1f}"
                    if j==max_ct_idx[i] and stars_y[i]: txt+=stars_y[i]
                    ax1.text(j,i,txt,ha="center",va="center",fontsize=7,
                             fontweight="bold" if j==max_ct_idx[i] else "normal")

            ax1.set_title(f"Top {N_plot} Significant Regions (BH FDR)\n"
                          f"* p<0.05  ** p<0.01  *** p<0.001",
                          fontsize=14,fontweight="bold")

            # legend: only simplified types present, multi-column below
            unique_stypes=sorted(set(stypes_y))
            handles=[mpatches.Patch(color=SIMPLE_TYPE_COLORS.get(t,"#95a5a6"),label=t)
                     for t in unique_stypes]
            ncol_ht=max(1,len(handles))  # all in one row since max ~7
            fig.legend(handles=handles,title="Region Type",fontsize=9,title_fontsize=10,
                       ncol=ncol_ht,loc="lower center",bbox_to_anchor=(0.55,-0.01),
                       framealpha=0.9,edgecolor="#bdc3c7",columnspacing=1.5)
            plt.tight_layout()
            fig.subplots_adjust(bottom=0.06)
            _save(fig,os.path.join(PLOT,f"heatmap_top{N_plot}_significance"))
        log_step(PREFIX,"plots","heatmap done")

        # 7d: Region type composition per celltype (SIMPLIFIED types)
        SIMPLE_CATS = ["enhancer","silencer","promoter","gene_body","uncharacterized","insulator","other_regulatory"]

        ct_stype_active = {}
        for ct in celltypes:
            counts = {st:0 for st in SIMPLE_CATS}
            for reg in filtered:
                if ct_counts[ct].get(reg,0) > 0:
                    raw = ";".join(sorted(region_type_map[reg]))
                    counts[_simplify_type(raw)] += 1
            ct_stype_active[ct] = counts

        fig,ax=plt.subplots(figsize=(max(9,len(celltypes)*1.2),7))
        bottoms=np.zeros(len(celltypes))
        for st in SIMPLE_CATS:
            vals=np.array([ct_stype_active[ct][st] for ct in celltypes],dtype=float)
            if vals.sum()==0: continue
            ax.bar(celltypes,vals,bottom=bottoms,label=f"{st} ({int(vals.sum()):,})",
                   color=SIMPLE_TYPE_COLORS[st],edgecolor="white",linewidth=0.3)
            bottoms+=vals
        ax.set_ylabel("Active Regions",fontsize=12)
        ax.set_title(f"{PREFIX}: Region Type Composition per Celltype",fontsize=14,fontweight="bold")
        _smart_legend(ax, title="Region Type", fontsize=9, title_fontsize=10)
        plt.xticks(rotation=45,ha="right",fontsize=10)
        fig.subplots_adjust(bottom=0.22)
        _save(fig,os.path.join(PLOT,"region_type_composition"))

        # 7e: Exclusive regions stacked (SIMPLIFIED types)
        excl_stype_counts = {}
        for ct in celltypes:
            sub=master[master["ExclusiveCelltype"]==ct]
            counts = {st:0 for st in SIMPLE_CATS}
            if not sub.empty:
                for raw_rt in sub["Region_Type"]:
                    counts[_simplify_type(raw_rt)] += 1
                sub.to_csv(os.path.join(EXCL,f"exclusive_{ct}.csv"),index=False)
            excl_stype_counts[ct] = counts

        fig,ax=plt.subplots(figsize=(max(9,len(celltypes)*1.2),7))
        bottoms=np.zeros(len(celltypes))
        for st in SIMPLE_CATS:
            vals=np.array([excl_stype_counts[ct][st] for ct in celltypes],dtype=float)
            if vals.sum()==0: continue
            ax.bar(celltypes,vals,bottom=bottoms,label=f"{st} ({int(vals.sum()):,})",
                   color=SIMPLE_TYPE_COLORS[st],edgecolor="white",linewidth=0.3)
            bottoms+=vals
        ax.set_ylabel("Exclusive Regions",fontsize=12)
        ax.set_title(f"{PREFIX}: Exclusive Regions per Celltype (by Type)",fontsize=14,fontweight="bold")
        _smart_legend(ax, title="Region Type", fontsize=9, title_fontsize=10)
        plt.xticks(rotation=45,ha="right",fontsize=10)
        fig.subplots_adjust(bottom=0.22)
        _save(fig,os.path.join(PLOT,"exclusive_regions_stacked"))

        # 7f: FDR distribution
        all_fdrs=[]
        for pdf in pair_results.values(): all_fdrs.extend(pdf["FDR"].tolist())
        all_fdrs=np.array(all_fdrs)
        fig,ax=plt.subplots(figsize=(8,5))
        ax.hist(all_fdrs[all_fdrs<1.0],bins=100,color="#2c3e50",alpha=0.8,edgecolor="white",linewidth=0.3)
        for fc,ls,col in [(0.05,"--","#e74c3c"),(0.01,":","#e67e22"),(0.001,"-.","#f39c12")]:
            ax.axvline(fc,color=col,ls=ls,lw=1.2,label=f"FDR={fc}")
        ax.set_xlabel("FDR (BH)",fontsize=12);ax.set_ylabel("Count",fontsize=12)
        ax.set_title(f"FDR Distribution ({len(all_fdrs):,} tests, {len(pairs)} pairs)",fontsize=13,fontweight="bold")
        _smart_legend(ax, fontsize=9, outside=False);ax.set_xlim(-0.02,1.02);plt.tight_layout()
        _save(fig,os.path.join(PLOT,"FDR_distribution"))

        # 7g: Sig regions per pair bar
        fig,ax=plt.subplots(figsize=(max(8,len(pairs)*0.8),6))
        pair_labels=[f"{a}\nvs\n{b}" for a,b in pairs]
        n_up=[((pair_results[(a,b)]["FDR"]<0.05)&(pair_results[(a,b)]["log2FC"]>0)).sum() for a,b in pairs]
        n_dn=[((pair_results[(a,b)]["FDR"]<0.05)&(pair_results[(a,b)]["log2FC"]<0)).sum() for a,b in pairs]
        x=np.arange(len(pairs));w=0.35
        b1=ax.bar(x-w/2,n_up,w,label="Up in A",color="#e74c3c",edgecolor="white")
        b2=ax.bar(x+w/2,n_dn,w,label="Up in B",color="#3498db",edgecolor="white")
        ax.set_xticks(x);ax.set_xticklabels(pair_labels,fontsize=7)
        ax.set_ylabel("Significant Regions (FDR<0.05)",fontsize=12)
        ax.set_title(f"{PREFIX}: Sig Differential Regions per Pair",fontsize=13,fontweight="bold")
        _smart_legend(ax, fontsize=9, outside=False)
        for bars in (b1,b2):
            for bar in bars:
                h=bar.get_height()
                if h>0: ax.text(bar.get_x()+bar.get_width()/2,h,f"{int(h):,}",ha="center",va="bottom",fontsize=6)
        plt.tight_layout()
        _save(fig,os.path.join(PLOT,"sig_regions_per_pair"))

        # 7h: Diff_pct by type (simplified)
        master["_stype"] = master["Region_Type"].apply(_simplify_type)
        fig,ax=plt.subplots(figsize=(10,6))
        for st in SIMPLE_CATS:
            sub=master[master["_stype"]==st]
            if sub.empty: continue
            ax.hist(sub["Diff_pct"],bins=60,alpha=0.5,
                    color=SIMPLE_TYPE_COLORS[st],label=f"{st} ({len(sub):,})")
        ax.set_xlabel("Diff_pct (Max% - Min%)",fontsize=12);ax.set_ylabel("Regions",fontsize=12)
        ax.set_title(f"{PREFIX}: Diff_pct Distribution by Region Type",fontsize=14,fontweight="bold")
        _smart_legend(ax, title="Region Type", fontsize=9, title_fontsize=10)
        fig.subplots_adjust(bottom=0.18)
        _save(fig,os.path.join(PLOT,"diff_pct_distribution_by_type"))

        # 7i: Jaccard
        log_step(PREFIX,"plots","Jaccard similarity")
        ct_centroids={ct:{r for r in filtered if ct_counts[ct].get(r,0)>0} for ct in celltypes}
        nct=len(celltypes)
        jac=np.zeros((nct,nct))
        for i in range(nct):
            for j in range(nct):
                if i==j: jac[i,j]=100.0
                else:
                    inter=len(ct_centroids[celltypes[i]]&ct_centroids[celltypes[j]])
                    union=len(ct_centroids[celltypes[i]]|ct_centroids[celltypes[j]])
                    jac[i,j]=(inter/union*100) if union else 0.0
        jac_df=pd.DataFrame(jac,index=celltypes,columns=celltypes)
        jac_df.to_csv(os.path.join(SIM,"cross_celltype_jaccard_unified.csv"))

        fig,ax=plt.subplots(figsize=(max(7,nct*0.9),max(6,nct*0.8)))
        mask=np.eye(nct,dtype=bool)
        vmax=jac_df.where(~mask).max().max()
        if np.isnan(vmax): vmax=100.0
        sns.heatmap(jac_df,cmap="viridis",square=True,vmin=0,vmax=vmax,mask=mask,annot=True,fmt=".1f",cbar_kws={"label":"Jaccard (%)"},ax=ax,linewidths=0.5,linecolor="white")
        for i in range(nct): ax.add_patch(plt.Rectangle((i,i),1,1,color="#e74c3c",lw=0))
        ax.set_title(f"{PREFIX}: Cross-Celltype Jaccard (Unified)",fontsize=13,fontweight="bold")
        plt.tight_layout()
        _save(fig,os.path.join(SIM,"cross_celltype_jaccard_unified"))

        save_step(OUT,"plots",True)
        log_step(PREFIX,"plots","ALL DONE")
    else:
        log_step(PREFIX,"plots","RESUMING — already complete")

    # ---- STEP 8: SUMMARY ----
    if not step_done(OUT,"summary"):
        log_step(PREFIX,"summary","building")
        pairs=list(combinations(celltypes,2))
        summary_rows=[]
        for ca,cb in pairs:
            pdf=pair_results.get((ca,cb))
            if pdf is None: continue
            row={"Comparison":f"{ca}_vs_{cb}","Total_regions":len(pdf),
                 "nCells_A":celltype_ncells[ca],"nCells_B":celltype_ncells[cb]}
            for fc in FDR_CUTS:
                sig=pdf["FDR"]<fc
                row[f"sig_FDR{fc}"]=sig.sum()
                row[f"up_{ca}_FDR{fc}"]=((sig)&(pdf["log2FC"]>0)).sum()
                row[f"up_{cb}_FDR{fc}"]=((sig)&(pdf["log2FC"]<0)).sum()
            row["max_log2FC"]=pdf["log2FC"].max()
            row["min_log2FC"]=pdf["log2FC"].min()
            row["min_FDR"]=pdf["FDR"].min()
            summary_rows.append(row)
        pd.DataFrame(summary_rows).to_csv(os.path.join(OUT,"pairwise_summary.csv"),index=False)

        # Vectorized type breakdown (avoids 283 × 591K × 5 nested loop)
        all_types=sorted(set(t for ts in region_type_map.values() for t in ts))
        tb_rows=[]
        for ct in celltypes:
            row={"Celltype":ct,"nCells":celltype_ncells[ct]}
            type_counts_ct={t:0 for t in all_types}
            for r in filtered:
                if ct_counts[ct].get(r,0)>0:
                    for t in region_type_map[r]:
                        type_counts_ct[t]+=1
            row.update(type_counts_ct)
            tb_rows.append(row)
        pd.DataFrame(tb_rows).to_csv(os.path.join(OUT,"region_type_breakdown_per_celltype.csv"),index=False)

        save_step(OUT,"summary",True)
        log_step(PREFIX,"summary","DONE")
    else:
        log_step(PREFIX,"summary","RESUMING — already complete")

    # ================================================================
    #  STEPS 9-12: REGULATORY-ONLY ANALYSIS (excludes gene_body)
    # ================================================================

    if not step_done(OUT,"reg_only"):
        log_step(PREFIX,"reg_only","starting regulatory-only analysis (no gene_body)")

        REG_DIR = os.path.join(OUT,"regulatory_only")
        REG_PAIR = os.path.join(REG_DIR,"pairwise")
        REG_PLOT = os.path.join(REG_DIR,"plots")
        REG_EXCL = os.path.join(REG_DIR,"exclusive_per_celltype")
        REG_SIM  = os.path.join(REG_DIR,"Similarity")
        for d in (REG_DIR,REG_PAIR,REG_PLOT,REG_EXCL,REG_SIM): os.makedirs(d,exist_ok=True)

        # --- 9: Filter master table to regulatory only ---
        log_step(PREFIX,"reg_only","filtering master table")
        master_reg = master[master["Region_Type"].apply(lambda x: _simplify_type(x)!="gene_body")].copy()
        master_reg["_stype"] = master_reg["Region_Type"].apply(_simplify_type)

        # Recalculate stats on filtered set
        pct_cols_r = [f"Pct_{ct}" for ct in celltypes]
        master_reg["Max_pct"]  = master_reg[pct_cols_r].max(axis=1)
        master_reg["Min_pct"]  = master_reg[pct_cols_r].min(axis=1)
        master_reg["Diff_pct"] = master_reg["Max_pct"] - master_reg["Min_pct"]
        master_reg["log2FC"]   = np.log2((master_reg["Max_pct"]+PSEUDOCOUNT)/(master_reg["Min_pct"]+PSEUDOCOUNT))
        master_reg["Max_Celltype"] = master_reg[pct_cols_r].idxmax(axis=1).str.replace("Pct_","")
        def _excl_r(row):
            p=[ct for ct in celltypes if row[f"Count_{ct}"]>0]
            return p[0] if len(p)==1 else ""
        master_reg["ExclusiveCelltype"] = master_reg.apply(_excl_r, axis=1)
        master_reg.sort_values("Diff_pct", ascending=False, inplace=True)
        master_reg.to_csv(os.path.join(REG_DIR,"regulatory_only_master_table.csv"), index=False)

        n_full = len(master); n_reg = len(master_reg)
        log_step(PREFIX,"reg_only",f"kept {n_reg:,} / {n_full:,} regions (removed {n_full-n_reg:,} gene_body)")

        # --- 10: Recompute pairwise FDR on regulatory-only set ---
        log_step(PREFIX,"reg_only","recomputing pairwise FDR (regulatory only)")
        pairs = list(combinations(celltypes,2))
        reg_pair_results = {}
        reg_all_pvals = []
        reg_pair_sizes = []

        for ct_a,ct_b in pairs:
            pdf_full = pair_results.get((ct_a,ct_b))
            if pdf_full is None: continue

            # Filter out gene_body
            pdf_r = pdf_full[pdf_full["Region_Type"].apply(lambda x: _simplify_type(x)!="gene_body")].copy()
            pdf_r["_stype"] = pdf_r["Region_Type"].apply(_simplify_type)

            # Re-apply BH FDR on the reduced p-value set
            pvals_r = pdf_r["p_value"].values
            fdr_r = bh_fdr(pvals_r)
            pdf_r["FDR"] = fdr_r
            pdf_r["neg_log10_FDR"] = -np.log10(pdf_r["FDR"].clip(lower=1e-300))

            reg_all_pvals.extend(pvals_r.tolist())
            reg_pair_sizes.append(len(pvals_r))
            reg_pair_results[(ct_a,ct_b)] = pdf_r

        # Global FDR across all regulatory-only pairs
        reg_gfdr = bh_fdr(np.array(reg_all_pvals))
        offset = 0
        for (ct_a,ct_b),sz in zip(pairs,reg_pair_sizes):
            df = reg_pair_results.get((ct_a,ct_b))
            if df is None: continue
            df["global_FDR"] = reg_gfdr[offset:offset+sz]
            offset += sz
            df["sig_FDR005"]  = df["FDR"] < 0.05
            df["sig_FDR001"]  = df["FDR"] < 0.01
            df["sig_FDR0001"] = df["FDR"] < 0.001
            df["direction"] = "ns"
            df.loc[(df["FDR"]<0.05)&(df["log2FC"]>0),"direction"] = f"up_in_{ct_a}"
            df.loc[(df["FDR"]<0.05)&(df["log2FC"]<0),"direction"] = f"up_in_{ct_b}"
            df.sort_values("FDR", inplace=True)
            df.to_csv(os.path.join(REG_PAIR,f"{ct_a}_vs_{ct_b}.csv"), index=False)

            ns = (df["FDR"]<0.05).sum()
            log_step(PREFIX,"reg_only",f"  {ct_a} vs {ct_b}: {ns:,} sig FDR<0.05 (reg only)")

        # --- 11: Regulatory-only plots ---
        log_step(PREFIX,"reg_only","generating regulatory-only plots")
        SIMPLE_CATS_REG = [c for c in ["enhancer","silencer","promoter","uncharacterized","insulator","other_regulatory"]]

        # 11a: Volcanos
        for (ct_a,ct_b),pdf in reg_pair_results.items():
            fig,ax=plt.subplots(figsize=(10,8))
            ns=pdf[pdf["FDR"]>=0.05]
            ax.scatter(ns["log2FC"],ns["neg_log10_FDR"],c="#cccccc",s=6,alpha=0.25,
                       edgecolors="none",zorder=1,rasterized=True)
            sig=pdf[pdf["FDR"]<0.05].copy()
            for st in SIMPLE_CATS_REG:
                m=sig["_stype"]==st
                if m.sum()==0: continue
                ax.scatter(sig.loc[m,"log2FC"],sig.loc[m,"neg_log10_FDR"],
                           c=SIMPLE_TYPE_COLORS.get(st,"#95a5a6"),s=18,alpha=0.65,
                           edgecolors="k",linewidths=0.2,label=f"{st} ({m.sum():,})",
                           zorder=2,rasterized=True)
            for fc,ls in [(0.05,"--"),(0.01,":"),(0.001,"-.")]:
                y=-np.log10(fc)
                ax.axhline(y,color="#7f8c8d",ls=ls,lw=0.7,alpha=0.6)
                ax.text(ax.get_xlim()[1]*0.98 if ax.get_xlim()[1]>0 else 1,y,
                        f" FDR={fc}",va="bottom",ha="right",fontsize=7,color="#7f8c8d")
            ax.axvline(1,color="#7f8c8d",ls="--",lw=0.7,alpha=0.5)
            ax.axvline(-1,color="#7f8c8d",ls="--",lw=0.7,alpha=0.5)
            n_up=((pdf["FDR"]<0.05)&(pdf["log2FC"]>0)).sum()
            n_dn=((pdf["FDR"]<0.05)&(pdf["log2FC"]<0)).sum()
            ax.text(0.02,0.98,f"Up in {ct_a}: {n_up:,}",transform=ax.transAxes,
                    fontsize=9,va="top",color="#c0392b",fontweight="bold")
            ax.text(0.98,0.98,f"Up in {ct_b}: {n_dn:,}",transform=ax.transAxes,
                    fontsize=9,va="top",ha="right",color="#2980b9",fontweight="bold")
            if not sig.empty:
                top8=sig.nlargest(8,"neg_log10_FDR")
                angles=np.linspace(30,330,len(top8))
                for idx,(_,r) in enumerate(top8.iterrows()):
                    full_coord=rlabel(r["Chromosome"],int(r["Start"]),int(r["End"]))
                    ang=np.radians(angles[idx])
                    ox=np.cos(ang)*60; oy=np.sin(ang)*40
                    ax.annotate(full_coord,xy=(r["log2FC"],r["neg_log10_FDR"]),
                                xytext=(ox,oy),textcoords="offset points",
                                fontsize=6,fontweight="bold",alpha=0.9,
                                bbox=dict(boxstyle="round,pad=0.2",fc="white",ec="#bdc3c7",alpha=0.85,lw=0.5),
                                arrowprops=dict(arrowstyle="-",lw=0.6,color="#555555",
                                                connectionstyle="arc3,rad=0.15"))
            ax.set_xlabel(r"log$_2$ Fold Change ($\leftarrow$ "+ct_b+r" | "+ct_a+r" $\rightarrow$)",fontsize=12)
            ax.set_ylabel(r"$-\log_{10}$ FDR",fontsize=12)
            ax.set_title(f"Regulatory Only: {ct_a} vs {ct_b}\n"
                         f"(n={len(pdf):,} regions, no gene_body, Fisher's exact + BH)",
                         fontsize=13,fontweight="bold")
            _smart_legend(ax,title="Region Type",fontsize=8,title_fontsize=9)
            ax.grid(True,alpha=0.15); fig.subplots_adjust(bottom=0.16)
            _save(fig,os.path.join(REG_PLOT,f"volcano_{ct_a}_vs_{ct_b}"))
        log_step(PREFIX,"reg_only","volcanos done")

        # 11b: MA plots
        for (ct_a,ct_b),pdf in reg_pair_results.items():
            fig,ax=plt.subplots(figsize=(8,6))
            mp_val=(pdf[f"Pct_{ct_a}"]+pdf[f"Pct_{ct_b}"])/2
            ns_m=pdf["FDR"]>=0.05
            ax.scatter(mp_val[ns_m],pdf.loc[ns_m,"log2FC"],c="#cccccc",s=6,alpha=0.25,
                       edgecolors="none",rasterized=True,zorder=1)
            up_m=(pdf["FDR"]<0.05)&(pdf["log2FC"]>0)
            dn_m=(pdf["FDR"]<0.05)&(pdf["log2FC"]<0)
            ax.scatter(mp_val[up_m],pdf.loc[up_m,"log2FC"],c="#e74c3c",s=12,alpha=0.45,
                       edgecolors="none",rasterized=True,label=f"Up in {ct_a} ({up_m.sum():,})",zorder=2)
            ax.scatter(mp_val[dn_m],pdf.loc[dn_m,"log2FC"],c="#3498db",s=12,alpha=0.45,
                       edgecolors="none",rasterized=True,label=f"Up in {ct_b} ({dn_m.sum():,})",zorder=2)
            ax.axhline(0,color="black",lw=0.8)
            ax.set_xlabel("Mean Presence (%)",fontsize=12)
            ax.set_ylabel(r"log$_2$ Fold Change",fontsize=12)
            ax.set_title(f"MA Plot (Regulatory Only): {ct_a} vs {ct_b}",fontsize=13,fontweight="bold")
            _smart_legend(ax,fontsize=9,title_fontsize=9)
            ax.grid(True,alpha=0.15); fig.subplots_adjust(bottom=0.18)
            _save(fig,os.path.join(REG_PLOT,f"MA_{ct_a}_vs_{ct_b}"))
        log_step(PREFIX,"reg_only","MA plots done")

        # 11c: Heatmap top-N (regulatory only)
        log_step(PREFIX,"reg_only","heatmap")
        reg_fdr_dicts={}
        for (ca,cb),pdf in reg_pair_results.items():
            fd={}
            for c_v,s_v,e_v,fdr_v in zip(pdf["Chromosome"],pdf["Start"],pdf["End"],pdf["FDR"]):
                fd[(c_v,int(s_v),int(e_v))]=fdr_v
            reg_fdr_dicts[(ca,cb)]=fd

        reg_filtered_list=[r for r in region_list if _simplify_type(";".join(sorted(region_type_map[r])))!="gene_body"]
        min_fdr_reg={}
        for reg in reg_filtered_list:
            best=1.0
            for fd in reg_fdr_dicts.values():
                fv=fd.get(reg)
                if fv is not None and fv<best: best=fv
            min_fdr_reg[reg]=best

        sorted_sig_r=sorted(min_fdr_reg.items(),key=lambda x:x[1])
        top_regs_r=[r for r,_ in sorted_sig_r[:TOP_N_HEATMAP]]
        top_fdrs_r=[f for _,f in sorted_sig_r[:TOP_N_HEATMAP]]
        Nr=len(top_regs_r)

        if Nr>0:
            heat_data_r=np.zeros((Nr,len(celltypes)))
            labels_yr=[]; stypes_yr=[]; stars_yr=[]
            for i,reg in enumerate(top_regs_r):
                c,s,e=reg
                labels_yr.append(rlabel(c,s,e))
                stypes_yr.append(_simplify_type(";".join(sorted(region_type_map[reg]))))
                fv=top_fdrs_r[i]
                stars_yr.append("***" if fv<0.001 else "**" if fv<0.01 else "*" if fv<0.05 else "")
                for j,ct in enumerate(celltypes):
                    cnt=ct_counts[ct].get(reg,0); nc=celltype_ncells[ct]
                    heat_data_r[i,j]=(cnt/nc*100) if nc else 0.0

            row_h=0.55; fig_w=max(12,len(celltypes)*1.8); fig_h=row_h*Nr+5
            fig,axes=plt.subplots(1,2,figsize=(fig_w,fig_h),gridspec_kw={"width_ratios":[1,15]})
            ax0=axes[0]
            for i,st in enumerate(stypes_yr):
                ax0.add_patch(plt.Rectangle((0,i),1,1,color=SIMPLE_TYPE_COLORS.get(st,"#95a5a6")))
            ax0.set_xlim(0,1);ax0.set_ylim(0,Nr);ax0.invert_yaxis()
            ax0.set_xticks([]);ax0.set_yticks([]);ax0.set_xlabel("Type",fontsize=10)
            ax1=axes[1]
            im=ax1.imshow(heat_data_r,cmap="YlOrRd",aspect="auto",vmin=0)
            cb=plt.colorbar(im,ax=ax1,shrink=0.5,pad=0.02);cb.set_label("Presence (%)",fontsize=11)
            ax1.set_xticks(np.arange(len(celltypes)))
            ax1.set_xticklabels(celltypes,rotation=45,ha="right",fontsize=11,fontweight="bold")
            ax1.xaxis.tick_top();ax1.xaxis.set_label_position("top")
            ax1.set_yticks(np.arange(Nr));ax1.set_yticklabels(labels_yr,fontsize=8)
            max_ct_idx_r=np.argmax(heat_data_r,axis=1)
            for i in range(Nr):
                for j in range(len(celltypes)):
                    val=heat_data_r[i,j]; txt=f"{val:.1f}"
                    if j==max_ct_idx_r[i] and stars_yr[i]: txt+=stars_yr[i]
                    ax1.text(j,i,txt,ha="center",va="center",fontsize=7,
                             fontweight="bold" if j==max_ct_idx_r[i] else "normal")
            ax1.set_title(f"Top {Nr} Significant Regulatory Regions (no gene_body)\n"
                          f"* p<0.05  ** p<0.01  *** p<0.001",fontsize=14,fontweight="bold")
            unique_st_r=sorted(set(stypes_yr))
            handles_r=[mpatches.Patch(color=SIMPLE_TYPE_COLORS.get(t,"#95a5a6"),label=t) for t in unique_st_r]
            fig.legend(handles=handles_r,title="Region Type",fontsize=9,title_fontsize=10,
                       ncol=max(1,len(handles_r)),loc="lower center",bbox_to_anchor=(0.55,-0.01),
                       framealpha=0.9,edgecolor="#bdc3c7",columnspacing=1.5)
            plt.tight_layout(); fig.subplots_adjust(bottom=0.06)
            _save(fig,os.path.join(REG_PLOT,f"heatmap_top{Nr}_regulatory_only"))
        log_step(PREFIX,"reg_only","heatmap done")

        # 11d: Composition + exclusive bars (regulatory only)
        ct_stype_r={}
        for ct in celltypes:
            counts={st:0 for st in SIMPLE_CATS_REG}
            for reg in reg_filtered_list:
                if ct_counts[ct].get(reg,0)>0:
                    counts[_simplify_type(";".join(sorted(region_type_map[reg])))]+=1
            ct_stype_r[ct]=counts

        fig,ax=plt.subplots(figsize=(max(9,len(celltypes)*1.2),7))
        bottoms=np.zeros(len(celltypes))
        for st in SIMPLE_CATS_REG:
            vals=np.array([ct_stype_r[ct][st] for ct in celltypes],dtype=float)
            if vals.sum()==0: continue
            ax.bar(celltypes,vals,bottom=bottoms,label=f"{st} ({int(vals.sum()):,})",
                   color=SIMPLE_TYPE_COLORS.get(st,"#95a5a6"),edgecolor="white",linewidth=0.3)
            bottoms+=vals
        ax.set_ylabel("Active Regulatory Regions",fontsize=12)
        ax.set_title(f"{PREFIX}: Regulatory Region Composition (no gene_body)",fontsize=14,fontweight="bold")
        _smart_legend(ax,title="Region Type",fontsize=9,title_fontsize=10)
        plt.xticks(rotation=45,ha="right",fontsize=10); fig.subplots_adjust(bottom=0.22)
        _save(fig,os.path.join(REG_PLOT,"region_type_composition_regulatory"))

        # Exclusive
        excl_st_r={}
        for ct in celltypes:
            sub=master_reg[master_reg["ExclusiveCelltype"]==ct]
            counts={st:0 for st in SIMPLE_CATS_REG}
            if not sub.empty:
                for rt in sub["Region_Type"]: counts[_simplify_type(rt)]+=1
                sub.to_csv(os.path.join(REG_EXCL,f"exclusive_{ct}.csv"),index=False)
            excl_st_r[ct]=counts

        fig,ax=plt.subplots(figsize=(max(9,len(celltypes)*1.2),7))
        bottoms=np.zeros(len(celltypes))
        for st in SIMPLE_CATS_REG:
            vals=np.array([excl_st_r[ct][st] for ct in celltypes],dtype=float)
            if vals.sum()==0: continue
            ax.bar(celltypes,vals,bottom=bottoms,label=f"{st} ({int(vals.sum()):,})",
                   color=SIMPLE_TYPE_COLORS.get(st,"#95a5a6"),edgecolor="white",linewidth=0.3)
            bottoms+=vals
        ax.set_ylabel("Exclusive Regulatory Regions",fontsize=12)
        ax.set_title(f"{PREFIX}: Exclusive Regulatory Regions (no gene_body)",fontsize=14,fontweight="bold")
        _smart_legend(ax,title="Region Type",fontsize=9,title_fontsize=10)
        plt.xticks(rotation=45,ha="right",fontsize=10); fig.subplots_adjust(bottom=0.22)
        _save(fig,os.path.join(REG_PLOT,"exclusive_regions_regulatory"))

        # 11e: FDR distribution + sig per pair (regulatory only)
        reg_fdrs=[]
        for pdf in reg_pair_results.values(): reg_fdrs.extend(pdf["FDR"].tolist())
        reg_fdrs=np.array(reg_fdrs)
        fig,ax=plt.subplots(figsize=(8,5))
        ax.hist(reg_fdrs[reg_fdrs<1.0],bins=100,color="#2c3e50",alpha=0.8,edgecolor="white",linewidth=0.3)
        for fc,ls,col in [(0.05,"--","#e74c3c"),(0.01,":","#e67e22"),(0.001,"-.","#f39c12")]:
            ax.axvline(fc,color=col,ls=ls,lw=1.2,label=f"FDR={fc}")
        ax.set_xlabel("FDR (BH)",fontsize=12);ax.set_ylabel("Count",fontsize=12)
        ax.set_title(f"FDR Distribution — Regulatory Only ({len(reg_fdrs):,} tests)",fontsize=13,fontweight="bold")
        _smart_legend(ax,fontsize=9,outside=False);ax.set_xlim(-0.02,1.02);plt.tight_layout()
        _save(fig,os.path.join(REG_PLOT,"FDR_distribution_regulatory"))

        fig,ax=plt.subplots(figsize=(max(8,len(pairs)*0.8),6))
        pair_labels=[f"{a}\nvs\n{b}" for a,b in pairs]
        n_up_r=[((reg_pair_results[(a,b)]["FDR"]<0.05)&(reg_pair_results[(a,b)]["log2FC"]>0)).sum() for a,b in pairs]
        n_dn_r=[((reg_pair_results[(a,b)]["FDR"]<0.05)&(reg_pair_results[(a,b)]["log2FC"]<0)).sum() for a,b in pairs]
        x=np.arange(len(pairs));w=0.35
        b1=ax.bar(x-w/2,n_up_r,w,label="Up in A",color="#e74c3c",edgecolor="white")
        b2=ax.bar(x+w/2,n_dn_r,w,label="Up in B",color="#3498db",edgecolor="white")
        ax.set_xticks(x);ax.set_xticklabels(pair_labels,fontsize=7)
        ax.set_ylabel("Significant Regions (FDR<0.05)",fontsize=12)
        ax.set_title(f"{PREFIX}: Sig Regulatory Regions per Pair (no gene_body)",fontsize=13,fontweight="bold")
        _smart_legend(ax,fontsize=9,outside=False)
        for bars in (b1,b2):
            for bar in bars:
                h=bar.get_height()
                if h>0: ax.text(bar.get_x()+bar.get_width()/2,h,f"{int(h):,}",ha="center",va="bottom",fontsize=6)
        plt.tight_layout()
        _save(fig,os.path.join(REG_PLOT,"sig_regions_per_pair_regulatory"))

        # 11f: Diff_pct by type (regulatory only)
        fig,ax=plt.subplots(figsize=(10,6))
        for st in SIMPLE_CATS_REG:
            sub=master_reg[master_reg["_stype"]==st]
            if sub.empty: continue
            ax.hist(sub["Diff_pct"],bins=60,alpha=0.5,
                    color=SIMPLE_TYPE_COLORS.get(st,"#95a5a6"),label=f"{st} ({len(sub):,})")
        ax.set_xlabel("Diff_pct (Max%-Min%)",fontsize=12);ax.set_ylabel("Regions",fontsize=12)
        ax.set_title(f"{PREFIX}: Diff_pct — Regulatory Only (no gene_body)",fontsize=14,fontweight="bold")
        _smart_legend(ax,title="Region Type",fontsize=9,title_fontsize=10)
        fig.subplots_adjust(bottom=0.18)
        _save(fig,os.path.join(REG_PLOT,"diff_pct_distribution_regulatory"))

        # 11g: Jaccard (regulatory only)
        ct_cent_r={ct:{r for r in reg_filtered_list if ct_counts[ct].get(r,0)>0} for ct in celltypes}
        nct=len(celltypes)
        jac_r=np.zeros((nct,nct))
        for i in range(nct):
            for j in range(nct):
                if i==j: jac_r[i,j]=100.0
                else:
                    inter=len(ct_cent_r[celltypes[i]]&ct_cent_r[celltypes[j]])
                    union=len(ct_cent_r[celltypes[i]]|ct_cent_r[celltypes[j]])
                    jac_r[i,j]=(inter/union*100) if union else 0.0
        jac_r_df=pd.DataFrame(jac_r,index=celltypes,columns=celltypes)
        jac_r_df.to_csv(os.path.join(REG_SIM,"jaccard_regulatory_only.csv"))
        fig,ax=plt.subplots(figsize=(max(7,nct*0.9),max(6,nct*0.8)))
        mask=np.eye(nct,dtype=bool); vmax=jac_r_df.where(~mask).max().max()
        if np.isnan(vmax): vmax=100.0
        sns.heatmap(jac_r_df,cmap="viridis",square=True,vmin=0,vmax=vmax,mask=mask,
                    annot=True,fmt=".1f",cbar_kws={"label":"Jaccard (%)"},ax=ax,
                    linewidths=0.5,linecolor="white")
        for i in range(nct): ax.add_patch(plt.Rectangle((i,i),1,1,color="#e74c3c",lw=0))
        ax.set_title(f"{PREFIX}: Jaccard Similarity (Regulatory Only)",fontsize=13,fontweight="bold")
        plt.tight_layout()
        _save(fig,os.path.join(REG_SIM,"jaccard_regulatory_only"))

        # --- 12: Regulatory-only summary ---
        reg_summary=[]
        for ca,cb in pairs:
            pdf=reg_pair_results.get((ca,cb))
            if pdf is None: continue
            row={"Comparison":f"{ca}_vs_{cb}","Total_regions":len(pdf),
                 "nCells_A":celltype_ncells[ca],"nCells_B":celltype_ncells[cb]}
            for fc in FDR_CUTS:
                sig=pdf["FDR"]<fc
                row[f"sig_FDR{fc}"]=sig.sum()
                row[f"up_{ca}_FDR{fc}"]=((sig)&(pdf["log2FC"]>0)).sum()
                row[f"up_{cb}_FDR{fc}"]=((sig)&(pdf["log2FC"]<0)).sum()
            row["max_log2FC"]=pdf["log2FC"].max()
            row["min_log2FC"]=pdf["log2FC"].min()
            row["min_FDR"]=pdf["FDR"].min()
            reg_summary.append(row)
        pd.DataFrame(reg_summary).to_csv(os.path.join(REG_DIR,"pairwise_summary_regulatory.csv"),index=False)

        save_step(OUT,"reg_only",True)
        log_step(PREFIX,"reg_only","DONE — all regulatory-only outputs saved")
    else:
        log_step(PREFIX,"reg_only","RESUMING — already complete")

    elapsed=time.time()-t0
    log_step(PREFIX,"COMPLETE",f"{elapsed/60:.1f} min total")
    return True

# ================================================================
#  MAIN: iterate all prefixes with global checkpoint
# ================================================================

global_ckpt = load_global_ckpt()
completed = set(global_ckpt.get("completed",[]))

log_step("MAIN","start",f"{len(PREFIXES)} groups, {len(completed)} already done")

for i,pfx in enumerate(PREFIXES):
    if pfx in completed:
        log_step(pfx,"SKIP","already completed in previous run")
        continue

    log_step(pfx,"START",f"[{i+1}/{len(PREFIXES)}]")
    try:
        result = run_prefix(pfx)
        if result:
            completed.add(pfx)
            global_ckpt["completed"] = sorted(completed)
            global_ckpt[pfx] = {"finished_at": time.strftime("%Y-%m-%d %H:%M:%S")}
            save_global_ckpt(global_ckpt)
            log_step(pfx,"SAVED","global checkpoint updated — output available now")
    except Exception as e:
        log_step(pfx,"ERROR",f"{e}")
        traceback.print_exc()
        # Continue to next prefix — don't abort the whole run
        continue

# ================================================================
#  CROSS-GROUP ANALYSIS — DISABLED (separated into dedicated scripts)
#  Brain: 14_cross_group_brain.py -> cross_group_brain/
#  CRC:   14_cross_group_CRC.py   -> cross_group_CRC/
# ================================================================

CROSS_DIR = os.path.join(BASE_DIR, "cross_group_regulatory_divergence")
CROSS_CKPT = os.path.join(CROSS_DIR, ".ckpt_done")

if False and len(completed) >= 2 and not os.path.exists(CROSS_CKPT):
    os.makedirs(CROSS_DIR, exist_ok=True)
    log_step("CROSS","start",f"comprehensive divergence analysis across {len(completed)} groups")

    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import squareform, pdist
    from scipy.stats import spearmanr, entropy
    from sklearn.decomposition import PCA
    try:
        from sklearn.manifold import TSNE
        HAS_TSNE = True
    except ImportError:
        HAS_TSNE = False

    # ============================================================
    # A. DATA COLLECTION
    # ============================================================
    log_step("CROSS","A","loading all regulatory master tables")

    subtype_profiles   = {}   # "Group__Subtype" -> {(chr,start,end): Pct}
    subtype_group      = {}   # "Group__Subtype" -> "Group"
    subtype_ncells     = {}   # "Group__Subtype" -> nCells
    subtype_type_comp  = {}   # "Group__Subtype" -> {stype: count}
    all_cross_regions  = set()
    group_sig_regions  = {}   # group -> set of significant regions (FDR<0.05 in any pair)
    TOP_PER_GROUP      = 500

    for pfx in sorted(completed):
        out_dir_pfx = os.path.join(BASE_DIR, f"{pfx}_differential_unified_celltypes")
        reg_master = os.path.join(out_dir_pfx, "regulatory_only", "regulatory_only_master_table.csv")
        if not os.path.exists(reg_master):
            log_step("CROSS","skip",f"{pfx}: no regulatory master table")
            continue

        df = pd.read_csv(reg_master, low_memory=False, dtype={"Chromosome":str})
        pct_c = [c for c in df.columns if c.startswith("Pct_")]
        cnt_c = [c for c in df.columns if c.startswith("Count_")]
        cts = [c.replace("Pct_","") for c in pct_c]

        # Top variable regions
        top_var = df.nlargest(TOP_PER_GROUP, "Diff_pct")
        df["_stype"] = df["Region_Type"].apply(_simplify_type)

        # Collect significant regions from pairwise CSVs
        sig_regs = set()
        pair_dir = os.path.join(out_dir_pfx, "regulatory_only", "pairwise")
        if os.path.isdir(pair_dir):
            for pf in glob.glob(os.path.join(pair_dir, "*.csv")):
                try:
                    pw = pd.read_csv(pf, usecols=["Chromosome","Start","End","FDR"],
                                     low_memory=False, dtype={"Chromosome":str})
                    for _,r in pw[pw["FDR"]<0.05].iterrows():
                        sig_regs.add((r["Chromosome"],int(r["Start"]),int(r["End"])))
                except: pass
        group_sig_regions[pfx] = sig_regs

        for ct in cts:
            label = f"{pfx}__{ct}"
            profile = {}
            for _,row in top_var.iterrows():
                reg = (row["Chromosome"], int(row["Start"]), int(row["End"]))
                profile[reg] = row[f"Pct_{ct}"]
                all_cross_regions.add(reg)
            subtype_profiles[label] = profile
            subtype_group[label] = pfx

            # nCells from master (sum of Count columns / average Pct or use a heuristic)
            # Approximate: look at first row with Count>0 to infer nCells
            for _,row in df.iterrows():
                cnt_val = row[f"Count_{ct}"]
                pct_val = row[f"Pct_{ct}"]
                if cnt_val > 0 and pct_val > 0:
                    subtype_ncells[label] = int(round(cnt_val / (pct_val/100)))
                    break
            else:
                subtype_ncells[label] = 0

            # Region type composition (all regions, not just top 500)
            type_comp = {}
            for st_cat in ["enhancer","silencer","promoter","uncharacterized","insulator","other_regulatory"]:
                active = (df["_stype"]==st_cat) & (df[f"Count_{ct}"]>0)
                type_comp[st_cat] = active.sum()
            subtype_type_comp[label] = type_comp

        log_step("CROSS","A",f"{pfx}: {len(cts)} subtypes, {len(sig_regs):,} sig regions")

    subtypes_all = sorted(subtype_profiles.keys())
    regions_all  = sorted(all_cross_regions)
    n_st = len(subtypes_all)
    n_reg = len(regions_all)
    log_step("CROSS","A",f"TOTAL: {n_st} subtypes, {n_reg:,} regions")

    if n_st < 2:
        log_step("CROSS","skip","need >= 2 subtypes")
    else:
        # ============================================================
        # B. PRESENCE MATRIX + DISTANCE
        # ============================================================
        log_step("CROSS","B","building presence matrix and distance")

        mat = np.zeros((n_st, n_reg), dtype=np.float32)
        for i, st in enumerate(subtypes_all):
            prof = subtype_profiles[st]
            for j, reg in enumerate(regions_all):
                mat[i, j] = prof.get(reg, 0.0)

        # Spearman distance
        dist_mat = np.zeros((n_st, n_st))
        for i in range(n_st):
            for j in range(i+1, n_st):
                rho, _ = spearmanr(mat[i], mat[j])
                if np.isnan(rho): rho = 0.0
                d = 1.0 - rho
                dist_mat[i,j] = dist_mat[j,i] = d

        dist_df = pd.DataFrame(dist_mat, index=subtypes_all, columns=subtypes_all)
        dist_df.to_csv(os.path.join(CROSS_DIR, "spearman_divergence_matrix.csv"))

        condensed = squareform(dist_mat)
        condensed = np.clip(condensed, 0, None)
        Z = linkage(condensed, method="average")
        ordered_idx = leaves_list(Z)
        ordered_labels = [subtypes_all[k] for k in ordered_idx]

        # Group color palette
        unique_groups = sorted(set(subtype_group.values()))
        gpal = dict(zip(unique_groups, sns.color_palette("husl", len(unique_groups)).as_hex()))
        row_colors = pd.Series([gpal[subtype_group[st]] for st in subtypes_all],
                               index=subtypes_all, name="Group")

        log_step("CROSS","B","distance and clustering done")

        # ============================================================
        # C1. CLUSTERMAP: Divergence heatmap + dendrogram tree
        # ============================================================
        log_step("CROSS","C1","clustermap")

        fs = (max(16, n_st*0.28), max(14, n_st*0.25))
        g = sns.clustermap(
            dist_df, row_linkage=Z, col_linkage=Z,
            cmap="RdYlBu_r", vmin=0, vmax=np.percentile(dist_mat[dist_mat>0],95),
            figsize=fs, row_colors=row_colors, col_colors=row_colors,
            linewidths=0, xticklabels=True, yticklabels=True,
            cbar_kws={"label":"Spearman Divergence (1 - \u03c1)", "shrink":0.4},
            dendrogram_ratio=(0.12, 0.12), annot=False,
        )
        g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), fontsize=max(3,min(7,400//n_st)), rotation=90)
        g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), fontsize=max(3,min(7,400//n_st)), rotation=0)
        g.fig.suptitle("Cross-Group Regulatory Divergence (Spearman Distance)\n"
                       f"Top {TOP_PER_GROUP} variable regulatory regions per group, {n_st} subtypes",
                       fontsize=14, fontweight="bold", y=1.02)
        handles_g = [mpatches.Patch(color=gpal[gn], label=gn) for gn in unique_groups]
        ncol_g = max(1, (len(handles_g)+5)//6)
        g.fig.legend(handles=handles_g, title="Sample Group", fontsize=7, title_fontsize=8,
                     ncol=ncol_g, loc="lower center", bbox_to_anchor=(0.5,-0.03),
                     framealpha=0.9, columnspacing=1.2)
        for ext in ("png","pdf"):
            g.savefig(os.path.join(CROSS_DIR, f"C1_divergence_clustermap.{ext}"), dpi=300, bbox_inches="tight")
        plt.close(); log_step("CROSS","C1","saved")

        # ============================================================
        # C2. PCA of regulatory fingerprints
        # ============================================================
        log_step("CROSS","C2","PCA")
        pca = PCA(n_components=min(10, n_st-1, n_reg))
        pca_coords = pca.fit_transform(mat)

        fig, ax = plt.subplots(figsize=(10, 8))
        for gn in unique_groups:
            mask = [subtype_group[st]==gn for st in subtypes_all]
            idx = [i for i,m in enumerate(mask) if m]
            ax.scatter(pca_coords[idx,0], pca_coords[idx,1],
                       c=gpal[gn], s=60, alpha=0.8, edgecolors="k", linewidths=0.4,
                       label=gn, zorder=2)
            for i in idx:
                short_name = subtypes_all[i].split("__")[1] if "__" in subtypes_all[i] else subtypes_all[i]
                ax.annotate(short_name, (pca_coords[i,0], pca_coords[i,1]),
                            fontsize=5, alpha=0.7, xytext=(3,3), textcoords="offset points")
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=12)
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=12)
        ax.set_title("PCA of Regulatory Landscape Fingerprints\n"
                     f"({n_reg:,} regions, {n_st} subtypes across {len(unique_groups)} sample groups)",
                     fontsize=13, fontweight="bold")
        _smart_legend(ax, title="Sample Group", fontsize=7, title_fontsize=8)
        ax.grid(True, alpha=0.15); fig.subplots_adjust(bottom=0.16)
        _save(fig, os.path.join(CROSS_DIR, "C2_PCA_regulatory_fingerprints"))

        # Scree plot
        fig, ax = plt.subplots(figsize=(7, 4))
        n_comp = len(pca.explained_variance_ratio_)
        ax.bar(range(1, n_comp+1), pca.explained_variance_ratio_*100, color="#3498db", edgecolor="white")
        ax.plot(range(1, n_comp+1), np.cumsum(pca.explained_variance_ratio_)*100, "o-", color="#e74c3c", markersize=5)
        ax.set_xlabel("Principal Component", fontsize=12)
        ax.set_ylabel("Variance Explained (%)", fontsize=12)
        ax.set_title("PCA Scree Plot", fontsize=13, fontweight="bold")
        ax.legend(["Cumulative", "Per-component"], fontsize=9, framealpha=0.85)
        plt.tight_layout()
        _save(fig, os.path.join(CROSS_DIR, "C2_PCA_scree"))
        log_step("CROSS","C2","saved")

        # ============================================================
        # C3. t-SNE (if available and n_st > 5)
        # ============================================================
        if HAS_TSNE and n_st > 5:
            log_step("CROSS","C3","t-SNE")
            perp = min(30, max(2, n_st//3))
            tsne = TSNE(n_components=2, perplexity=perp, random_state=42, metric="precomputed")
            tsne_coords = tsne.fit_transform(dist_mat)

            fig, ax = plt.subplots(figsize=(10, 8))
            for gn in unique_groups:
                mask = [subtype_group[st]==gn for st in subtypes_all]
                idx = [i for i,m in enumerate(mask) if m]
                ax.scatter(tsne_coords[idx,0], tsne_coords[idx,1],
                           c=gpal[gn], s=60, alpha=0.8, edgecolors="k", linewidths=0.4,
                           label=gn, zorder=2)
                for i in idx:
                    short_name = subtypes_all[i].split("__")[1] if "__" in subtypes_all[i] else subtypes_all[i]
                    ax.annotate(short_name, (tsne_coords[i,0], tsne_coords[i,1]),
                                fontsize=5, alpha=0.7, xytext=(3,3), textcoords="offset points")
            ax.set_xlabel("t-SNE 1", fontsize=12); ax.set_ylabel("t-SNE 2", fontsize=12)
            ax.set_title(f"t-SNE of Regulatory Divergence (perplexity={perp})",
                         fontsize=13, fontweight="bold")
            _smart_legend(ax, title="Sample Group", fontsize=7, title_fontsize=8)
            ax.grid(True, alpha=0.15); fig.subplots_adjust(bottom=0.16)
            _save(fig, os.path.join(CROSS_DIR, "C3_tSNE_regulatory_divergence"))
            log_step("CROSS","C3","saved")

        # ============================================================
        # C4. Region-type enrichment shift (stacked bar by dendrogram order)
        # ============================================================
        log_step("CROSS","C4","region-type composition shift")
        REG_CATS = ["enhancer","silencer","promoter","uncharacterized","insulator","other_regulatory"]

        fig, ax = plt.subplots(figsize=(max(14, n_st*0.3), 7))
        bottoms = np.zeros(n_st)
        x_pos = np.arange(n_st)
        for st_cat in REG_CATS:
            vals = np.array([subtype_type_comp[ordered_labels[k]].get(st_cat, 0) for k in range(n_st)], dtype=float)
            totals = np.array([sum(subtype_type_comp[ordered_labels[k]].values()) for k in range(n_st)], dtype=float)
            pcts = np.where(totals>0, vals/totals*100, 0.0)
            ax.bar(x_pos, pcts, bottom=bottoms, label=st_cat,
                   color=SIMPLE_TYPE_COLORS.get(st_cat,"#95a5a6"), edgecolor="white", linewidth=0.2)
            bottoms += pcts
        ax.set_xticks(x_pos)
        ax.set_xticklabels([ordered_labels[k].replace("__","\n") for k in range(n_st)],
                           fontsize=max(3,min(6,300//n_st)), rotation=90)
        ax.set_ylabel("Region Type Proportion (%)", fontsize=12)
        ax.set_title("Regulatory Composition Shift Across All Subtypes\n"
                     "(ordered by divergence dendrogram)", fontsize=13, fontweight="bold")
        _smart_legend(ax, title="Region Type", fontsize=8, title_fontsize=9)
        fig.subplots_adjust(bottom=0.25)
        _save(fig, os.path.join(CROSS_DIR, "C4_region_type_composition_shift"))
        log_step("CROSS","C4","saved")

        # ============================================================
        # C5. Top-region cross-group heatmap (with dendrogram column order)
        # ============================================================
        log_step("CROSS","C5","top region cross-group heatmap")
        top_cross_regs = []; top_cross_labels = []
        for pfx in sorted(completed):
            reg_master = os.path.join(BASE_DIR, f"{pfx}_differential_unified_celltypes",
                                      "regulatory_only","regulatory_only_master_table.csv")
            if not os.path.exists(reg_master): continue
            df = pd.read_csv(reg_master, low_memory=False, dtype={"Chromosome":str})
            top10 = df.nlargest(10, "Diff_pct")
            for _,row in top10.iterrows():
                reg = (row["Chromosome"],int(row["Start"]),int(row["End"]))
                if reg not in top_cross_regs:
                    top_cross_regs.append(reg)
                    st = _simplify_type(row["Region_Type"])
                    top_cross_labels.append(f"{rlabel(*reg)} [{st}] ({pfx})")

        if len(top_cross_regs) > 0:
            n_tr = len(top_cross_regs)
            rank_mat = np.zeros((n_tr, n_st))
            for i, reg in enumerate(top_cross_regs):
                for j, st in enumerate(subtypes_all):
                    rank_mat[i,j] = subtype_profiles[st].get(reg, 0.0)
            rank_mat_ord = rank_mat[:, ordered_idx]

            fig_h = max(10, n_tr*0.35); fig_w = max(16, n_st*0.28)
            fig, ax = plt.subplots(figsize=(fig_w, fig_h))
            im = ax.imshow(rank_mat_ord, cmap="YlOrRd", aspect="auto", vmin=0)
            cb = plt.colorbar(im, ax=ax, shrink=0.35, pad=0.01)
            cb.set_label("Presence (%)", fontsize=10)
            ax.set_xticks(np.arange(n_st))
            ax.set_xticklabels(ordered_labels, rotation=90, fontsize=max(3,min(6,300//n_st)))
            ax.xaxis.tick_top()
            ax.set_yticks(np.arange(n_tr))
            ax.set_yticklabels(top_cross_labels, fontsize=6)
            ax.set_title("Top 10 Regulatory Regions per Group: Presence Across All Subtypes\n"
                         "(columns ordered by divergence dendrogram)", fontsize=13, fontweight="bold", pad=20)
            plt.tight_layout()
            _save(fig, os.path.join(CROSS_DIR, "C5_top_regions_cross_group_heatmap"))
            log_step("CROSS","C5","saved")

        # ============================================================
        # C6. Cross-group region concordance (coordinate overlap)
        # ============================================================
        log_step("CROSS","C6","cross-group concordance")
        groups_with_sig = sorted(group_sig_regions.keys())
        ng = len(groups_with_sig)
        conc_mat = np.zeros((ng, ng))
        for i in range(ng):
            for j in range(ng):
                si = group_sig_regions[groups_with_sig[i]]
                sj = group_sig_regions[groups_with_sig[j]]
                if i == j:
                    conc_mat[i,j] = len(si)
                else:
                    conc_mat[i,j] = len(si & sj)

        conc_df = pd.DataFrame(conc_mat.astype(int), index=groups_with_sig, columns=groups_with_sig)
        conc_df.to_csv(os.path.join(CROSS_DIR, "C6_significant_region_concordance.csv"))

        fig, ax = plt.subplots(figsize=(max(8, ng*0.7), max(7, ng*0.6)))
        mask_diag = np.eye(ng, dtype=bool)
        sns.heatmap(conc_df, annot=True, fmt="d", cmap="Blues", ax=ax,
                    mask=mask_diag, linewidths=0.5, linecolor="white",
                    cbar_kws={"label":"Shared Significant Regions"})
        for i in range(ng):
            ax.text(i+0.5, i+0.5, f"{int(conc_mat[i,i]):,}", ha="center", va="center",
                    fontsize=7, fontweight="bold", color="#c0392b")
        ax.set_title("Cross-Group Concordance: Shared Significant Regulatory Regions\n"
                     "(diagonal = total sig regions per group, off-diagonal = shared count)",
                     fontsize=12, fontweight="bold")
        plt.xticks(rotation=45, ha="right", fontsize=9)
        plt.yticks(rotation=0, fontsize=9)
        plt.tight_layout()
        _save(fig, os.path.join(CROSS_DIR, "C6_concordance_heatmap"))
        log_step("CROSS","C6","saved")

        # ============================================================
        # C7. Regulatory complexity index per subtype
        # ============================================================
        log_step("CROSS","C7","regulatory complexity index")
        complexity_rows = []
        for st in ordered_labels:
            prof = subtype_profiles[st]
            tc = subtype_type_comp.get(st, {})
            total_active = sum(tc.values())
            # Shannon diversity of region types
            type_counts = np.array([tc.get(c,0) for c in REG_CATS], dtype=float)
            type_probs = type_counts / type_counts.sum() if type_counts.sum()>0 else type_counts
            shannon = entropy(type_probs, base=2)
            # Presence statistics
            pct_vals = np.array(list(prof.values()))
            complexity_rows.append({
                "Subtype": st, "Group": subtype_group[st],
                "nCells": subtype_ncells.get(st, 0),
                "Total_active_regions": total_active,
                "Shannon_type_diversity": round(shannon, 4),
                "Mean_presence_pct": round(pct_vals.mean(), 2) if len(pct_vals)>0 else 0,
                "Median_presence_pct": round(np.median(pct_vals), 2) if len(pct_vals)>0 else 0,
                "Regions_gt10pct": int((pct_vals>10).sum()) if len(pct_vals)>0 else 0,
            })
        complexity_df = pd.DataFrame(complexity_rows)
        complexity_df.to_csv(os.path.join(CROSS_DIR, "C7_regulatory_complexity_index.csv"), index=False)

        # Plot: bubble chart (total active vs Shannon diversity, size=nCells)
        fig, ax = plt.subplots(figsize=(12, 8))
        for gn in unique_groups:
            sub = complexity_df[complexity_df["Group"]==gn]
            sizes = np.clip(sub["nCells"].values/10, 10, 300)
            ax.scatter(sub["Total_active_regions"], sub["Shannon_type_diversity"],
                       s=sizes, c=gpal[gn], alpha=0.7, edgecolors="k", linewidths=0.4,
                       label=gn, zorder=2)
            for _,r in sub.iterrows():
                short_name = r["Subtype"].split("__")[1] if "__" in r["Subtype"] else r["Subtype"]
                ax.annotate(short_name, (r["Total_active_regions"], r["Shannon_type_diversity"]),
                            fontsize=5, alpha=0.7, xytext=(3,3), textcoords="offset points")
        ax.set_xlabel("Total Active Regulatory Regions", fontsize=12)
        ax.set_ylabel("Shannon Diversity of Region Types (bits)", fontsize=12)
        ax.set_title("Regulatory Complexity: Region Count vs Type Diversity\n"
                     "(bubble size = cell count)", fontsize=13, fontweight="bold")
        _smart_legend(ax, title="Sample Group", fontsize=7, title_fontsize=8)
        ax.grid(True, alpha=0.15); fig.subplots_adjust(bottom=0.16)
        _save(fig, os.path.join(CROSS_DIR, "C7_regulatory_complexity_bubble"))
        log_step("CROSS","C7","saved")

        # ============================================================
        # C8. Conserved vs divergent regulatory regions
        # ============================================================
        log_step("CROSS","C8","conserved vs divergent regions")
        # For each region, count how many subtypes have it >5%
        region_breadth = {}
        for j, reg in enumerate(regions_all):
            n_active = (mat[:, j] > 5.0).sum()
            region_breadth[reg] = n_active

        # Categorize: universal (>80% subtypes), shared (20-80%), specific (<20%)
        n80 = int(n_st * 0.8); n20 = int(n_st * 0.2)
        universal = [r for r,b in region_breadth.items() if b >= n80]
        shared    = [r for r,b in region_breadth.items() if n20 <= b < n80]
        specific  = [r for r,b in region_breadth.items() if b < n20]

        conserv_summary = {
            "Universal (>80% subtypes)": len(universal),
            "Shared (20-80%)": len(shared),
            "Subtype-specific (<20%)": len(specific),
            "Total": n_reg,
        }
        pd.DataFrame([conserv_summary]).to_csv(
            os.path.join(CROSS_DIR, "C8_conserved_vs_divergent_summary.csv"), index=False)

        fig, ax = plt.subplots(figsize=(7, 5))
        cats = list(conserv_summary.keys())[:-1]
        vals = [conserv_summary[c] for c in cats]
        colors = ["#27ae60", "#f39c12", "#e74c3c"]
        bars = ax.bar(cats, vals, color=colors, edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height(),
                    f"{v:,}\n({v/n_reg*100:.1f}%)", ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.set_ylabel("Number of Regulatory Regions", fontsize=12)
        ax.set_title("Conserved vs Divergent Regulatory Regions\n"
                     f"(across {n_st} subtypes, >5% presence threshold)",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        _save(fig, os.path.join(CROSS_DIR, "C8_conserved_vs_divergent_bar"))
        log_step("CROSS","C8","saved")

        # ============================================================
        # C9. Pairwise divergence summary CSV
        # ============================================================
        log_step("CROSS","C9","summary CSVs")
        summary_rows = []
        for i, st_a in enumerate(subtypes_all):
            for j, st_b in enumerate(subtypes_all):
                if i >= j: continue
                summary_rows.append({
                    "Subtype_A": st_a, "Group_A": subtype_group[st_a],
                    "Subtype_B": st_b, "Group_B": subtype_group[st_b],
                    "Spearman_divergence": round(dist_mat[i,j], 6),
                    "Same_group": subtype_group[st_a] == subtype_group[st_b],
                })
        pd.DataFrame(summary_rows).sort_values("Spearman_divergence").to_csv(
            os.path.join(CROSS_DIR, "C9_pairwise_divergence_all_subtypes.csv"), index=False)

        # Intra vs inter-group divergence boxplot
        div_df = pd.DataFrame(summary_rows)
        fig, ax = plt.subplots(figsize=(8, 6))
        colors_box = {"True": "#3498db", "False": "#e74c3c"}
        for same, grp in div_df.groupby("Same_group"):
            lbl = "Within group" if same else "Between groups"
            ax.hist(grp["Spearman_divergence"], bins=40, alpha=0.5,
                    color=colors_box[str(same)], label=f"{lbl} (n={len(grp):,})")
        ax.set_xlabel("Spearman Divergence (1 - \u03c1)", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title("Intra- vs Inter-Group Regulatory Divergence",
                     fontsize=13, fontweight="bold")
        _smart_legend(ax, fontsize=10, outside=False)
        plt.tight_layout()
        _save(fig, os.path.join(CROSS_DIR, "C9_intra_vs_inter_divergence"))
        log_step("CROSS","C9","saved")

        # Mark complete
        with open(CROSS_CKPT, "w") as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S"))
        log_step("CROSS","COMPLETE","all cross-group outputs saved to " + CROSS_DIR)

elif os.path.exists(CROSS_CKPT):
    log_step("CROSS","SKIP","already completed")
else:
    log_step("CROSS","SKIP",f"need >=2 completed groups (have {len(completed)})")

_monitor_state["stop"] = True
log_step("MAIN","ALL DONE",f"{len(completed)}/{len(PREFIXES)} groups completed")