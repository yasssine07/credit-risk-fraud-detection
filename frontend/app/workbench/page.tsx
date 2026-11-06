"use client";

import { useEffect, useMemo, useState } from "react";
import { Activity, BarChart3, Database, FileUp, FlaskConical, Layers3, LoaderCircle, ShieldAlert, Sparkles } from "lucide-react";
import "./workbench.css";
import "./score.css";

type Mode = "profile" | "train" | "clusters" | "score";
type ApiResult = Record<string, any>;
const apiBase = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000/api/v1";

export default function WorkbenchPage() {
  const [mode, setMode] = useState<Mode>("profile");
  const [file, setFile] = useState<File | null>(null);
  const [target, setTarget] = useState("Class");
  const [model, setModel] = useState("random_forest");
  const [task, setTask] = useState("fraud");
  const [positiveLabel, setPositiveLabel] = useState("1");
  const [balancing, setBalancing] = useState("none");
  const [selection, setSelection] = useState("none");
  const [explain, setExplain] = useState(true);
  const [algorithm, setAlgorithm] = useState("kmeans");
  const [result, setResult] = useState<ApiResult | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [registeredModels, setRegisteredModels] = useState<ApiResult[]>([]);
  const [modelVersion, setModelVersion] = useState("");
  const [manualFeatures, setManualFeatures] = useState("{}");

  const title = useMemo(() => ({ profile: "Explore a dataset", train: "Train and evaluate", clusters: "Discover patterns", score: "Score one record" }[mode]), [mode]);

  useEffect(() => {
    fetch(`${apiBase}/models`).then((response) => response.json()).then((body) => {
      const models = body.models ?? []; setRegisteredModels(models);
      if (!modelVersion && models[0]) setModelVersion(models[0].model_version);
    }).catch(() => setRegisteredModels([]));
  }, [modelVersion]);

  async function run() {
    if (mode === "score") {
      const selected = registeredModels.find((item) => item.model_version === modelVersion);
      if (!selected) { setError("Train and register a model before scoring records."); return; }
      let features: Record<string, unknown>;
      try { features = JSON.parse(manualFeatures); } catch { setError("Enter features as a valid JSON object."); return; }
      setBusy(true); setError(""); setResult(null);
      try {
        const endpoint = selected.task === "credit_risk" ? "/predictions/credit-risk" : "/predictions/fraud";
        const response = await fetch(`${apiBase}${endpoint}`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ model_version: modelVersion, features }) });
        const body = await response.json();
        if (!response.ok) throw new Error(formatApiError(body.detail, "The API could not score this record."));
        setResult(body);
      } catch (reason) { setError(reason instanceof Error ? reason.message : "Could not connect to the scoring API."); }
      finally { setBusy(false); }
      return;
    }
    if (!file) { setError("Choose a CSV file to continue."); return; }
    setBusy(true); setError(""); setResult(null);
    const form = new FormData(); form.append("file", file);
    if (mode !== "clusters" && target.trim()) form.append("target_column", target.trim());
    let endpoint = "/datasets/profile";
    if (mode === "train") {
      endpoint = "/training/evaluate";
      form.append("model_name", model); form.append("balance_strategy", balancing);
      form.append("feature_selection", selection); form.append("include_explanations", String(explain));
      form.append("task", task); form.append("positive_label", positiveLabel);
    }
    if (mode === "clusters") {
      endpoint = "/clustering/discover"; form.append("algorithm", algorithm); form.append("target_column", target.trim());
    }
    try {
      const response = await fetch(`${apiBase}${endpoint}`, { method: "POST", body: form });
      const body = await response.json();
      if (!response.ok) throw new Error(formatApiError(body.detail, "The API could not process this CSV."));
      setResult(body);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not connect to the API. Check that FastAPI is running.");
    } finally { setBusy(false); }
  }

  return <main className="workspace-shell">
    <header className="workspace-topbar"><a className="work-brand" href="/"><span><ShieldAlert size={17}/></span>finrisk<span className="blue-word">.ai</span></a><div className="topbar-status"><i/> LOCAL RESEARCH WORKSPACE</div><a className="back-link" href="/">Back to overview</a></header>
    <section className="workspace-intro"><div className="work-eyebrow">MODEL LABORATORY <span> / </span> DATA WORKSPACE</div><h1>Make the evidence <em>visible.</em></h1><p>Inspect an uploaded dataset, evaluate a real baseline model, or explore behavioral clusters. No scores appear until you provide data.</p></section>
    <section className="work-layout">
      <aside className="work-sidebar"><div className="side-caption">WORKSPACE</div>
        <button className={mode === "profile" ? "side-item active" : "side-item"} onClick={() => {setMode("profile");setResult(null);}}><Database size={16}/> Data explorer</button>
        <button className={mode === "train" ? "side-item active" : "side-item"} onClick={() => {setMode("train");setResult(null);}}><FlaskConical size={16}/> Model evaluation</button>
        <button className={mode === "clusters" ? "side-item active" : "side-item"} onClick={() => {setMode("clusters");setResult(null);}}><Layers3 size={16}/> Pattern discovery</button>
        <button className={mode === "score" ? "side-item active" : "side-item"} onClick={() => {setMode("score");setResult(null);}}><Activity size={16}/> Score a record</button>
        <div className="side-note"><Sparkles size={15}/><b>Signals need context</b><p>Model outputs support investigation. They do not establish fraud or make credit decisions.</p></div>
      </aside>
      <div className="work-content"><div className="work-heading"><div><div className="work-eyebrow">{mode === "profile" ? "01 / DATA QUALITY" : mode === "train" ? "02 / SUPERVISED LEARNING" : mode === "clusters" ? "03 / UNSUPERVISED LEARNING" : "04 / MODEL INFERENCE"}</div><h2>{title}</h2></div><span className="actual-data-tag"><i/> REAL DATA ONLY</span></div>
        <div className="input-panel">{mode === "score" ? <><div className="panel-label"><Activity size={16}/> REGISTERED MODEL <small>REAL TRAINED ARTIFACT</small></div><div className="form-grid score-form"><label>Model version<select value={modelVersion} onChange={(event) => setModelVersion(event.target.value)}>{registeredModels.map((item) => <option key={item.model_version} value={item.model_version}>{item.task} · {item.model_name} · {item.model_version.slice(0, 8)}</option>)}</select><small>{registeredModels.find((item) => item.model_version === modelVersion)?.feature_names?.join(", ") ?? "No registered model available yet."}</small></label><label>Record features (JSON)<textarea value={manualFeatures} onChange={(event) => setManualFeatures(event.target.value)} placeholder={'{"Amount": 42.5, "V1": -0.12}'}/><small>Use the feature names listed above. Omitted fields are imputed by the fitted pipeline.</small></label></div></> : <><div className="panel-label"><FileUp size={16}/> SOURCE FILE <small>CSV · up to 250 MiB</small></div>
          <label className="dropzone"><input type="file" accept=".csv,text/csv" onChange={(event) => {setFile(event.target.files?.[0] ?? null);setResult(null);}}/><span className="upload-icon"><FileUp size={20}/></span><b>{file?.name ?? "Choose a transaction or credit dataset"}</b><small>{file ? `${(file.size / 1024).toFixed(1)} KB · ready to process` : "Select a CSV from your device"}</small><span className="browse-button">Browse files</span></label>
          <div className="form-grid"><label>Target label <input value={target} onChange={(event) => setTarget(event.target.value)} placeholder="e.g. Class"/><small>Used for profiling and supervised training.</small></label>
            {mode === "train" && <><label>Prediction task <select value={task} onChange={(event) => setTask(event.target.value)}><option value="fraud">Fraud detection</option><option value="credit_risk">Credit risk</option></select><small>Controls downstream risk labels.</small></label><label>Positive target label <input value={positiveLabel} onChange={(event) => setPositiveLabel(event.target.value)} placeholder="e.g. 1 or bad"/><small>Class treated as fraud/default.</small></label><label>Baseline model <select value={model} onChange={(event) => setModel(event.target.value)}><option value="random_forest">Random Forest</option><option value="logistic_regression">Logistic Regression</option><option value="decision_tree">Decision Tree</option><option value="support_vector_machine">Support Vector Machine</option><option value="k_nearest_neighbors">K-Nearest Neighbors</option><option value="gradient_boosting">Gradient Boosting</option></select><small>One model per evaluation run. Registered for later scoring.</small></label><label>Imbalance method <select value={balancing} onChange={(event) => setBalancing(event.target.value)}><option value="none">No resampling</option><option value="random_over">Random oversampling</option><option value="random_under">Random undersampling</option><option value="smote">SMOTE</option><option value="adasyn">ADASYN</option><option value="smoteenn">SMOTEENN</option><option value="class_weight">Class weighting</option></select><small>Applied inside the training pipeline.</small></label><label>Feature selection <select value={selection} onChange={(event) => setSelection(event.target.value)}><option value="none">All features</option><option value="mutual_info">Mutual information</option><option value="rfe">Recursive elimination</option><option value="random_forest">Random Forest importance</option></select><small>Selection is learned from training data.</small></label><label className="check-field"><input type="checkbox" checked={explain} onChange={(event) => setExplain(event.target.checked)}/><span><b>Include SHAP explanations</b><small>Bounded global and local contributions.</small></span></label></>}
            {mode === "clusters" && <label>Clustering algorithm <select value={algorithm} onChange={(event) => setAlgorithm(event.target.value)}><option value="kmeans">K-Means</option><option value="dbscan">DBSCAN</option><option value="hierarchical">Hierarchical</option><option value="gaussian_mixture">Gaussian mixture</option></select><small>Target is excluded from numeric features.</small></label>}
          </div></>}
          <div className="submit-row">
            <span>{mode === "score" ? "Scores use a locally registered, trained model." : "Rows are processed in memory for this request."}</span>
            <button onClick={run} disabled={busy}>
              {busy ? <><LoaderCircle className="spin" size={16}/> Processing</> : <>{mode === "profile" ? <BarChart3 size={16}/> : <Activity size={16}/>} {mode === "profile" ? "Generate profile" : mode === "train" ? "Evaluate model" : mode === "clusters" ? "Discover patterns" : "Score record"}</>}
            </button>
          </div>
        </div>
        {error && <div className="api-error" role="alert">{error}</div>}
        {result && <ResultsPanel mode={mode} result={result}/>}
      </div>
    </section>
    <footer className="work-footer"><span>FINRISK AI · RESEARCH FOUNDATION</span><span>Predictions are analytical estimates, not decisions.</span></footer>
  </main>;
}

function ResultsPanel({ mode, result }: { mode: Mode; result: ApiResult }) {
  if (mode === "score") return <section className="result-panel"><div className="result-title"><div><span className="work-eyebrow">MODEL ESTIMATE</span><h3>{result.classification}</h3></div><span className="result-file">{result.task} · {result.model_version.slice(0, 8)}</span></div><div className="result-stats"><Stat label={result.task === "credit_risk" ? "Default probability" : "Fraud probability"} value={percent(result.probability)} emphasis/>{result.credit_risk_score !== undefined && <Stat label="Credit risk score" value={result.credit_risk_score}/>}<Stat label="Low threshold" value={percent(result.thresholds.low)}/><Stat label="High threshold" value={percent(result.thresholds.high)}/></div>{result.explanation?.available && <div className="data-block"><h4>Local SHAP contributions</h4><div className="importance-list">{result.explanation.top_contributions.map((item: any) => <div key={item.feature}><span>{item.feature}</span><div><i style={{width:`${Math.max(3, Math.abs(item.shap_value) * 100)}%`}}/></div><b>{item.shap_value.toFixed(4)}</b></div>)}</div><p className="result-notice">Relative to the mean transformed training vector. These contributions describe model behavior, not causes.</p></div>}<p className="result-notice">{result.notice}</p></section>;
  if (mode === "profile") {
    const profile = result.profile;
    return <section className="result-panel"><div className="result-title"><div><span className="work-eyebrow">OBSERVED DATA</span><h3>Dataset profile</h3></div><span className="result-file">{result.filename}</span></div><div className="result-stats"><Stat label="Rows" value={profile.row_count}/><Stat label="Columns" value={profile.column_count}/><Stat label="Missing cells" value={profile.total_missing_cells}/><Stat label="Duplicate rows" value={profile.duplicate_row_count}/></div>{profile.target && <div className="data-block"><h4>Target distribution · {profile.target.column}</h4><div className="class-list">{profile.target.class_counts.map((item: any) => <div key={item.label}><span>{item.label}</span><b>{item.count.toLocaleString()}</b><i style={{width:`${Math.max(2, item.count / profile.row_count * 100)}%`}}/></div>)}</div></div>}<div className="data-block"><h4>Column summary</h4><div className="table-scroll"><table><thead><tr><th>Feature</th><th>Type</th><th>Missing</th><th>Unique</th><th>Summary</th></tr></thead><tbody>{profile.columns.map((column: any) => <tr key={column.name}><td>{column.name}</td><td>{column.dtype}</td><td>{column.missing_percent}%</td><td>{column.unique_count}</td><td>{column.numeric_summary ? `Median ${column.numeric_summary.median}` : "Categorical"}</td></tr>)}</tbody></table></div></div><p className="result-notice">{profile.notice}</p></section>;
  }
  if (mode === "train") {
    const report = result.validation; const metrics = report.metrics;
    return <section className="result-panel"><div className="result-title"><div><span className="work-eyebrow">VALIDATION PARTITION</span><h3>Model evaluation</h3></div><span className="result-file">{result.model?.replaceAll("_", " ")}</span></div><div className="result-stats"><Stat label="Recall" value={percent(metrics.recall)} emphasis/><Stat label="PR-AUC" value={percent(metrics.pr_auc_average_precision)} emphasis/><Stat label="Precision" value={percent(metrics.precision)}/><Stat label="Specificity" value={percent(metrics.specificity)}/><Stat label="ROC-AUC" value={percent(metrics.roc_auc)}/><Stat label="Balanced accuracy" value={percent(metrics.balanced_accuracy)}/></div><div className="data-block"><h4>Confusion matrix</h4><div className="confusion-grid"><span>Actual \ Predicted</span>{report.confusion_matrix.labels.map((label: any) => <b key={`h-${label}`}>{label}</b>)}{report.confusion_matrix.values.map((row: number[], index: number) => <div className="confusion-row" key={index}><b>{report.confusion_matrix.labels[index]}</b>{row.map((value, i) => <strong key={i}>{value.toLocaleString()}</strong>)}</div>)}</div></div>{result.explanations && <div className="data-block"><h4>Global SHAP importance</h4><div className="importance-list">{result.explanations.global_importance.map((item: any) => <div key={item.feature}><span>{item.feature}</span><div><i style={{width:`${Math.max(3,item.mean_absolute_shap / result.explanations.global_importance[0].mean_absolute_shap * 100)}%`}}/></div><b>{item.mean_absolute_shap.toFixed(4)}</b></div>)}</div><p className="result-notice">{result.explanations.notice}</p></div>}<p className="result-notice">{result.notice}</p></section>;
  }
  return <section className="result-panel"><div className="result-title"><div><span className="work-eyebrow">BEHAVIORAL PATTERNS</span><h3>Cluster discovery</h3></div><span className="result-file">{result.algorithm} · {result.reducer}</span></div><div className="result-stats"><Stat label="Rows analyzed" value={result.row_count}/><Stat label="Clusters" value={result.cluster_count_excluding_noise}/><Stat label="Noise points" value={result.noise_row_count}/><Stat label="Silhouette" value={result.metrics.silhouette?.toFixed(3) ?? "—"}/></div><div className="data-block"><h4>Cluster sizes</h4><div className="class-list">{result.cluster_sizes.map((item: any) => <div key={item.cluster}><span>{item.cluster === -1 ? "Noise" : `Cluster ${item.cluster}`}</span><b>{item.count.toLocaleString()}</b><i style={{width:`${Math.max(2,item.count/result.row_count*100)}%`}}/></div>)}</div></div><p className="result-notice">{result.notice}</p></section>;
}

function Stat({ label, value, emphasis = false }: { label: string; value: string | number; emphasis?: boolean }) { return <div className={emphasis ? "result-stat emphasis" : "result-stat"}><small>{label}</small><b>{value}</b></div>; }
function percent(value: number) { return `${(value * 100).toFixed(1)}%`; }
function formatApiError(detail: unknown, fallback: string) {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) return detail.map((issue: any) => `${Array.isArray(issue.loc) ? issue.loc.slice(1).join(".") : "Input"}: ${issue.msg ?? "Invalid value"}`).join("; ");
  return fallback;
}
