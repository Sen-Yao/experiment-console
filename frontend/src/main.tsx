import React from 'react'
import ReactDOM from 'react-dom/client'
import { Activity, AlertTriangle, Check, Play, RefreshCw, ShieldCheck, Square, Terminal } from 'lucide-react'
import './styles.css'

type Job = {
  job_id: string
  name: string
  status: string
  entity?: string | null
  project?: string | null
  sweep_id?: string | null
  remote_host?: string | null
  remote_cwd?: string | null
  agent_pids: string[]
  created_at: string
  updated_at: string
}

type Sweep = {
  id: string
  entity: string
  project: string
  state: string
  runCount: number
  expectedRunCount: number
  progress: number
}

type Overview = {
  status: string
  degraded?: string | null
  job_counts: Record<string, number>
  jobs: Job[]
  sweeps: Sweep[]
  active_sweeps: number
  finished_sweeps: number
  total_runs: number
  generated_at: string
}

type AuditEvent = {
  event_id: string
  event_type: string
  message: string
  intent_id?: string | null
  job_id?: string | null
  created_at: string
}

type IntentRecord = {
  intent_id: string
  intent: string
  status: string
  confirmation_phrase: string
  plan: {
    summary: string
    risk_level: string
    warnings: string[]
    expected_side_effects: string[]
    commands: Array<{ label: string; argv: string[]; reason: string; side_effect: boolean; host?: string | null }>
  }
}

type MetricProps = {
  label: string
  value: number
  icon: React.ReactNode
}

type FieldProps = {
  label: string
  value: string
  onChange: (nextValue: string) => void
}

const blankLaunch = {
  job_name: '',
  config_path: '',
  entity: 'my-team',
  project: 'my-project',
  remote_host: 'gpu-host-1',
  remote_cwd: '',
  conda_env: '',
  max_agents: ''
}

function App() {
  const [overview, setOverview] = React.useState<Overview | null>(null)
  const [events, setEvents] = React.useState<AuditEvent[]>([])
  const [intent, setIntent] = React.useState<IntentRecord | null>(null)
  const [confirmText, setConfirmText] = React.useState('')
  const [launch, setLaunch] = React.useState(blankLaunch)
  const [message, setMessage] = React.useState('')
  const [busy, setBusy] = React.useState(false)

  const refresh = React.useCallback(async () => {
    const [overviewResp, eventsResp] = await Promise.all([
      fetch('/api/overview'),
      fetch('/api/events?limit=40')
    ])
    setOverview(await overviewResp.json())
    setEvents(await eventsResp.json())
  }, [])

  React.useEffect(() => {
    refresh().catch((err) => setMessage(String(err)))
    const id = window.setInterval(() => refresh().catch(() => undefined), 30000)
    return () => window.clearInterval(id)
  }, [refresh])

  async function previewLaunch() {
    setBusy(true)
    setMessage('')
    try {
      const payload: Record<string, unknown> = {
        ...launch,
        max_agents: launch.max_agents ? Number(launch.max_agents) : null,
        conda_env: launch.conda_env || null
      }
      const resp = await fetch('/api/intents/preview', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ intent: 'launch_sweep', payload })
      })
      const data = await resp.json()
      if (!resp.ok) throw new Error(data.detail || 'preview failed')
      setIntent(data.intent)
      setConfirmText('')
    } catch (err) {
      setMessage(err instanceof Error ? err.message : String(err))
    } finally {
      setBusy(false)
    }
  }

  async function previewJobIntent(kind: 'status_query' | 'stop_job' | 'recover_agents', job: Job) {
    setBusy(true)
    setMessage('')
    try {
      const payload = kind === 'status_query' ? { job_id: job.job_id } : { job_id: job.job_id }
      const resp = await fetch('/api/intents/preview', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ intent: kind, payload })
      })
      const data = await resp.json()
      if (!resp.ok) throw new Error(data.detail || 'preview failed')
      setIntent(data.intent)
      setConfirmText('')
    } catch (err) {
      setMessage(err instanceof Error ? err.message : String(err))
    } finally {
      setBusy(false)
    }
  }

  async function confirmAndExecute() {
    if (!intent) return
    setBusy(true)
    setMessage('')
    try {
      if (intent.plan.risk_level !== 'read_only') {
        const confirmResp = await fetch(`/api/intents/${intent.intent_id}/confirm`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ confirmation_phrase: confirmText })
        })
        const confirmData = await confirmResp.json()
        if (!confirmResp.ok) throw new Error(confirmData.detail || 'confirm failed')
      }
      const execResp = await fetch(`/api/intents/${intent.intent_id}/execute`, { method: 'POST' })
      const execData = await execResp.json()
      if (!execResp.ok) throw new Error(execData.detail || 'execute failed')
      setMessage('Execution finished')
      setIntent(execData.intent)
      await refresh()
    } catch (err) {
      setMessage(err instanceof Error ? err.message : String(err))
    } finally {
      setBusy(false)
    }
  }

  const jobs = overview?.jobs ?? []
  const sweeps = overview?.sweeps ?? []

  return (
    <main>
      <header className="topbar">
        <div>
          <h1>Experiment Console</h1>
          <p>Local control plane for W&B sweeps, SSH GPU agents, and auditable experiment actions.</p>
        </div>
        <button className="iconButton" onClick={() => refresh()} title="Refresh">
          <RefreshCw size={18} />
        </button>
      </header>

      {message && <div className="notice"><AlertTriangle size={16} />{message}</div>}
      {overview?.degraded && <div className="notice muted"><ShieldCheck size={16} />Degraded W&B view: {overview.degraded}</div>}

      <section className="stats">
        <Metric label="Running Jobs" value={overview?.job_counts.running ?? 0} icon={<Activity size={20} />} />
        <Metric label="Attention" value={overview?.job_counts.attention ?? 0} icon={<AlertTriangle size={20} />} />
        <Metric label="Active Sweeps" value={overview?.active_sweeps ?? 0} icon={<Play size={20} />} />
        <Metric label="Total Runs" value={overview?.total_runs ?? 0} icon={<Terminal size={20} />} />
      </section>

      <section className="workspace">
        <div className="panel launchPanel">
          <div className="panelHeader">
            <h2>Launch Sweep</h2>
            <button onClick={previewLaunch} disabled={busy}><Play size={16} />Preview</button>
          </div>
          <div className="formGrid">
            <Field label="Job name" value={launch.job_name} onChange={(v) => setLaunch({ ...launch, job_name: v })} />
            <Field label="Config path" value={launch.config_path} onChange={(v) => setLaunch({ ...launch, config_path: v })} />
            <Field label="Entity" value={launch.entity} onChange={(v) => setLaunch({ ...launch, entity: v })} />
            <Field label="Project" value={launch.project} onChange={(v) => setLaunch({ ...launch, project: v })} />
            <Field label="Remote host" value={launch.remote_host} onChange={(v) => setLaunch({ ...launch, remote_host: v })} />
            <Field label="Remote cwd" value={launch.remote_cwd} onChange={(v) => setLaunch({ ...launch, remote_cwd: v })} />
            <Field label="Conda env" value={launch.conda_env} onChange={(v) => setLaunch({ ...launch, conda_env: v })} />
            <Field label="Max agents" value={launch.max_agents} onChange={(v) => setLaunch({ ...launch, max_agents: v })} />
          </div>
        </div>

        <div className="panel">
          <div className="panelHeader"><h2>Intent Gate</h2></div>
          {intent ? (
            <div className="intent">
              <div className={`risk ${intent.plan.risk_level}`}>{intent.plan.risk_level}</div>
              <h3>{intent.plan.summary}</h3>
              <div className="commandList">
                {intent.plan.commands.map((cmd) => (
                  <div className="command" key={cmd.label}>
                    <strong>{cmd.label}</strong>
                    <code>{cmd.argv.join(' ')}</code>
                    <span>{cmd.reason}</span>
                  </div>
                ))}
              </div>
              {intent.plan.risk_level !== 'read_only' && (
                <label className="field">
                  <span>Confirmation phrase</span>
                  <code className="phrase">{intent.confirmation_phrase}</code>
                  <input value={confirmText} onChange={(e) => setConfirmText(e.target.value)} />
                </label>
              )}
              <button
                className={intent.plan.risk_level === 'read_only' ? '' : 'danger'}
                onClick={confirmAndExecute}
                disabled={busy || (intent.plan.risk_level !== 'read_only' && confirmText !== intent.confirmation_phrase)}
              >
                <Check size={16} />{intent.plan.risk_level === 'read_only' ? 'Execute Read-only Query' : 'Confirm & Execute'}
              </button>
            </div>
          ) : (
            <div className="empty">Preview an action to generate a plan and confirmation phrase.</div>
          )}
        </div>
      </section>

      <section className="gridTwo">
        <div className="panel">
          <div className="panelHeader"><h2>Jobs</h2></div>
          <div className="table">
            {jobs.map((job) => (
              <div className="row" key={job.job_id}>
                <div>
                  <strong>{job.name}</strong>
                  <span>{job.job_id}</span>
                </div>
                <Status value={job.status} />
                <span>{job.remote_host || '-'}</span>
                <span>{job.sweep_id || '-'}</span>
                <div className="actions">
                  <button onClick={() => previewJobIntent('status_query', job)} title="Status"><RefreshCw size={15} /></button>
                  <button onClick={() => previewJobIntent('recover_agents', job)} title="Recover"><Play size={15} /></button>
                  <button onClick={() => previewJobIntent('stop_job', job)} title="Stop"><Square size={15} /></button>
                </div>
              </div>
            ))}
            {!jobs.length && <div className="empty">No local console jobs yet.</div>}
          </div>
        </div>

        <div className="panel">
          <div className="panelHeader"><h2>W&B Sweeps</h2></div>
          <div className="sweepList">
            {sweeps.slice(0, 10).map((sweep) => (
              <div className="sweep" key={`${sweep.project}/${sweep.id}`}>
                <div>
                  <strong>{sweep.id}</strong>
                  <span>{sweep.entity}/{sweep.project}</span>
                </div>
                <Status value={sweep.state.toLowerCase()} />
                <span>{sweep.runCount}/{sweep.expectedRunCount}</span>
              </div>
            ))}
            {!sweeps.length && <div className="empty">No W&B sweeps loaded.</div>}
          </div>
        </div>
      </section>

      <section className="panel">
        <div className="panelHeader"><h2>Audit</h2></div>
        <div className="events">
          {events.map((event) => (
            <div className="event" key={event.event_id}>
              <span>{new Date(event.created_at).toLocaleString()}</span>
              <strong>{event.event_type}</strong>
              <p>{event.message}</p>
            </div>
          ))}
        </div>
      </section>
    </main>
  )
}

function Metric({ label, value, icon }: MetricProps) {
  return (
    <div className="metric">
      {icon}
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  )
}

function Field({ label, value, onChange }: FieldProps) {
  return (
    <label className="field">
      <span>{label}</span>
      <input value={value} onChange={(event) => onChange(event.target.value)} />
    </label>
  )
}

function Status({ value }: { value: string }) {
  return <span className={`status ${value}`}>{value}</span>
}

ReactDOM.createRoot(document.getElementById('root') as HTMLElement).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
)
