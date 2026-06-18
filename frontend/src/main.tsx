import React from 'react'
import ReactDOM from 'react-dom/client'
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  Clock3,
  Gauge,
  ListChecks,
  RefreshCw,
  Server,
  Timer,
} from 'lucide-react'
import experimentLogoUrl from './assets/experiment-results.svg'
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
  monitor?: {
    kind?: string
    run?: {
      log?: string
      status_path?: string
    }
  }
  created_at?: string
  updated_at?: string
}

type Sweep = {
  id: string
  name?: string
  entity: string
  project: string
  state: string
  runCount: number
  expectedRunCount: number
  progress: number
  createdAt?: string
  source?: string
  finished_runs?: number
  running_runs?: number
  failed_runs?: number
  speed_per_hour?: number
  eta_seconds?: number | null
  last_sync_at?: string
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
  event_type: string
  message: string
  created_at: string
  detail?: Record<string, unknown>
}

type TabKey = 'sweeps' | 'jobs' | 'events'
type TimePartMap = Record<string, string>

const SHANGHAI_TIME_ZONE = 'Asia/Shanghai'
const dateTimeFormatter = new Intl.DateTimeFormat('en-CA', {
  timeZone: SHANGHAI_TIME_ZONE,
  year: 'numeric',
  month: '2-digit',
  day: '2-digit',
  hour: '2-digit',
  minute: '2-digit',
  second: '2-digit',
  hour12: false,
  hourCycle: 'h23',
})
const shortTimeFormatter = new Intl.DateTimeFormat('en-CA', {
  timeZone: SHANGHAI_TIME_ZONE,
  month: '2-digit',
  day: '2-digit',
  hour: '2-digit',
  minute: '2-digit',
  hour12: false,
  hourCycle: 'h23',
})

function App() {
  const [overview, setOverview] = React.useState<Overview | null>(null)
  const [events, setEvents] = React.useState<AuditEvent[]>([])
  const [tab, setTab] = React.useState<TabKey>('sweeps')
  const [loading, setLoading] = React.useState(false)
  const [message, setMessage] = React.useState('')

  const refresh = React.useCallback(async () => {
    setLoading(true)
    setMessage('')
    try {
      const [overviewResp, eventsResp] = await Promise.all([
        fetch('/api/overview'),
        fetch('/api/events?limit=40'),
      ])
      if (!overviewResp.ok) throw new Error(`overview ${overviewResp.status}`)
      if (!eventsResp.ok) throw new Error(`events ${eventsResp.status}`)
      setOverview(await overviewResp.json())
      setEvents(await eventsResp.json())
    } catch (err) {
      setMessage(err instanceof Error ? err.message : String(err))
    } finally {
      setLoading(false)
    }
  }, [])

  React.useEffect(() => {
    refresh()
    const id = window.setInterval(refresh, 30000)
    return () => window.clearInterval(id)
  }, [refresh])

  const initialLoading = !overview && loading
  const sweeps = overview?.sweeps ?? []
  const jobs = overview?.jobs ?? []
  const primary = selectPrimarySweep(sweeps)
  const grouped = groupSweeps(sweeps)

  return (
    <main className="consoleShell">
      <header className="systemBar">
        <div className="brandBlock">
          <img className="brandLogo" src={experimentLogoUrl} alt="" />
          <div className="brandText">
            <span className="eyebrow">Experiment Console</span>
            <h1>实验控制台</h1>
          </div>
        </div>
        <div className="systemReadouts" aria-label="系统状态">
          <StatusPill status={overview?.status ?? 'loading'} degraded={overview?.degraded} />
          <MiniReadout icon={<Activity size={15} />} label="活跃 Sweep" value={String(overview?.active_sweeps ?? 0)} />
          <MiniReadout icon={<Server size={15} />} label="运行作业" value={String(overview?.job_counts.running ?? 0)} />
          <MiniReadout icon={<Clock3 size={15} />} label="同步" value={formatDateTime(overview?.generated_at)} />
          <button className="iconButton" onClick={refresh} title="刷新" aria-label="刷新" disabled={loading}>
            <RefreshCw size={18} />
          </button>
        </div>
      </header>

      {message && <div className="notice"><AlertTriangle size={16} />{message}</div>}
      {overview?.degraded && <div className="notice amber"><AlertTriangle size={16} />W&B 视图降级：{overview.degraded}</div>}

      <HeroSweepPanel sweep={primary} loading={initialLoading} />

      <section className="tabsPanel">
        <div className="tabs" role="tablist" aria-label="控制台分区">
          <TabButton active={tab === 'sweeps'} onClick={() => setTab('sweeps')} icon={<Gauge size={16} />} label="Sweep" />
          <TabButton active={tab === 'jobs'} onClick={() => setTab('jobs')} icon={<ListChecks size={16} />} label="作业" />
          <TabButton active={tab === 'events'} onClick={() => setTab('events')} icon={<Clock3 size={16} />} label="审计" />
        </div>

        {tab === 'sweeps' && <CompactSweepList grouped={grouped} />}
        {tab === 'jobs' && <JobList jobs={jobs} />}
        {tab === 'events' && <EventList events={events} />}
      </section>
    </main>
  )
}

function HeroSweepPanel({ sweep, loading }: { sweep?: Sweep; loading?: boolean }) {
  if (loading) {
    return (
      <section className="heroPanel emptyHero loadingHero">
        <div>
          <span className="sectionLabel">主监控</span>
          <h2>正在同步实验读数</h2>
          <p>正在从 Console 控制面读取当前 Sweep 和作业状态。</p>
        </div>
      </section>
    )
  }
  if (!sweep) {
    return (
      <section className="heroPanel emptyHero">
        <div>
          <span className="sectionLabel">主监控</span>
          <h2>暂无可展示 Sweep</h2>
        </div>
      </section>
    )
  }
  const expected = Math.max(0, sweep.expectedRunCount || 0)
  const finished = Math.max(0, sweep.finished_runs ?? (sweep.state === 'FINISHED' ? expected : sweep.runCount || 0))
  const running = Math.max(0, sweep.running_runs ?? 0)
  const failed = Math.max(0, sweep.failed_runs ?? 0)
  const progress = expected > 0 ? Math.min(finished / expected, 1) : Math.min(sweep.progress || 0, 1)

  return (
    <section className={`heroPanel tone-${toneForStatus(sweep.state)}`}>
      <div className="heroTopline">
        <div>
          <span className="sectionLabel">主 Sweep</span>
          <h2>{sweep.name || sweep.id}</h2>
          <p>{sweep.entity}/{sweep.project}</p>
        </div>
        <span className="stateBadge">{statusText(sweep.state)}</span>
      </div>

      <div className="progressReadout">
        <div className="bigNumber">
          <strong>{finished}</strong>
          <span>/ {expected || '-'}</span>
        </div>
        <div className="progressTrack" aria-label="Sweep 进度">
          <span style={{ width: `${Math.round(progress * 100)}%` }} />
        </div>
        <div className="percent">{Math.round(progress * 100)}%</div>
      </div>

      <TelemetryStrip sweep={sweep} />

      <div className="stateLedger">
        <LedgerItem label="运行中" value={running} tone="running" />
        <LedgerItem label="已完成" value={finished} tone="finished" />
        <LedgerItem label="失败" value={failed} tone="failed" />
      </div>
    </section>
  )
}

function TelemetryStrip({ sweep }: { sweep: Sweep }) {
  return (
    <div className="telemetryStrip">
      <Telemetry icon={<Timer size={16} />} label="ETA" value={formatEta(sweep.eta_seconds)} />
      <Telemetry icon={<Clock3 size={16} />} label="预计完成" value={formatExpectedCompletion(sweep.last_sync_at, sweep.eta_seconds)} />
      <Telemetry icon={<Gauge size={16} />} label="速度" value={formatSpeed(sweep.speed_per_hour)} />
    </div>
  )
}

function CompactSweepList({ grouped }: { grouped: Record<string, Sweep[]> }) {
  const sections: Array<[string, Sweep[]]> = [
    ['运行中', grouped.running],
    ['已完成', grouped.finished],
    ['其他', grouped.other],
  ]
  return (
    <div className="compactList">
      {sections.map(([label, items]) => (
        <section className="listSection" key={label}>
          <h3>{label}</h3>
          {items.length === 0 ? (
            <div className="emptyLine">暂无</div>
          ) : (
            items.map((sweep) => <SweepRow key={`${sweep.entity}/${sweep.project}/${sweep.id}`} sweep={sweep} />)
          )}
        </section>
      ))}
    </div>
  )
}

function SweepRow({ sweep }: { sweep: Sweep }) {
  const expected = sweep.expectedRunCount || 0
  const finished = sweep.finished_runs ?? sweep.runCount ?? 0
  const pct = expected ? Math.min(finished / expected, 1) : sweep.progress || 0
  return (
    <div className="sweepRow">
      <div>
        <strong>{sweep.name || sweep.id}</strong>
        <span>{sweep.entity}/{sweep.project}</span>
      </div>
      <span className={`smallBadge tone-${toneForStatus(sweep.state)}`}>{statusText(sweep.state)}</span>
      <code>{finished}/{expected || '-'}</code>
      <div className="miniTrack"><span style={{ width: `${Math.round(pct * 100)}%` }} /></div>
    </div>
  )
}

function JobList({ jobs }: { jobs: Job[] }) {
  if (!jobs.length) return <div className="emptyLine">暂无 Runner 作业</div>
  return (
    <div className="jobList">
      {jobs.map((job) => {
        const target = job.sweep_id || (job.monitor?.kind === 'single_run' ? 'single-run' : '-')
        return (
          <div className="jobRow" key={job.job_id}>
            <div>
              <strong>{job.name || job.job_id}</strong>
              <span>{job.job_id}</span>
            </div>
            <span className={`smallBadge tone-${toneForStatus(job.status)}`}>{statusText(job.status)}</span>
            <code title={job.monitor?.run?.log || undefined}>{target}</code>
            <span>{job.remote_host || '-'}</span>
          </div>
        )
      })}
    </div>
  )
}

function EventList({ events }: { events: AuditEvent[] }) {
  if (!events.length) return <div className="emptyLine">暂无审计记录</div>
  return (
    <div className="eventList">
      {events.map((event, index) => (
        <div className="eventRow" key={`${event.created_at}-${index}`}>
          <span>{formatDateTime(event.created_at)}</span>
          <strong>{event.event_type}</strong>
          <p>{event.message}</p>
        </div>
      ))}
    </div>
  )
}

function MiniReadout({ icon, label, value }: { icon: React.ReactNode; label: string; value: string }) {
  return (
    <div className="miniReadout">
      {icon}
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  )
}

function Telemetry({ icon, label, value }: { icon: React.ReactNode; label: string; value: string }) {
  return (
    <div className="telemetry">
      {icon}
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  )
}

function LedgerItem({ label, value, tone }: { label: string; value: number; tone: string }) {
  return (
    <div className={`ledgerItem ${tone}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  )
}

function TabButton({ active, onClick, icon, label }: { active: boolean; onClick: () => void; icon: React.ReactNode; label: string }) {
  return (
    <button className={active ? 'tab active' : 'tab'} onClick={onClick}>
      {icon}
      {label}
    </button>
  )
}

function StatusPill({ status, degraded }: { status: string; degraded?: string | null }) {
  if (status === 'loading') {
    return (
      <span className="statusPill syncing">
        <RefreshCw size={15} />
        正在同步
      </span>
    )
  }
  const ok = status === 'ok' && !degraded
  return (
    <span className={ok ? 'statusPill ok' : 'statusPill attention'}>
      {ok ? <CheckCircle2 size={15} /> : <AlertTriangle size={15} />}
      {ok ? '控制台运行正常' : '控制台需关注'}
    </span>
  )
}

function selectPrimarySweep(sweeps: Sweep[]) {
  return [...sweeps].sort((a, b) => {
    const ar = a.state === 'RUNNING' ? 1 : 0
    const br = b.state === 'RUNNING' ? 1 : 0
    if (ar !== br) return br - ar
    return String(b.createdAt || '').localeCompare(String(a.createdAt || ''))
  })[0]
}

function groupSweeps(sweeps: Sweep[]) {
  return {
    running: sweeps.filter((s) => s.state === 'RUNNING'),
    finished: sweeps.filter((s) => s.state === 'FINISHED'),
    other: sweeps.filter((s) => !['RUNNING', 'FINISHED'].includes(s.state)),
  }
}

function toneForStatus(status: string) {
  const s = status.toLowerCase()
  if (['running', 'pending'].includes(s)) return 'running'
  if (['finished', 'complete', 'completed'].includes(s)) return 'finished'
  if (['failed', 'crashed', 'killed'].includes(s)) return 'failed'
  if (['attention', 'stalled'].includes(s)) return 'attention'
  return 'neutral'
}

function statusText(status: string) {
  const s = status.toLowerCase()
  if (s === 'running') return '运行中'
  if (s === 'finished') return '已完成'
  if (s === 'failed') return '失败'
  if (s === 'cancelled' || s === 'canceled') return '已取消'
  if (s === 'attention') return '需关注'
  if (s === 'stalled') return '已悬挂'
  return status || '未知'
}

function formatterParts(formatter: Intl.DateTimeFormat, date: Date): TimePartMap {
  return Object.fromEntries(formatter.formatToParts(date).map((part) => [part.type, part.value]))
}

function parseTime(value?: string) {
  if (!value) return null
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return null
  return date
}

function formatDateTime(value?: string) {
  const date = parseTime(value)
  if (!date) return '-'
  const parts = formatterParts(dateTimeFormatter, date)
  return `${parts.year}-${parts.month}-${parts.day} ${parts.hour}:${parts.minute}:${parts.second}`
}

function formatShortDateTime(date: Date) {
  const parts = formatterParts(shortTimeFormatter, date)
  return `${parts.month}-${parts.day} ${parts.hour}:${parts.minute}`
}

function formatExpectedCompletion(lastSync?: string, etaSeconds?: number | null) {
  const base = parseTime(lastSync)
  if (!base || !etaSeconds || etaSeconds <= 0) return '-'
  return formatShortDateTime(new Date(base.getTime() + etaSeconds * 1000))
}

function formatEta(seconds?: number | null) {
  if (!seconds || seconds <= 0) return '-'
  const hours = Math.floor(seconds / 3600)
  const minutes = Math.round((seconds % 3600) / 60)
  if (hours <= 0) return `${minutes} 分钟`
  return `${hours} 小时 ${minutes} 分钟`
}

function formatSpeed(speed?: number) {
  if (!speed || speed <= 0) return '-'
  return `${speed.toFixed(speed >= 10 ? 0 : 1)} runs/h`
}

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
)
