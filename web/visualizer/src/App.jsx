import { useCallback, useEffect, useRef, useState } from 'react'
import { Link, NavLink, Route, Routes, useLocation, useParams, useSearchParams } from 'react-router-dom'

const API = '/api/web/v1'

export async function api(path, options = {}) {
  const response = await fetch(`${API}${path}`, {
    headers: { 'Content-Type': 'application/json', ...options.headers },
    ...options,
  })
  const contentType = response.headers?.get?.('content-type') || ''
  if (contentType && !contentType.toLowerCase().includes('application/json')) {
    const body = await response.text().catch(() => '')
    const receivedHtml = contentType.toLowerCase().includes('text/html') || /^\s*<!doctype\s+html/i.test(body)
    if (receivedHtml) throw new Error('后端未提供此 API，可能仍在运行旧版本。请重启 tau3 web 后刷新页面。')
    throw new Error(`API 返回了不支持的内容类型：${contentType}`)
  }
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}))
    throw new Error(payload.detail || `${response.status} ${response.statusText}`)
  }
  return response.json()
}

function useLoad(loader, dependencies = [], interval = 0) {
  const [state, setState] = useState({ data: null, error: null, loading: true })
  const load = useCallback(async () => {
    try { setState((old) => ({ ...old, error: null })); const data = await loader(); setState({ data, error: null, loading: false }) }
    catch (error) { setState({ data: null, error: error.message, loading: false }) }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, dependencies)
  useEffect(() => {
    let cancelled = false
    let timer
    async function tick() {
      await load()
      if (!cancelled && interval) timer = setTimeout(tick, interval)
    }
    tick()
    return () => { cancelled = true; if (timer) clearTimeout(timer) }
  }, [load, interval])
  return { ...state, reload: load }
}

function useIndexedLoad(loader, dependencies = [], interval = 5000) {
  const state = useLoad(loader, dependencies)
  const reload = state.reload
  const revision = useRef(null)
  useEffect(() => {
    let cancelled = false
    let timer
    async function tick() {
      try {
        if (!document.hidden) {
          const health = await api('/health')
          const nextRevision = health?.index?.revision
          if (revision.current == null) revision.current = nextRevision
          else if (nextRevision !== revision.current) {
            revision.current = nextRevision
            await reload()
          }
        }
      } catch { /* The page's main request remains the visible error source. */ }
      if (!cancelled) timer = setTimeout(tick, interval)
    }
    timer = setTimeout(tick, interval)
    return () => { cancelled = true; if (timer) clearTimeout(timer) }
  }, [reload, interval])
  return state
}

function Layout() {
  const links = [['/', '总览'], ['/classic-view', '经典 View'], ['/runs', '运行轨迹'], ['/synthetic-packages', '合成任务包'], ['/documents', '种子文档'], ['/database', '种子数据库'], ['/calibration', '校准与作业']]
  return <div className="shell">
    <aside><div className="brand"><span>τ</span><div><strong>tau3</strong><small>可视化与校准</small></div></div><nav>{links.map(([to, label]) => <NavLink key={to} to={to} end={to === '/'}>{label}</NavLink>)}</nav><div className="local-note">LOCAL · 127.0.0.1</div></aside>
    <main><Routes>
      <Route path="/" element={<Overview />} />
      <Route path="/classic-view" element={<ClassicView />} />
      <Route path="/classic-view/:runId" element={<ClassicRun />} />
      <Route path="/classic-view/:runId/simulations/:simulationId" element={<ClassicDetail />} />
      <Route path="/runs" element={<Runs />} />
      <Route path="/runs/:runId" element={<RunDetail />} />
      <Route path="/runs/:runId/simulations/:simulationId" element={<RunTrajectory />} />
      <Route path="/synthesis-samples/:roundId/tasks/:taskId" element={<SynthesisSampleTask />} />
      <Route path="/synthesis-samples/:roundId/rows/:rowId" element={<SynthesisSampleTrajectory />} />
      <Route path="/synthetic-packages" element={<SyntheticPackages />} />
      <Route path="/synthetic-packages/:packageId" element={<SyntheticPackageTasks />} />
      <Route path="/synthetic-packages/:packageId/tasks/:taskId" element={<SyntheticPackageTask />} />
      <Route path="/bundles" element={<Bundles />} />
      <Route path="/bundles/:bundleId" element={<BundleDetail />} />
      <Route path="/bundles/:bundleId/tasks/:taskId" element={<TaskDetail />} />
      <Route path="/trajectories/:trajectoryId" element={<SyntheticTrajectory />} />
      <Route path="/documents" element={<Documents />} />
      <Route path="/documents/:documentId" element={<DocumentDetail />} />
      <Route path="/database" element={<Database />} />
      <Route path="/database/:table/:recordId" element={<RecordDetail />} />
      <Route path="/calibration" element={<Calibration />} />
    </Routes></main>
  </div>
}

export function parentPagePath(pathname) {
  const parts = pathname.split('/').filter(Boolean)
  if (parts[0] === 'classic-view' && parts.length === 2) return '/classic-view'
  if (parts[0] === 'runs' && parts.length === 2) return '/runs'
  if (parts[0] === 'synthetic-packages' && parts.length === 4 && parts[2] === 'tasks') return `/synthetic-packages/${parts[1]}`
  if (parts[0] === 'synthetic-packages' && parts.length === 2) return '/synthetic-packages'
  if (parts[0] === 'bundles' && parts.length === 4 && parts[2] === 'tasks') return `/bundles/${parts[1]}`
  if (parts[0] === 'bundles' && parts.length === 2) return '/bundles'
  if (parts[0] === 'documents' && parts.length === 2) return '/documents'
  if (parts[0] === 'database' && parts.length === 3) return '/database'
  return null
}

function Page({ title, subtitle, actions, children }) { const { pathname } = useLocation(); const parent = parentPagePath(pathname); return <><header className="page-header"><div><h1>{title}</h1>{subtitle && <p>{subtitle}</p>}</div><div className="actions">{parent && <BackButton to={parent} />}{actions}</div></header>{children}</> }
function Loading() { return <div className="empty">正在读取数据…</div> }
function ErrorBox({ error }) { return error ? <div className="error">{error}</div> : null }
function Badge({ children, tone = '' }) { return <span className={`badge ${tone}`}>{children ?? '—'}</span> }
function Json({ value }) { return <pre className="json">{JSON.stringify(value, null, 2)}</pre> }
function Search({ value, onChange, placeholder = '搜索…', list }) { return <input className="search" list={list} value={value} onChange={(event) => onChange(event.target.value)} placeholder={placeholder} /> }
function BackButton({ to, label = '返回上一级' }) { return <Link className="button-link back-button" to={to}>← {label}</Link> }
function Pager({ data, page, onChange }) { if (!data || data.total <= data.page_size) return null; const pages = Math.ceil(data.total / data.page_size); return <div className="pager"><button disabled={page <= 1} onClick={() => onChange(page - 1)}>上一页</button><span>第 {page} / {pages} 页 · 共 {data.total} 条</span><button disabled={page >= pages} onClick={() => onChange(page + 1)}>下一页</button></div> }
function statusTone(status) { return status === 'published' || status === 'succeeded' ? 'good' : status === 'failed' || status === 'cancelled' ? 'bad' : status === 'draft' || status === 'running' ? 'warn' : '' }

function Overview() {
  const { data, error, loading, reload } = useLoad(() => api('/overview'), [], 5000)
  if (loading) return <Loading />
  const cards = data ? [['运行结果', data.runs], ['合成任务包', data.synthetic_packages], ['临时合成轮次', data.synthesis_rounds], ['合成任务', data.tasks], ['全部轨迹', data.trajectories], ['成功轨迹', data.successful_trajectories], ['知识文档', data.documents], ['DB 记录', data.database_records], ['过期 Bundle', data.stale_bundles]] : []
  return <Page title="工作台总览" subtitle="当前本地数据与生成状态" actions={<button onClick={reload}>立即刷新</button>}><ErrorBox error={error} />
    {data?.index?.status === 'indexing' && <div className="notice">后台正在更新索引，页面可以继续使用；统计数据会在索引完成后自动刷新。当前已载入 {data.index.runs} 个运行结果、{data.index.bundles} 个 Bundle。</div>}
    {data?.index?.status === 'failed' && <div className="error">索引更新失败：{data.index.error}</div>}
    <section className="metric-grid">{cards.map(([label, value]) => <article className="metric" key={label}><span>{label}</span><strong>{value ?? '—'}</strong></article>)}</section>
    <section className="panel"><h2>失败类型与数据新鲜度</h2><div className="row">{data?.failure_types?.length ? data.failure_types.map((item) => <Badge tone="bad" key={item.type}>{item.type} · {item.count}</Badge>) : <span>暂无失败轨迹</span>}</div><p className="muted">索引状态：{data?.index?.status || '未知'} · 完成时间：{data?.index?.completed_at ? new Date(data.index.completed_at).toLocaleString() : '—'}</p></section>
    {data?.external_job && <section className="panel"><h2>外部合成作业</h2><div className="row"><Badge tone={data.external_job.process_alive ? 'warn' : ''}>{data.external_job.phase || '未知阶段'}</Badge><span>{data.external_job.bundle}</span><span>{data.external_job.accepted ?? 0} accepted</span><span>{data.external_job.process_alive ? '进程运行中（只读）' : '进程已结束'}</span></div></section>}
    <section className="panel"><h2>数据策略</h2><p>后台维护轻量索引，API 不等待全量目录扫描；完整轨迹与长文档只在打开时加载。轮询会等待上一次请求完成，原始轨迹保持不可变。</p></section>
  </Page>
}

function Runs() {
  const [searchParams, setSearchParams] = useSearchParams()
  const requestedView = searchParams.get('view')
  const view = ['synthetic', 'synthesis'].includes(requestedView) ? requestedView : 'regular'
  const switchView = (next) => setSearchParams(next === 'regular' ? {} : { view: next })
  return <Page title="运行轨迹" subtitle="分别浏览常规评测、合成任务轨迹与临时小规模样本">
    <div className="tabs" role="tablist" aria-label="轨迹来源">
      <button role="tab" aria-selected={view === 'regular'} className={view === 'regular' ? 'active' : ''} onClick={() => switchView('regular')}>常规评测</button>
      <button role="tab" aria-selected={view === 'synthetic'} className={view === 'synthetic' ? 'active' : ''} onClick={() => switchView('synthetic')}>合成任务轨迹</button>
      <button role="tab" aria-selected={view === 'synthesis'} className={view === 'synthesis' ? 'active' : ''} onClick={() => switchView('synthesis')}>临时合成样本</button>
    </div>
    {view === 'regular' ? <RegularRuns /> : view === 'synthetic' ? <SyntheticRuns /> : <SynthesisSamples />}
  </Page>
}

function RunSummary({ items }) {
  return <section className="run-summary">{items.map(([label, value, tone = '']) => <div key={label}><span>{label}</span><strong className={tone}>{value ?? '—'}</strong></div>)}</section>
}

function RegularRuns() {
  const [query, setQuery] = useState(''); const [page, setPage] = useState(1)
  const { data, error, loading } = useLoad(() => api(`/runs?q=${encodeURIComponent(query)}&page=${page}&page_size=100`), [query, page], 5000)
  return <section className="run-module"><div className="module-heading"><div><h2>data/simulations 测评结果</h2><p>按结果文件分组，进入后查看该次评测的全部 simulation。</p></div><Search value={query} onChange={(value) => { setQuery(value); setPage(1) }} placeholder="搜索运行名称或领域…" /></div><ErrorBox error={error} />{loading ? <Loading /> : <><RunSummary items={[["运行结果", data?.summary?.runs], ["轨迹", data?.summary?.trajectories], ["成功", data?.summary?.successful, 'good-text']]} /><div className="cards">{data?.items.map((run) => <Link className="card" to={`/runs/${run.id}`} key={run.id}><div><h3>{run.name}</h3><small>{run.file} · {run.domain || 'unknown'}</small></div><div className="card-stats"><Badge tone="good">{run.success_count} 成功</Badge><Badge>{run.simulation_count} 轨迹</Badge>{run.error && <Badge tone="bad">读取失败</Badge>}</div></Link>)}</div><Pager data={data} page={page} onChange={setPage} /></>}</section>
}

function SyntheticRuns() {
  const [query, setQuery] = useState(''); const [status, setStatus] = useState(''); const [bundleId, setBundleId] = useState(''); const [packagePrefix, setPackagePrefix] = useState(''); const [taskPrefix, setTaskPrefix] = useState(''); const [datasetVariant, setDatasetVariant] = useState(''); const [bundleTask, setBundleTask] = useState(''); const [candidateTrial, setCandidateTrial] = useState(''); const [termination, setTermination] = useState(''); const [model, setModel] = useState(''); const [review, setReview] = useState(''); const [page, setPage] = useState(1)
  const params = new URLSearchParams({ q: query, status, bundle_id: bundleId, package_prefix: packagePrefix, task_prefix: taskPrefix, dataset_variant: datasetVariant, bundle_task: bundleTask, candidate_trial: candidateTrial, termination, model, review, page: String(page), page_size: '100' })
  const { data, error, loading } = useIndexedLoad(() => api(`/synthetic-trajectories?${params}`), [query, status, bundleId, packagePrefix, taskPrefix, datasetVariant, bundleTask, candidateTrial, termination, model, review, page])
  const updateFilter = (setter) => (value) => { setter(value); setPage(1) }
  const clearFilters = () => { setQuery(''); setStatus(''); setBundleId(''); setPackagePrefix(''); setTaskPrefix(''); setDatasetVariant(''); setBundleTask(''); setCandidateTrial(''); setTermination(''); setModel(''); setReview(''); setPage(1) }
  const options = data?.filter_options || {}
  const allOptions = [...new Set([...(options.bundle_task || []), ...(options.candidate_trial || []), ...(options.termination || []), ...(options.model || [])])]
  const visiblePackages = (data?.bundle_options || []).filter((item) => !packagePrefix || item.prefix === packagePrefix)
  const selectPackage = (id) => { setBundleId(id); setPage(1) }
  return <section className="run-module">
    <div className="module-heading"><div><h2>data/synthetic 合成轨迹与训练样本</h2><p>按任务包系列、具体任务包和 Task 前缀逐级定位；大型 targeted 包和训练 JSONL 在选择后按需建立索引。</p></div><div className="module-filters"><Search list="synthetic-all-options" value={query} onChange={updateFilter(setQuery)} placeholder="搜索轨迹、训练样本或任务包…" /><select aria-label="任务包系列" value={packagePrefix} onChange={(event) => { updateFilter(setPackagePrefix)(event.target.value); setBundleId(''); setTaskPrefix(''); setDatasetVariant('') }}><option value="">全部系列</option>{data?.package_groups?.map((group) => <option key={group.prefix} value={group.prefix}>{group.prefix} · {group.packages} 包</option>)}</select><select aria-label="具体任务包" value={bundleId} onChange={(event) => { updateFilter(setBundleId)(event.target.value); setTaskPrefix(''); setDatasetVariant('') }}><option value="">选择具体任务包</option>{visiblePackages.map((bundle) => <option key={bundle.id} value={bundle.id}>{bundle.name}{bundle.loaded ? '' : ' · 按需加载'}</option>)}</select><select aria-label="Task 前缀" value={taskPrefix} onChange={(event) => updateFilter(setTaskPrefix)(event.target.value)}><option value="">全部 Task 前缀</option>{data?.task_prefixes?.map((prefix) => <option key={prefix} value={prefix}>{prefix}</option>)}</select><select aria-label="训练数据视图" value={datasetVariant} onChange={(event) => updateFilter(setDatasetVariant)(event.target.value)}><option value="">全部数据视图</option><option value="full">full</option><option value="focus">focus</option><option value="balanced">balanced</option></select><select aria-label="合成轨迹状态" value={status} onChange={(event) => updateFilter(setStatus)(event.target.value)}><option value="">全部状态</option><option value="training">训练样本</option><option value="success">成功</option><option value="failed">失败</option><option value="error">基础设施错误</option><option value="unknown">结果未知</option></select><button onClick={clearFilters}>清空筛选</button></div></div>
    <section className="prefix-groups panel"><strong>任务包系列</strong><div className="row">{data?.package_groups?.map((group) => <button className={packagePrefix === group.prefix ? 'active' : ''} key={group.prefix} onClick={() => { setPackagePrefix(group.prefix); setBundleId(''); setTaskPrefix(''); setPage(1) }}>{group.prefix} <small>{group.packages} 包 · {group.tasks} tasks · {group.trajectories} traces</small></button>)}</div></section>
    {query && !bundleId && data?.matching_packages?.length > 0 && <section className="panel"><h3>匹配的任务包</h3><p className="muted">选择具体包后加载其中的轨迹；不会一次扫描整个系列。</p><div className="cards compact">{data.matching_packages.slice(0, 20).map((item) => <button className="card" key={item.id} onClick={() => selectPackage(item.id)}><strong>{item.name}</strong><span><Badge>{item.prefix}</Badge> <Badge>{item.task_count} tasks</Badge> <Badge>{item.loaded ? `${item.trajectory_count} traces` : '点击加载'}</Badge></span></button>)}</div></section>}
    <datalist id="synthetic-all-options">{allOptions.map((value) => <option value={value} key={value} />)}</datalist>
    <datalist id="bundle-task-options">{options.bundle_task?.map((value) => <option value={value} key={value} />)}</datalist>
    <datalist id="candidate-trial-options">{options.candidate_trial?.map((value) => <option value={value} key={value} />)}</datalist>
    <datalist id="termination-options">{options.termination?.map((value) => <option value={value} key={value} />)}</datalist>
    <datalist id="model-options">{options.model?.map((value) => <option value={value} key={value} />)}</datalist>
    <ErrorBox error={error} />
    {loading ? <Loading /> : <><RunSummary items={[["轨迹/训练样本", data?.summary?.trajectories], ["训练样本", data?.summary?.training], ["成功", data?.summary?.successful, 'good-text'], ["失败", data?.summary?.failed, 'bad-text'], ["基础设施错误", data?.summary?.errors, 'bad-text']]} /><div className="classic-table"><table className="synthetic-runs"><thead><tr><th>任务包系列 / Bundle</th><th>Task 前缀 / Task</th><th>数据视图 / Trial</th><th>结果</th><th>终止原因</th><th>模型</th><th>Review</th></tr><tr className="column-filters"><th>{packagePrefix || '全部系列'}</th><th><input list="bundle-task-options" aria-label="按 Bundle 或 Task 筛选" value={bundleTask} onChange={(event) => updateFilter(setBundleTask)(event.target.value)} placeholder="输入或下拉选择" /></th><th><input list="candidate-trial-options" aria-label="按 Candidate 或 Trial 筛选" value={candidateTrial} onChange={(event) => updateFilter(setCandidateTrial)(event.target.value)} placeholder="输入或下拉选择" /></th><th><select aria-label="按结果筛选" value={status} onChange={(event) => updateFilter(setStatus)(event.target.value)}><option value="">全部</option><option value="training">训练样本</option><option value="success">成功</option><option value="failed">失败</option><option value="error">基础设施错误</option><option value="unknown">未知</option></select></th><th><input list="termination-options" aria-label="按终止原因筛选" value={termination} onChange={(event) => updateFilter(setTermination)(event.target.value)} placeholder="输入或下拉选择" /></th><th><input list="model-options" aria-label="按模型筛选" value={model} onChange={(event) => updateFilter(setModel)(event.target.value)} placeholder="输入或下拉选择" /></th><th><select aria-label="按 Review 筛选" value={review} onChange={(event) => updateFilter(setReview)(event.target.value)}><option value="">全部</option><option value="reviewed">有 Review</option><option value="unreviewed">无 Review</option></select></th></tr></thead><tbody>{data?.items?.length ? data.items.map((trajectory) => <tr key={trajectory.id}><td><Badge>{trajectory.package_prefix}</Badge><small className="block">{trajectory.bundle_name} · {trajectory.bundle_status || 'unknown'}</small></td><td><Link to={`/trajectories/${trajectory.id}?from=runs`}>{trajectory.task_id}</Link><small className="block">{trajectory.task_prefix}</small></td><td><code>{trajectory.dataset_variant || trajectory.candidate_id || '—'}</code><small className="block">{trajectory.source_kind ? `${trajectory.source_kind} · ` : ''}{trajectory.sample_line ? `line ${trajectory.sample_line}` : trajectory.trial || '—'}</small></td><td><Badge tone={trajectory.status === 'success' ? 'good' : trajectory.status === 'training' || trajectory.status === 'unknown' ? 'warn' : 'bad'}>{trajectory.status === 'training' ? '训练样本' : trajectory.reward ?? trajectory.status}</Badge></td><td>{trajectory.status === 'training' ? '不适用' : trajectory.termination_reason || '—'}</td><td>{trajectory.model || '—'}</td><td>{trajectory.status === 'training' ? '未包含' : trajectory.has_review ? <span title={trajectory.review_summary || ''}>有</span> : '—'}</td></tr>) : <tr><td className="empty-row" colSpan="7">{packagePrefix && !bundleId ? '请选择一个具体任务包以加载并展示轨迹或训练样本。' : '没有符合条件的合成轨迹或训练样本。'}</td></tr>}</tbody></table></div><Pager data={data} page={page} onChange={setPage} /></>}
  </section>
}

function SynthesisSamples() {
  const [query, setQuery] = useState(''); const [page, setPage] = useState(1); const [roundId, setRoundId] = useState('')
  const state = useLoad(() => api(`/synthesis-samples/rounds?q=${encodeURIComponent(query)}&page=${page}&page_size=100`), [query, page], 5000)
  const selected = state.data?.items?.find((item) => item.id === roundId)
  return <section className="run-module">
    <div className="module-heading"><div><h2>data/synthesis 临时小规模样本</h2><p>先选择合成轮次，再按需读取 pilot、small 和 validation task；请求日志与大型 JSONL 不参与首页扫描。</p></div><Search value={query} onChange={(value) => { setQuery(value); setPage(1); setRoundId('') }} placeholder="搜索轮次名称或状态…" /></div>
    <ErrorBox error={state.error} />
    {state.loading ? <Loading /> : <><RunSummary items={[["临时轮次", state.data?.summary?.rounds], ["活动轮次", state.data?.summary?.active], ["包含 Task", state.data?.summary?.with_tasks]]} /><div className="cards synthesis-rounds">{state.data?.items?.length ? state.data.items.map((round) => <button className={`card ${round.id === roundId ? 'selected' : ''}`} key={round.id} onClick={() => setRoundId(round.id)}><div><h3>{round.name}</h3><small>{round.parent ? `parent · ${round.parent}` : round.relative_path}</small></div><div className="card-stats">{round.active && <Badge tone="warn">活动轮次</Badge>}<Badge tone={String(round.status).toLowerCase() === 'complete' ? 'good' : String(round.status).toLowerCase() === 'incomplete' ? 'bad' : ''}>{round.status}</Badge>{round.stage && <Badge>{round.stage}</Badge>}</div><div className="row">{Object.entries(round.reports || {}).map(([stage, report]) => <Badge key={stage}>{stage} · {report.valid_tasks ?? 0}/{report.expected_tasks ?? '?'} tasks · {report.qualified_rows ?? 0} rows</Badge>)}</div></button>) : <div className="empty panel">没有找到临时合成轮次。</div>}</div><Pager data={state.data} page={page} onChange={setPage} /></>}
    {roundId && <SynthesisRoundTasks roundId={roundId} roundName={selected?.name || roundId} />}
  </section>
}

function SynthesisRoundTasks({ roundId, roundName }) {
  const [query, setQuery] = useState(''); const [stage, setStage] = useState(''); const [family, setFamily] = useState(''); const [page, setPage] = useState(1)
  useEffect(() => { setQuery(''); setStage(''); setFamily(''); setPage(1) }, [roundId])
  const state = useLoad(() => api(`/synthesis-samples/rounds/${roundId}/tasks?q=${encodeURIComponent(query)}&stage=${encodeURIComponent(stage)}&family=${encodeURIComponent(family)}&page=${page}&page_size=100`), [roundId, query, stage, family, page])
  return <section className="panel synthesis-task-list"><div className="section-heading"><div><h3>{roundName} · Task 样本</h3><small>Task 定义与训练轨迹分别按需读取</small></div><div className="module-filters"><Search value={query} onChange={(value) => { setQuery(value); setPage(1) }} placeholder="搜索 Task、用途或 family…" /><select aria-label="临时样本阶段" value={stage} onChange={(event) => { setStage(event.target.value); setPage(1) }}><option value="">全部阶段</option>{state.data?.stages?.map((value) => <option key={value} value={value}>{value}</option>)}</select><select aria-label="临时样本 Family" value={family} onChange={(event) => { setFamily(event.target.value); setPage(1) }}><option value="">全部 Family</option>{state.data?.families?.map((value) => <option key={value} value={value}>{value}</option>)}</select></div></div><ErrorBox error={state.error} />{state.loading ? <Loading /> : <><RunSummary items={[["Task", state.data?.summary?.tasks], ["训练轨迹", state.data?.summary?.training_samples], ["损坏记录", state.data?.summary?.errors, state.data?.summary?.errors ? 'bad-text' : '']]} /><div className="classic-table"><table><thead><tr><th>Stage / Slot</th><th>Task</th><th>Family</th><th>Difficulty</th><th>目标操作</th><th>训练轨迹</th></tr></thead><tbody>{state.data?.items?.length ? state.data.items.map((item) => <tr key={item.id}><td><Badge>{item.stage}</Badge><small className="block">slot {item.slot}</small></td><td>{item.error ? <span className="bad-text">{item.id} · {item.error}</span> : <><Link to={`/synthesis-samples/${roundId}/tasks/${encodeURIComponent(item.id)}`}>{item.id}</Link><small className="block">{item.purpose || '—'}</small></>}</td><td>{item.family || '—'}</td><td>{item.difficulty || '—'}</td><td>{item.target_operations?.length || 0}</td><td><Badge tone={item.sample_count ? 'good' : 'warn'}>{item.sample_count || 0}</Badge></td></tr>) : <tr><td className="empty-row" colSpan="6">没有符合条件的临时样本。</td></tr>}</tbody></table></div><Pager data={state.data} page={page} onChange={setPage} /></>}</section>
}

function SynthesisSampleTask() {
  const { roundId, taskId } = useParams(); const state = useLoad(() => api(`/synthesis-samples/rounds/${roundId}/tasks/${encodeURIComponent(taskId)}`), [roundId, taskId])
  if (state.loading) return <Page title="临时合成 Task"><Loading /></Page>
  const data = state.data
  return <Page title={data?.task?.id || taskId} subtitle={`${data?.round?.name || roundId} · ${data?.stage || '—'} / slot ${data?.slot || '—'}`} actions={<BackButton to="/runs?view=synthesis" label="返回临时样本" />}><ErrorBox error={state.error} />{data && <div className="detail-grid"><div><section className="panel"><h2>完整 Task 定义</h2><Json value={data.task} /></section><section className="panel"><h2>Candidate 与 Checks</h2><Json value={data.candidate} /></section></div><aside className="right-column"><section className="panel"><h2>Provenance</h2><Json value={data.provenance} /></section><section className="panel"><h2>已导出训练轨迹</h2>{data.samples.length ? data.samples.map((sample) => <Link className="row-link" key={sample.row_id} to={`/synthesis-samples/${roundId}/rows/${sample.row_id}`}><span><strong>{sample.stage} · line {sample.line}</strong><small className="block">seed {sample.seed} · weight {sample.weight}</small></span><Badge>{sample.token_audit?.tokens ?? '—'} tokens</Badge></Link>) : <p className="muted">该 Task 尚无合格训练轨迹。</p>}</section></aside></div>}</Page>
}

function SynthesisSampleTrajectory() {
  const { roundId, rowId } = useParams(); const state = useLoad(() => api(`/synthesis-samples/rounds/${roundId}/rows/${rowId}`), [roundId, rowId]); const taskId = state.data?.simulation?.task_id
  return <TrajectoryView {...state} backTo={taskId ? `/synthesis-samples/${roundId}/tasks/${encodeURIComponent(taskId)}` : '/runs?view=synthesis'} />
}

function RunDetail() {
  const { runId } = useParams(); const [query, setQuery] = useState(''); const [failed, setFailed] = useState(false); const [page, setPage] = useState(1)
  const { data, error, loading } = useLoad(() => api(`/runs/${runId}/simulations?q=${encodeURIComponent(query)}&failed=${failed}&page=${page}&page_size=100`), [runId, query, failed, page], 5000)
  return <Page title="轨迹列表" subtitle={`Run ${runId}`} actions={<><Search value={query} onChange={(value) => { setQuery(value); setPage(1) }} /><label className="check"><input type="checkbox" checked={failed} onChange={(e) => { setFailed(e.target.checked); setPage(1) }} />仅失败</label></>}><ErrorBox error={error} />{loading ? <Loading /> : <><table><thead><tr><th>Task</th><th>Trial</th><th>Reward</th><th>终止</th><th>Review</th></tr></thead><tbody>{data?.items.map((sim) => <tr key={sim.id}><td><Link to={`/runs/${runId}/simulations/${sim.id}`}>{sim.task_id}</Link></td><td>{sim.trial}</td><td><Badge tone={sim.reward === 1 ? 'good' : 'bad'}>{sim.reward ?? '—'}</Badge></td><td>{sim.termination_reason}</td><td>{sim.has_review ? '有' : '—'}</td></tr>)}</tbody></table><Pager data={data} page={page} onChange={setPage} /></>}</Page>
}

function ResultMark({ value, label }) {
  if (value == null) return <span className="mark muted">{label || '—'}</span>
  return <span className={`mark ${value ? 'pass' : 'fail'}`} aria-label={value ? '通过' : '失败'}>{value ? '✓' : '✕'}</span>
}

function ActionScore({ value }) {
  if (!value) return <span className="muted">—</span>
  const tone = value.correct === value.count ? 'good' : value.correct ? 'warn' : 'bad'
  return <Badge tone={tone}>{value.correct}/{value.count}</Badge>
}

function RubricBreakdown({ basis, breakdown, note }) {
  if (!basis?.length) return <span className="muted" title={note || '未保存 reward basis'}>未校验</span>
  return <div className="rubric-chips">{basis.map((name) => { const value = breakdown?.[name]; return <Badge key={name} tone={value === 1 ? 'good' : value == null ? 'warn' : 'bad'}>{name} · {value ?? '?'}</Badge> })}</div>
}

function TerminationMark({ reason }) {
  const icons = { user_stop: '👤', agent_stop: '🤖', max_steps: '⏱', too_many_errors: '💥', agent_error: '💥', user_error: '💥' }
  return <span title={reason || '未知'}>{icons[reason] || '?'}</span>
}

function ErrorCount({ count, critical, reviewed }) {
  if (!reviewed) return <span className="muted">—</span>
  return <Badge tone={count === 0 ? 'good' : critical ? 'bad' : 'warn'}>{count}{critical && count ? '!' : ''}</Badge>
}

function ClassicView() {
  const { data, error, loading, reload } = useLoad(() => api('/classic-view/runs'), [], 5000)
  return <Page title="经典 View" subtitle="复刻 tau3 view 的结果文件选择、汇总表与逐条详情" actions={<button onClick={reload}>刷新文件</button>}><ErrorBox error={error} />{loading ? <Loading /> : <div className="cards">{data?.items.length ? data.items.map((run) => <Link className="card" to={`/classic-view/${run.id}`} key={run.id}><div><h3>{run.name}</h3><small>{run.relative_path}</small></div><div className="card-stats"><Badge tone={run.file === 'results_reviewed.json' ? 'good' : ''}>{run.file}</Badge><Badge>{run.simulation_count} simulations</Badge>{run.error && <Badge tone="bad">读取失败</Badge>}</div></Link>) : <div className="empty panel">没有找到 results.json 或 results_reviewed.json。</div>}</div>}</Page>
}

function ClassicRun() {
  const { runId } = useParams(); const [query, setQuery] = useState(''); const [mode, setMode] = useState('all'); const [page, setPage] = useState(1); const [showTasks, setShowTasks] = useState(false)
  const state = useLoad(() => api(`/classic-view/runs/${runId}/simulations?mode=${mode}&q=${encodeURIComponent(query)}&page=${page}&page_size=50`), [runId, query, mode, page], 5000)
  const data = state.data
  return <Page title="Simulations" subtitle={`经典 View · ${runId}`} actions={<><Search value={query} onChange={(value) => { setQuery(value); setPage(1) }} placeholder="搜索 Task 或 Simulation ID…" /><select aria-label="轨迹筛选" value={mode} onChange={(event) => { setMode(event.target.value); setPage(1) }}><option value="all">全部</option><option value="failed">仅失败轨迹</option><option value="all_failed">所有 Trial 均失败的 Task</option></select></>}><ErrorBox error={state.error} />{state.loading ? <Loading /> : <><div className="classic-table"><table><thead><tr><th>#</th><th>Task</th><th>Trial</th><th>Reward</th><th>Rubric 分项</th><th>DB</th><th>Read Acts</th><th>Write Acts</th><th>Auth</th><th>Stop</th><th title="无响应时段">🔇</th><th>🤖 Err</th><th>👤 Err</th><th>🤖 Tags</th><th>👤 Tags</th></tr></thead><tbody>{data?.items.map((sim) => <tr key={sim.id}><td className="muted">{sim.index}</td><td><Link to={`/classic-view/${runId}/simulations/${sim.id}`} title={sim.task_id}>{sim.task_id}</Link></td><td>{sim.trial ?? '—'}</td><td title={sim.reward == null ? '未评分' : String(sim.reward)}><ResultMark value={sim.reward == null ? null : Math.abs(sim.reward - 1) <= 1e-6} /></td><td><RubricBreakdown basis={sim.reward_basis} breakdown={sim.reward_breakdown} note={sim.verifier_note} /></td><td><ResultMark value={sim.db_match} /></td><td><ActionScore value={sim.read_actions} /></td><td><ActionScore value={sim.write_actions} /></td><td><ResultMark value={sim.auth_status == null || sim.auth_status === 'not_needed' ? null : sim.auth_status === 'succeeded'} label={sim.auth_status === 'not_needed' ? 'n/a' : undefined} /></td><td><TerminationMark reason={sim.termination_reason} /></td><td><ResultMark value={sim.had_unresponsive_period == null ? null : !sim.had_unresponsive_period} /></td><td><ErrorCount count={sim.agent_error_count} critical={sim.agent_critical} reviewed={sim.has_review} /></td><td><ErrorCount count={sim.user_error_count} critical={sim.user_critical} reviewed={sim.has_review} /></td><td className="tag-cell">{sim.agent_tags.length ? sim.agent_tags.map((tag) => <Badge key={tag} tone={sim.agent_critical ? 'bad' : 'warn'}>{tag}</Badge>) : '—'}</td><td className="tag-cell">{sim.user_tags.length ? sim.user_tags.map((tag) => <Badge key={tag} tone={sim.user_critical ? 'bad' : 'warn'}>{tag}</Badge>) : '—'}</td></tr>)}</tbody></table></div>{mode === 'all_failed' && <p className="notice">全部 Trial 均失败的 Task：{data?.failed_task_count}</p>}<Pager data={data} page={page} onChange={setPage} /><details className="panel task-definitions" onToggle={(event) => setShowTasks(event.currentTarget.open)}><summary>查看结果文件内的任务定义</summary>{showTasks && <ClassicTasks runId={runId} />}</details></>}</Page>
}

function ClassicTasks({ runId }) {
  const { data, error, loading } = useLoad(() => api(`/classic-view/runs/${runId}/tasks`), [runId])
  if (loading) return <Loading />
  return <><ErrorBox error={error} />{data?.items.map((item) => <details key={item.id}><summary><code>{item.id}</code> · {item.description?.purpose || '无任务说明'}</summary><Json value={item} /></details>)}</>
}

function ClassicMessage({ message, index, fullToolResults, flagged }) {
  const isTool = message.role === 'tool'; const content = message.content == null ? '' : String(message.content); const truncated = isTool && !fullToolResults && content.length > 500
  return <tr id={`turn-${message.turn_idx ?? index}`} className={flagged ? 'flagged' : ''}><td><strong>{message.role}</strong>{message.requestor && <small className="block">requestor: {message.requestor}</small>}</td><td><pre>{truncated ? `${content.slice(0, 500)} […已截断 ${content.length - 500} 字符]` : content}</pre>{message.error && <Badge tone="bad">Tool error</Badge>}</td><td>{message.tool_calls?.map((call) => <div className="classic-tool" key={call.id || call.name}><strong>Tool: {call.name}</strong><Json value={call.arguments} /></div>)}{isTool && message.id && <code>Tool ID: {message.id}</code>}</td><td>{message.turn_idx ?? index}</td></tr>
}

function ScrollTopButton() {
  const [visible, setVisible] = useState(false)
  useEffect(() => {
    const update = () => setVisible(window.scrollY > 500)
    update(); window.addEventListener('scroll', update, { passive: true })
    return () => window.removeEventListener('scroll', update)
  }, [])
  return visible ? <button className="scroll-top" aria-label="回到页面顶部" onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}>↑ 回到顶部</button> : null
}

function ClassicDetail() {
  const { runId, simulationId } = useParams(); const [fullToolResults, setFullToolResults] = useState(false); const [liveReview, setLiveReview] = useState(null); const state = useLoad(() => api(`/runs/${runId}/simulations/${simulationId}`), [runId, simulationId]); const sim = state.data?.simulation
  useEffect(() => { setLiveReview(null) }, [simulationId])
  const currentDiagnosis = liveReview?.diagnosis
  const reviewTurns = new Set([...(sim?.review?.errors || []), ...(sim?.user_only_review?.errors || []), ...(currentDiagnosis?.errors || [])].map((item) => item.turn_idx).filter((turn) => turn != null))
  const flaggedTurns = new Set([...reviewTurns, ...(state.data?.verifier_diagnostics?.failed_turns || [])])
  if (state.loading) return <Page title={`Simulation · ${simulationId}`} actions={<BackButton to={`/classic-view/${runId}`} />}><Loading /></Page>
  return <Page title={`Simulation · ${sim?.task_id || simulationId}`} subtitle={`Trial ${sim?.trial ?? '—'} · ${simulationId}`} actions={<><Link className="button-link" to={`/classic-view/${runId}`}>返回列表</Link><label className="check"><input type="checkbox" checked={fullToolResults} onChange={(event) => setFullToolResults(event.target.checked)} />完整 Tool Results</label></>}><ErrorBox error={state.error} />{state.data && <><section className="panel"><h2>Task Details</h2><Json value={state.data.task} /></section><section className="panel"><h2>Simulation Overview</h2><div className="classic-overview"><span>Simulation ID</span><code>{sim.id}</code><span>Task ID</span><strong>{sim.task_id}</strong><span>Trial</span><strong>{sim.trial ?? '—'}</strong><span>Start / End</span><strong>{sim.start_time || '—'} / {sim.end_time || '—'}</strong><span>Duration</span><strong>{sim.duration == null ? '—' : `${sim.duration.toFixed(2)}s`}</strong><span>Mode</span><strong>{sim.mode || '—'}</strong><span>Termination</span><strong><TerminationMark reason={sim.termination_reason} /> {sim.termination_reason}</strong><span>Agent / User cost</span><strong>{sim.agent_cost ?? '—'} / {sim.user_cost ?? '—'}</strong><span>Reward</span><Badge tone={Math.abs((sim.reward_info?.reward ?? 0) - 1) <= 1e-6 ? 'good' : 'bad'}>{sim.reward_info?.reward ?? '—'}</Badge></div><details><summary>Reward checks 与附加信息</summary><Json value={{ reward_info: sim.reward_info, auth_classification: sim.auth_classification, info: sim.info }} /></details></section><VerifierRubric task={state.data.task} rewardInfo={sim.reward_info} diagnostics={state.data.verifier_diagnostics} /><section className="panel"><div className="section-heading"><h2>Messages</h2><small>Rubric 参数错误、额外调用和 Review 对应轮次会高亮；点击 Rubric 中的 turn 可直接跳转。</small></div><div className="classic-table"><table className="classic-messages"><thead><tr><th>Role</th><th>Content</th><th>Details</th><th>Turn</th></tr></thead><tbody>{(sim.messages || []).map((message, index) => <ClassicMessage key={`${message.turn_idx}-${index}`} message={message} index={index} fullToolResults={fullToolResults} flagged={flaggedTurns.has(message.turn_idx)} />)}</tbody></table></div></section><section className="panel"><h2>错误归因诊断</h2>{sim.review && <><h3>原始 LLM Conversation Review</h3><Json value={sim.review} /></>}{sim.user_only_review && <><h3>原始 User-only Review</h3><Json value={sim.user_only_review} /></>}<FixedTrajectoryDiagnosis key={state.data.artifact.id} artifact={state.data.artifact} onResult={setLiveReview} /></section></>}<ScrollTopButton /></Page>
}

function packageFormatLabel(format) {
  return { tau3_aa: 'τ3-AA Bundle', world_package: 'World Package', task_batch: 'Task Batch', streaming_batch: 'Streaming Batch', targeted_round: 'Targeted Round', training_dataset: 'Training Dataset' }[format] || format
}

function SyntheticPackages() {
  const [query, setQuery] = useState(''); const [status, setStatus] = useState(''); const [format, setFormat] = useState(''); const [page, setPage] = useState(1)
  const state = useIndexedLoad(() => api(`/synthetic-packages?q=${encodeURIComponent(query)}&status=${encodeURIComponent(status)}&format=${encodeURIComponent(format)}&page=${page}&page_size=100`), [query, status, format, page])
  const data = state.data
  return <Page title="合成任务包" subtitle="统一展示 data/synthetic 中不同格式的任务包" actions={<><Search value={query} onChange={(value) => { setQuery(value); setPage(1) }} placeholder="搜索名称、路径、格式或状态…" /><select aria-label="任务包格式" value={format} onChange={(event) => { setFormat(event.target.value); setPage(1) }}><option value="">全部格式</option><option value="tau3_aa">τ3-AA Bundle</option><option value="world_package">World Package</option><option value="task_batch">Task Batch</option><option value="streaming_batch">Streaming Batch</option><option value="targeted_round">Targeted Round</option><option value="training_dataset">Training Dataset</option></select><select aria-label="任务包状态" value={status} onChange={(event) => { setStatus(event.target.value); setPage(1) }}><option value="">全部状态</option><option value="PASS">PASS</option><option value="published">published</option><option value="draft">draft</option><option value="running">running</option><option value="INCOMPLETE">INCOMPLETE</option></select></>}><ErrorBox error={state.error} />{state.loading ? <Loading /> : <><RunSummary items={[["任务包", data?.summary?.packages], ["子任务", data?.summary?.tasks], ["轨迹/训练样本", data?.summary?.trajectories]]} /><div className="cards">{data?.items?.length ? data.items.map((item) => <Link className="card" to={`/synthetic-packages/${item.id}`} key={item.id}><div><h3>{item.name}</h3><small>{item.relative_path}</small></div><div className="card-stats"><Badge>{packageFormatLabel(item.format)}</Badge><Badge tone={statusTone(item.status)}>{item.status}</Badge>{item.validation?.status && <Badge tone={String(item.validation.status).toLowerCase().includes('pass') ? 'good' : 'warn'}>验证 · {item.validation.status}</Badge>}<Badge>{item.task_count} tasks</Badge><Badge>{item.trajectory_count} {item.format === 'training_dataset' ? 'samples' : 'traces'}</Badge>{item.training_summary && <Badge>full {item.training_summary.full} · focus {item.training_summary.focus} · balanced {item.training_summary.balanced}</Badge>}{item.error && <Badge tone="bad">读取失败</Badge>}</div></Link>) : <div className="empty panel">没有符合条件的合成任务包。</div>}</div><Pager data={data} page={page} onChange={setPage} /></>}</Page>
}

function SyntheticPackageTasks() {
  const { packageId } = useParams(); const [query, setQuery] = useState(''); const [page, setPage] = useState(1)
  const state = useIndexedLoad(() => api(`/synthetic-packages/${packageId}/tasks?q=${encodeURIComponent(query)}&page=${page}&page_size=100`), [packageId, query, page])
  const data = state.data
  return <Page title={data?.package?.name || '任务包子任务'} subtitle={data?.package ? `${packageFormatLabel(data.package.format)} · ${data.package.relative_path}` : packageId} actions={<Search value={query} onChange={(value) => { setQuery(value); setPage(1) }} placeholder="搜索 Task ID、用途、阶段或 Slot…" />}><ErrorBox error={state.error} />{state.loading ? <Loading /> : <><section className="package-summary panel"><div className="row"><Badge>{packageFormatLabel(data?.package?.format)}</Badge><Badge tone={statusTone(data?.package?.status)}>{data?.package?.status}</Badge>{data?.package?.validation?.status && <Badge>验证 · {data.package.validation.status}</Badge>}</div>{data?.package?.training_summary && <div className="rubric-summary"><Badge>独立 Task {data.package.training_summary.unique_tasks}</Badge><Badge>full {data.package.training_summary.full}</Badge><Badge>focus {data.package.training_summary.focus}</Badge><Badge>balanced {data.package.training_summary.balanced}</Badge></div>}<small>{data?.package?.format === 'training_dataset' ? '训练交付按 Task 聚合；full、focus、balanced 是数据视图，SFT 与 general-agent 是同一逻辑样本的两种序列化。' : '任务使用 package_id + task_id 命名空间，避免不同任务包中的重复 ID 冲突。'}</small></section><table><thead><tr><th>Task ID</th><th>Family / Purpose</th><th>来源</th><th>阶段</th><th>Slot</th><th>Candidate</th><th>状态</th><th>Split</th><th>轨迹/样本</th><th>文档</th></tr></thead><tbody>{data?.items?.length ? data.items.map((task) => <tr key={task.id}><td><Link to={`/synthetic-packages/${packageId}/tasks/${encodeURIComponent(task.id)}`}>{task.id}</Link><small className="block">{task.task_prefix}</small></td><td>{task.family || task.purpose || '—'}</td><td>{task.source || '—'}</td><td>{task.stage || '—'}</td><td>{task.slot_id || '—'}</td><td>{task.candidate_id || '—'}</td><td><Badge tone={task.accepted === false ? 'bad' : task.accepted ? 'good' : ''}>{task.accepted == null ? task.status : task.accepted ? 'accepted' : 'rejected'}</Badge></td><td>{task.splits?.join(', ') || '—'}</td><td>{task.trajectory_count}{task.sample_counts && <small className="block">full {task.sample_counts.full || 0} · focus {task.sample_counts.focus || 0} · balanced {task.sample_counts.balanced || 0}</small>}</td><td>{task.required_document_count}</td></tr>) : <tr><td className="empty-row" colSpan="10">这个任务包当前没有可展示的子任务。</td></tr>}</tbody></table><Pager data={data} page={page} onChange={setPage} /></>}</Page>
}

function SyntheticPackageTask() {
  const { packageId, taskId } = useParams()
  const state = useLoad(() => api(`/synthetic-packages/${packageId}/tasks/${encodeURIComponent(taskId)}`), [packageId, taskId])
  const data = state.data
  if (state.loading) return <Page title={taskId}><Loading /></Page>
  const training = data?.training_dataset
  return <Page title={taskId} subtitle={data?.package ? `${data.package.name} · ${packageFormatLabel(data.package.format)}` : packageId}><ErrorBox error={state.error} />{data && <div className="detail-grid"><div>{training ? <section className="panel"><h2>训练数据说明</h2><p className="notice">{training.note}</p><div className="rubric-summary"><Badge>full {training.sample_counts.full || 0}</Badge><Badge>focus {training.sample_counts.focus || 0}</Badge><Badge>balanced {training.sample_counts.balanced || 0}</Badge></div><p className="muted">Rubric 保留在任务定义中，但这些导出样本没有本次执行的 reward、action check 或模型 review，不能据此判定通过或失败。</p></section> : <VerifierRubric task={data.task} rewardInfo={null} />}<section className="panel"><h2>完整任务定义</h2><Json value={data.task} /></section></div><aside className="right-column"><section className="panel"><h2>任务元数据</h2><div className="kv"><span>来源</span><strong>{data.source || '—'}</strong><span>Split</span><strong>{data.splits?.join(', ') || '—'}</strong>{data.targeted_round && <><span>阶段 / Slot</span><strong>{data.targeted_round.stage} / {data.targeted_round.slot_id}</strong><span>Candidate</span><strong>{data.targeted_round.candidate_id}</strong></>}{training && <><span>Shard Slot</span><strong>{training.slot_ids.join(', ') || '—'}</strong></>}<span>状态</span><strong>{data.package.status}</strong></div>{data.package.editable && data.package.source_bundle_id && <Link className="button-link" to={`/bundles/${data.package.source_bundle_id}/tasks/${encodeURIComponent(taskId)}`}>进入兼容 Bundle 校准页</Link>}</section><section className="panel"><h2>验证证据</h2><Json value={data.validation} /></section><section className="panel"><h2>{training ? '训练样本' : '关联轨迹'}</h2>{data.trajectories.length ? data.trajectories.map((trajectory) => <Link className="row-link" key={trajectory.id} to={`/trajectories/${trajectory.id}?package_id=${packageId}&task_id=${encodeURIComponent(taskId)}`}><span>{training ? `${trajectory.dataset_variant} · ${trajectory.source_kind} · line ${trajectory.trial}` : trajectory.trial}</span>{training ? <Badge tone="warn">训练样本</Badge> : <Badge tone={trajectory.reward === 1 ? 'good' : 'bad'}>{trajectory.reward ?? '—'}</Badge>}</Link>) : <p>没有显式合成轨迹或训练样本。</p>}{data.run_simulations.map((simulation) => <Link className="row-link" key={`${simulation.run_id}-${simulation.id}`} to={`/runs/${simulation.run_id}/simulations/${simulation.id}`}><span>Simulation · Trial {simulation.trial ?? '—'}</span><span><Badge tone={simulation.reward === 1 ? 'good' : 'bad'}>{simulation.reward ?? '—'}</Badge> <Badge>{simulation.association === 'explicit' ? '显式关联' : '唯一 Task ID 推断'}</Badge></span></Link>)}</section><section className="panel"><h2>Required Documents</h2>{data.task.required_documents?.length ? data.task.required_documents.map((id) => <Link className="row-link" key={id} to={`/documents/${id}`}>{id}</Link>) : <p>无</p>}</section></aside></div>}</Page>
}

function Bundles() {
  const [query, setQuery] = useState(''); const [status, setStatus] = useState(''); const [page, setPage] = useState(1); const { data, error, loading } = useLoad(() => api(`/bundles?q=${encodeURIComponent(query)}&status=${status}&page=${page}&page_size=100`), [query, status, page], 5000)
  return <Page title="合成 Bundle" subtitle="历史、草稿与已发布 bundle 均按独立命名空间展示" actions={<><Search value={query} onChange={(value) => { setQuery(value); setPage(1) }} /><select value={status} onChange={(event) => { setStatus(event.target.value); setPage(1) }}><option value="">全部状态</option><option value="draft">draft</option><option value="published">published</option></select></>}><ErrorBox error={error} />{loading ? <Loading /> : <><div className="cards">{data?.items.map((bundle) => <Link className="card" to={`/bundles/${bundle.id}`} key={bundle.id}><div><h3>{bundle.name}</h3><small>{bundle.relative_path}</small></div><div className="card-stats"><Badge tone={statusTone(bundle.status)}>{bundle.status}</Badge>{bundle.stale && <Badge tone="bad">环境已变化</Badge>}<Badge>{bundle.task_count} tasks</Badge><Badge>{bundle.trajectory_count} traces</Badge></div></Link>)}</div><Pager data={data} page={page} onChange={setPage} /></>}</Page>
}

function BundleDetail() {
  const { bundleId } = useParams(); const [query, setQuery] = useState(''); const [accepted, setAccepted] = useState(''); const [page, setPage] = useState(1)
  const suffix = accepted === '' ? '' : `&accepted=${accepted}`
  const { data, error, loading } = useLoad(() => api(`/bundles/${bundleId}/tasks?q=${encodeURIComponent(query)}&page=${page}&page_size=100${suffix}`), [bundleId, query, accepted, page], 5000)
  return <Page title="合成任务" subtitle={`Bundle ${bundleId}`} actions={<><Search value={query} onChange={(value) => { setQuery(value); setPage(1) }} /><select value={accepted} onChange={(e) => { setAccepted(e.target.value); setPage(1) }}><option value="">全部状态</option><option value="true">已接纳</option><option value="false">未接纳</option></select></>}><ErrorBox error={error} />{loading ? <Loading /> : <><table><thead><tr><th>ID</th><th>Family</th><th>状态</th><th>Split</th><th>轨迹</th><th>文档</th></tr></thead><tbody>{data?.items.map((task) => <tr key={task.id}><td><Link to={`/bundles/${bundleId}/tasks/${task.id}`}>{task.id}</Link><small className="block">{task.purpose}</small></td><td>{task.family || '—'}</td><td><Badge tone={task.accepted ? 'good' : 'warn'}>{task.accepted == null ? task.source : task.accepted ? 'accepted' : 'pending'}</Badge></td><td>{task.splits.join(', ') || '—'}</td><td>{task.trajectory_count}</td><td>{task.required_documents.length}</td></tr>)}</tbody></table><Pager data={data} page={page} onChange={setPage} /></>}</Page>
}

function Editor({ artifact, initial, baseHash, onCreated }) {
  const [text, setText] = useState(JSON.stringify(initial, null, 2)); const [draft, setDraft] = useState(initial); const [mode, setMode] = useState('form'); const [rationale, setRationale] = useState(''); const [message, setMessage] = useState('')
  useEffect(() => { setText(JSON.stringify(initial, null, 2)); setDraft(initial) }, [initial])
  function switchMode(next) { if (next === 'form') { try { setDraft(JSON.parse(text)); setMessage('') } catch { setMessage('JSON 格式不正确，无法切换结构化编辑'); return } } else { setText(JSON.stringify(draft, null, 2)) } setMode(next) }
  function setField(key, value) { setDraft({ ...draft, [key]: value }) }
  async function save() { try { const value = mode === 'raw' ? JSON.parse(text) : draft; const result = await api('/change-sets', { method: 'POST', body: JSON.stringify({ rationale, targets: [{ artifact, base_sha256: baseHash, operations: [{ op: 'replace', path: '', value }] }] }) }); const validated = await api(`/change-sets/${result.id}/validate`, { method: 'POST' }); setMessage(`已创建并校验 ${validated.id}，请到“校准与作业”审核应用。`); onCreated?.() } catch (error) { setMessage(error.message) } }
  return <section className="panel editor"><div className="editor-head"><h2>人工校准</h2><div><button className={mode === 'form' ? 'selected' : ''} onClick={() => switchMode('form')}>结构化</button><button className={mode === 'raw' ? 'selected' : ''} onClick={() => switchMode('raw')}>原始 JSON</button></div></div>{mode === 'raw' ? <textarea value={text} onChange={(e) => setText(e.target.value)} spellCheck="false" /> : <div className="fields">{Object.entries(draft || {}).map(([key, value]) => <label key={key}><span>{key}{key === 'id' && '（不可修改）'}</span>{typeof value === 'object' && value !== null ? <textarea value={JSON.stringify(value, null, 2)} readOnly title="复杂字段请切换到原始 JSON 编辑" /> : key === 'content' ? <textarea value={value ?? ''} onChange={(event) => setField(key, event.target.value)} /> : <input value={value ?? ''} disabled={key === 'id'} onChange={(event) => setField(key, typeof value === 'number' ? Number(event.target.value) : typeof value === 'boolean' ? event.target.value === 'true' : event.target.value)} />}</label>)}</div>}<input value={rationale} onChange={(e) => setRationale(e.target.value)} placeholder="修改理由（必填）" /><button disabled={!rationale.trim()} onClick={save}>保存为变更集并校验</button>{message && <p className="notice">{message}</p>}</section>
}

function TaskDetail() {
  const { bundleId, taskId } = useParams(); const { data, error, loading } = useLoad(() => api(`/bundles/${bundleId}/tasks/${taskId}`), [bundleId, taskId])
  if (loading) return <Page title={taskId}><Loading /></Page>
  return <Page title={taskId} subtitle={`${data?.bundle.name} · ${data?.bundle.status}`}><ErrorBox error={error} />{data && <div className="detail-grid"><div><section className="panel"><h2>任务定义</h2><Json value={data.task} /></section><Editor artifact={{ kind: 'task', id: taskId, bundle_id: bundleId, task_id: taskId }} initial={data.task} baseHash={data.sha256} /></div><aside className="right-column"><section className="panel"><h2>关联轨迹</h2>{data.trajectories.length ? data.trajectories.map((t) => <Link className="row-link" key={t.id} to={`/trajectories/${t.id}`}>{t.trial}<Badge tone={t.summary.reward === 1 ? 'good' : 'bad'}>{t.summary.reward ?? '—'}</Badge></Link>) : <p>尚未执行</p>}</section><section className="panel"><h2>Required documents</h2>{(data.task.required_documents || []).map((id) => <Link className="row-link" key={id} to={`/documents/${id}`}>{id}</Link>)}</section><section className="panel"><h2>相关 DB 记录</h2>{data.db_references?.length ? data.db_references.map((ref) => <Link className="row-link" key={`${ref.table}-${ref.id}`} to={`/database/${ref.table}/${ref.id}`}><span>{ref.id}</span><Badge>{ref.table}</Badge></Link>) : <p>未发现显式记录引用</p>}</section><section className="panel"><h2>Candidate checks</h2><Json value={data.candidate ? { accepted: data.candidate.accepted, static_passed: data.candidate.static_passed, text_checked: data.candidate.text_checked, checks: data.candidate.checks, errors: data.candidate.errors } : null} /></section></aside></div>}</Page>
}

function RunTrajectory() { const { runId, simulationId } = useParams(); const state = useLoad(() => api(`/runs/${runId}/simulations/${simulationId}`), [runId, simulationId]); return <TrajectoryView {...state} backTo={`/runs/${runId}`} /> }
function SyntheticTrajectory() { const { trajectoryId } = useParams(); const location = useLocation(); const state = useLoad(() => api(`/trajectories/${trajectoryId}`), [trajectoryId]); const artifact = state.data?.artifact; const params = new URLSearchParams(location.search); const fromRuns = params.get('from') === 'runs'; const packageId = params.get('package_id'); const packageTaskId = params.get('task_id'); const backTo = packageId && packageTaskId ? `/synthetic-packages/${packageId}/tasks/${encodeURIComponent(packageTaskId)}` : fromRuns ? '/runs?view=synthetic' : artifact?.bundle_id && artifact?.task_id ? `/bundles/${artifact.bundle_id}/tasks/${artifact.task_id}` : '/synthetic-packages'; return <TrajectoryView {...state} backTo={backTo} /> }

function AnnotationForm({ artifact, onSaved }) {
  const [form, setForm] = useState({ source: 'agent', turn_idx: '', severity: 'critical', error_tags: '', reasoning: '', correct_behavior: '', verdict: 'additional' }); const [error, setError] = useState('')
  async function submit(event) { event.preventDefault(); try { await api('/annotations', { method: 'POST', body: JSON.stringify({ ...form, turn_idx: form.turn_idx === '' ? null : Number(form.turn_idx), error_tags: form.error_tags.split(',').map((x) => x.trim()).filter(Boolean), artifact }) }); setForm({ ...form, reasoning: '', correct_behavior: '' }); setError('已保存人工归因'); onSaved?.() } catch (e) { setError(e.message) } }
  return <form className="annotation" onSubmit={submit}><h3>新增人工归因</h3><div className="form-row"><select value={form.source} onChange={(e) => setForm({ ...form, source: e.target.value })}><option value="agent">Agent</option><option value="user">User</option><option value="system">System</option><option value="unknown">Unknown</option></select><input type="number" placeholder="Turn" value={form.turn_idx} onChange={(e) => setForm({ ...form, turn_idx: e.target.value })} /><select value={form.severity} onChange={(e) => setForm({ ...form, severity: e.target.value })}><option>critical</option><option>minor</option></select></div><input placeholder="标签，逗号分隔" value={form.error_tags} onChange={(e) => setForm({ ...form, error_tags: e.target.value })} /><textarea placeholder="归因说明" value={form.reasoning} onChange={(e) => setForm({ ...form, reasoning: e.target.value })} required /><textarea placeholder="正确行为" value={form.correct_behavior} onChange={(e) => setForm({ ...form, correct_behavior: e.target.value })} /><button>保存归因</button>{error && <p className="notice">{error}</p>}</form>
}

function FixedTrajectoryDiagnosis({ artifact, onResult }) {
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(true)

  const load = useCallback(async () => {
    try {
      setLoading(true); setError('')
      const payload = await api('/fixed-diagnoses/resolve', {
        method: 'POST', body: JSON.stringify(artifact),
      })
      setResult(payload)
      onResult(payload)
    } catch (reason) { setError(reason.message); setResult(null); onResult(null) }
    finally { setLoading(false) }
  }, [artifact, onResult])

  useEffect(() => {
    load()
  }, [load])

  const diagnosis = result?.diagnosis
  const errors = diagnosis?.errors || []
  return <div className="trajectory-diagnosis">
    <div className="diagnosis-actions"><button className="primary" disabled={loading} onClick={load}>{loading ? '正在加载固定诊断…' : '重新加载固定诊断'}</button>{result?.status === 'matched' && <Badge tone="good">已匹配固定文件</Badge>}</div>
    <p className="muted">只读取服务端约定目录中的固定 JSON 文件，不调用 LLM，也不会产生模型费用。</p>
    {error && <ErrorBox error={error} />}
    {result?.status === 'file_missing' && <div className="notice"><strong>尚无固定诊断文件。</strong><span>请创建：<code>{result.expected_file}</code></span></div>}
    {result?.status === 'artifact_missing' && <div className="notice"><strong>文件存在，但没有匹配当前轨迹。</strong><span>请在 <code>{result.expected_file}</code> 的 diagnoses 中添加 artifact_id=<code>{artifact.id}</code>，或唯一匹配 trial=<code>{String(result.trial)}</code>。</span></div>}
    {diagnosis && <div className="diagnosis-result"><h3>固定文件诊断</h3><p>{diagnosis.summary || (diagnosis.has_errors ? '固定文件记录了错误' : '固定文件明确记录为无错误')}</p><div className="row"><Badge tone={diagnosis.has_errors ? 'bad' : 'good'}>{diagnosis.has_errors ? `${errors.length} 个错误` : '固定文件明确标记：无错误'}</Badge><Badge>来源 · 固定文件</Badge></div>{errors.map((item, index) => <article className="diagnosis-error" key={`${item.turn_idx}-${index}`}><div className="row"><Badge tone={item.severity?.startsWith('critical') ? 'bad' : 'warn'}>{item.source || 'unknown'} · {item.severity || 'unknown'}</Badge>{item.turn_idx != null && <Badge>Turn {item.turn_idx}</Badge>}{item.error_tags?.map((tag) => <Badge key={tag}>{tag}</Badge>)}</div><p>{item.reasoning}</p>{item.correct_behavior && <p><strong>正确行为：</strong>{item.correct_behavior}</p>}</article>)}</div>}
    {result?.expected_file && <details><summary>固定文件位置与匹配信息</summary><Json value={{ expected_file: result.expected_file, package: result.package, task_id: result.task_id, trial: result.trial, file_updated_at: result.file_updated_at }} /></details>}
  </div>
}

function RubricStatus({ passed }) {
  if (passed == null) return <Badge tone="warn">未产生结果</Badge>
  return <Badge tone={passed ? 'good' : 'bad'}>{passed ? '通过' : '未通过'}</Badge>
}

function RubricSection({ title, basis, active, children }) {
  return <section className="rubric-section"><header><div><h3>{title}</h3><code>{basis}</code></div><Badge tone={active ? 'good' : ''}>{active ? '计入最终 Reward' : '仅诊断/参考'}</Badge></header>{children}</section>
}

function TurnLink({ turn }) {
  return turn == null ? <span>—</span> : <a className="turn-link" href={`#turn-${turn}`}>turn {turn}</a>
}

function ActionDiagnostics({ diagnostics, active }) {
  const summary = diagnostics?.summary || {}
  const labels = { matched: '完全匹配', argument_mismatch: '参数不匹配', missing: '缺失调用', not_evaluated: '未执行校验' }
  const tones = { matched: 'good', argument_mismatch: 'bad', missing: 'bad', not_evaluated: 'warn' }
  return <RubricSection title="工具调用与动作对照" basis="ACTION" active={active}>
    {diagnostics.premature_termination && <div className="notice"><strong>轨迹提前终止：{diagnostics.termination_reason}</strong><span>{diagnostics.reward_info_note || 'Action verifier 可能没有执行，不能把“未产生结果”解释为 action 已通过。'}</span></div>}
    <div className="rubric-summary"><Badge tone="good">匹配 {summary.matched || 0}</Badge><Badge tone={summary.argument_mismatch ? 'bad' : ''}>参数错误 {summary.argument_mismatch || 0}</Badge><Badge tone={summary.missing ? 'bad' : ''}>缺失 {summary.missing || 0}</Badge><Badge tone={summary.not_evaluated ? 'warn' : ''}>未校验 {summary.not_evaluated || 0}</Badge><Badge tone={summary.extra ? 'warn' : ''}>额外调用 {summary.extra || 0}</Badge></div>
    {diagnostics.actions.map((item) => <details className={`action-diagnostic ${item.status}`} open={item.status !== 'matched'} key={item.action_id || item.index}><summary><span>{item.index + 1}. {item.expected.requestor || 'assistant'} → {item.expected.name}</span><Badge tone={tones[item.status]}>{labels[item.status]}</Badge></summary><div className="action-compare"><div><h4>期望调用</h4><Json value={{ arguments: item.expected.arguments || {}, compare_args: item.expected.compare_args || '全部参数', info: item.expected.info }} /></div><div><h4>评估结果</h4><Json value={item.check || { status: '未保存 action check' }} /></div></div>{item.matched_call && <p><b>匹配位置：</b><TurnLink turn={item.matched_call.turn_idx} /></p>}{item.observed_candidates.length > 0 ? <div className="observed-calls"><h4>同名实际调用</h4>{item.observed_candidates.map((call) => <article key={call.key}><div className="row"><TurnLink turn={call.turn_idx} /><code>{call.requestor} → {call.name}</code>{call.argument_diffs.length ? <Badge tone="bad">{call.argument_diffs.length} 个字段不同</Badge> : <Badge tone="good">参数一致</Badge>}</div><Json value={call.arguments} />{call.argument_diffs.length > 0 && <table className="argument-diff"><thead><tr><th>字段</th><th>期望</th><th>实际</th></tr></thead><tbody>{call.argument_diffs.map((diff) => <tr key={diff.field}><td><code>{diff.field}</code></td><td><Json value={diff.expected_present === false ? '〈缺失〉' : diff.expected} /></td><td><Json value={diff.actual_present === false ? '〈缺失〉' : diff.actual} /></td></tr>)}</tbody></table>}</article>)}</div> : <p className="bad-text">轨迹中没有找到同 requestor、同工具名的调用。</p>}</details>)}
    {diagnostics.extra_calls.length > 0 && <details className="extra-actions" open><summary>额外或重复调用 · {diagnostics.extra_calls.length}</summary><p className="notice">当前 Action evaluator 只检查必需调用是否存在，额外调用通常不扣 ACTION reward；它们仍可能导致 DB、策略或安全检查失败，需要人工核对。</p><table><thead><tr><th>轮次</th><th>调用者</th><th>工具</th><th>参数</th></tr></thead><tbody>{diagnostics.extra_calls.slice(0, 100).map((call) => <tr key={call.key}><td><TurnLink turn={call.turn_idx} /></td><td>{call.requestor}</td><td><code>{call.name}</code></td><td><Json value={call.arguments} /></td></tr>)}</tbody></table>{diagnostics.extra_calls.length > 100 && <p>仅显示前 100 条。</p>}</details>}
  </RubricSection>
}

function VerifierRubric({ task, rewardInfo, diagnostics }) {
  const criteria = task?.evaluation_criteria || {}
  const reward = rewardInfo || {}
  const basis = reward.reward_basis || criteria.reward_basis || []
  const active = (name) => basis.includes(name)
  const actions = criteria.actions || []
  const actionChecks = reward.action_checks || []
  const envAssertions = criteria.env_assertions || []
  const envChecks = reward.env_assertions || []
  const nlAssertions = criteria.nl_assertions || []
  const nlChecks = reward.nl_assertions || []
  const communicateInfo = criteria.communicate_info || []
  const communicateChecks = reward.communicate_checks || []
  const actionCheck = (action, index) => actionChecks.find((item) => item.action?.action_id === action.action_id) || actionChecks[index]
  return <section className="panel verifier-rubric"><div className="section-heading"><div><h2>Verifier Rubric</h2><small>期望标准与本次实际校验结果逐项对照</small></div><div className="row"><RubricStatus passed={reward.reward == null ? null : reward.reward === 1} /><Badge>Reward {reward.reward ?? '—'}</Badge></div></div>
    <div className="rubric-basis"><strong>最终 Reward 依据</strong>{basis.length ? basis.map((name) => <Badge key={name} tone={(reward.reward_breakdown?.[name] ?? 1) === 1 ? 'good' : 'bad'}>{name}{reward.reward_breakdown?.[name] != null ? ` · ${reward.reward_breakdown[name]}` : ''}</Badge>) : <span className="muted">未记录</span>}</div>
    {(active('DB') || reward.db_check) && <RubricSection title="数据库最终状态" basis="DB" active={active('DB')}><article className="rubric-item"><div className="rubric-item-head"><strong>预测环境是否达到目标 DB 状态</strong><RubricStatus passed={reward.db_check?.db_match} /></div><p><b>期望：</b>执行任务后数据库状态与参考解法推导出的目标状态一致。参考动作 {actions.length} 项。</p><p><b>实际：</b>{reward.db_check ? `db_match=${reward.db_check.db_match}，db_reward=${reward.db_check.db_reward}` : '未保存 DB verifier 结果。'}</p>{reward.db_check?.db_match === false && <p className="notice">原始结果只保存了 DB 是否匹配及分数，没有保存逐字段状态差异。请结合下方 Action 对照和实际 Tool Results 定位导致最终状态错误的调用。</p>}</article></RubricSection>}
    {diagnostics ? <ActionDiagnostics diagnostics={diagnostics} active={active('ACTION')} /> : (actions.length > 0 || actionChecks.length > 0) && <RubricSection title="工具调用与动作" basis="ACTION" active={active('ACTION')}>{(actions.length ? actions : actionChecks.map((item) => item.action)).map((action, index) => { const check = actionCheck(action, index); return <article className="rubric-item" key={action?.action_id || index}><div className="rubric-item-head"><strong>{index + 1}. {action?.requestor || 'assistant'} → {action?.name || '未知工具'}</strong><RubricStatus passed={check?.action_match} /></div><p><b>期望参数：</b><code>{JSON.stringify(action?.arguments || {})}</code></p>{action?.compare_args?.length ? <p><b>仅比较字段：</b>{action.compare_args.join(', ')}</p> : <p className="muted">比较全部参数</p>}<p><b>实际：</b>{check ? `action_match=${check.action_match}，action_reward=${check.action_reward}，tool_type=${check.tool_type || '—'}` : '未找到对应 action check。'}</p></article>})}</RubricSection>}
    {(envAssertions.length > 0 || envChecks.length > 0) && <RubricSection title="环境断言" basis="ENV_ASSERTION" active={active('ENV_ASSERTION')}>{(envAssertions.length ? envAssertions : envChecks.map((item) => item.env_assertion)).map((assertion, index) => { const check = envChecks[index]; return <article className="rubric-item" key={`${assertion?.func_name}-${index}`}><div className="rubric-item-head"><strong>{index + 1}. {assertion?.env_type || 'environment'} · {assertion?.func_name || '断言'}</strong><RubricStatus passed={check?.met} /></div><p><b>期望值：</b><code>{JSON.stringify(assertion?.assert_value)}</code></p><p><b>调用参数：</b><code>{JSON.stringify(assertion?.arguments || {})}</code></p><p><b>实际：</b>{check ? `met=${check.met}，reward=${check.reward}` : '未保存该环境断言结果。'}</p>{assertion?.message && <p><b>失败提示：</b>{assertion.message}</p>}</article>})}</RubricSection>}
    {(nlAssertions.length > 0 || nlChecks.length > 0) && <RubricSection title="自然语言断言" basis="NL_ASSERTION" active={active('NL_ASSERTION')}>{(nlAssertions.length ? nlAssertions : nlChecks.map((item) => item.nl_assertion)).map((assertion, index) => { const check = nlChecks.find((item) => item.nl_assertion === assertion) || nlChecks[index]; return <article className="rubric-item" key={`${assertion}-${index}`}><div className="rubric-item-head"><strong>{index + 1}. 回答内容标准</strong><RubricStatus passed={check?.met} /></div><p><b>期望：</b>{assertion}</p><p><b>判定理由：</b>{check?.justification || '未保存该自然语言断言的判定。'}</p></article>})}</RubricSection>}
    {(communicateInfo.length > 0 || communicateChecks.length > 0) && <RubricSection title="必要沟通内容" basis="COMMUNICATE" active={active('COMMUNICATE')}>{(communicateInfo.length ? communicateInfo : communicateChecks.map((item) => item.info)).map((info, index) => { const check = communicateChecks.find((item) => item.info === info) || communicateChecks[index]; return <article className="rubric-item" key={`${info}-${index}`}><div className="rubric-item-head"><strong>{index + 1}. 必须向用户说明</strong><RubricStatus passed={check?.met} /></div><p><b>期望：</b>{info}</p><p><b>判定理由：</b>{check?.justification || '未保存该沟通检查的判定。'}</p></article>})}</RubricSection>}
    {!basis.length && !actions.length && !envAssertions.length && !nlAssertions.length && !communicateInfo.length && <p className="muted">该轨迹没有保存可识别的 verifier rubric。</p>}
    <details><summary>查看完整原始 Rubric 与 Reward JSON</summary><Json value={{ evaluation_criteria: criteria, reward_info: reward }} /></details>
  </section>
}

function TrainingSampleSummary({ sample }) {
  return <section className="panel training-sample"><div className="section-heading"><div><h2>训练样本</h2><small>只读 JSONL 数据视图，不代表一次带评分的评测运行</small></div><Badge tone="warn">{sample.dataset_variant} · {sample.source_kind}</Badge></div><p className="notice">{sample.note}</p><div className="kv"><span>JSONL 行</span><strong>{sample.line}</strong><span>Shard / Shard line</span><strong>{sample.slot || '—'} / {sample.shard_line ?? '—'}</strong><span>Retrieval</span><strong>{sample.retrieval_config || '—'}</strong><span>Sample hash</span><code>{sample.sample_hash || '—'}</code></div><details><summary>SFT 元数据、工具与 loss mask</summary><Json value={{ ...sample.representations.sft, loss_mask: sample.loss_mask }} /></details><details><summary>General-agent 序列化</summary><Json value={sample.representations.general_agent} /></details>{sample.audits?.length > 0 && <details open><summary>Audit / Replay 证据 · {sample.audits.length}</summary><Json value={sample.audits} /></details>}</section>
}

function TrajectoryView({ data, error, loading, reload, backTo }) {
  const [promptOpen, setPromptOpen] = useState(false)
  const [liveReview, setLiveReview] = useState(null)
  const artifactId = data?.artifact?.id
  useEffect(() => { setLiveReview(null) }, [artifactId])
  if (loading) return <Page title="轨迹" actions={<BackButton to={backTo} />}><Loading /></Page>
  const sim = data?.simulation; const training = data?.training_sample; const currentDiagnosis = liveReview?.diagnosis; const reviewTurns = new Set([...(sim?.review?.errors || []), ...(sim?.user_only_review?.errors || []), ...(currentDiagnosis?.errors || []), ...(data?.human_attributions || [])].map((e) => e.turn_idx).filter((x) => x != null)); const flaggedTurns = new Set([...reviewTurns, ...(data?.verifier_diagnostics?.failed_turns || [])])
  return <Page title={`${training ? '训练样本' : '轨迹'} · ${sim?.task_id || ''}`} subtitle={`${training ? `${training.dataset_variant}/${training.source_kind}` : data?.artifact.kind} · ${data?.artifact.trial ?? sim?.trial ?? ''}`} actions={<><BackButton to={backTo} /><button onClick={() => setPromptOpen(!promptOpen)}>{promptOpen ? '收起' : '查看'} Initial Prompts</button></>}>
    <ErrorBox error={error} />
    {data && <><div className="trace-layout"><div>
      {promptOpen && <section className="panel">{data.prompts.map((p, i) => <details key={`${p.participant}-${i}`} open><summary>{p.participant} initial prompt <Badge>{p.origin}</Badge></summary><pre className="prompt">{p.content}</pre></details>)}</section>}
      {training ? <TrainingSampleSummary sample={training} /> : <VerifierRubric task={data.task} rewardInfo={sim.reward_info} diagnostics={data.verifier_diagnostics} />}
      <section className="timeline">{(sim.messages || []).map((message, index) => <article id={`turn-${message.turn_idx ?? index}`} className={`message ${message.role} ${flaggedTurns.has(message.turn_idx) ? 'flagged' : ''}`} key={`${message.turn_idx}-${index}`}><header><strong>{message.role === 'assistant' ? 'Agent' : message.role === 'user' ? 'User' : 'Tool'}</strong><span>turn {message.turn_idx ?? index}</span>{training && message.role === 'assistant' && <Badge tone={message.training_loss ? 'good' : ''}>{message.training_loss ? '参与监督' : '不计 loss'}</Badge>}{message.generation_time_seconds != null && <span>{message.generation_time_seconds.toFixed(2)}s</span>}</header>{message.content && <pre>{message.content}</pre>}{message.tool_calls?.map((call) => <details className="tool" key={call.id || call.name} open><summary>{call.requestor} → {call.name}</summary><Json value={call.arguments} /></details>)}{message.reasoning?.map((reason, i) => <details className="reasoning" key={i}><summary>思考过程 · {reason.available ? reason.source : '未保存'}</summary>{reason.available ? <pre>{reason.text}</pre> : <p>该次模型响应没有保存可识别的 reasoning 字段。</p>}</details>)}</article>)}</section>
    </div><aside className="right-column sticky">
      <section className="panel"><h2>{training ? '样本属性' : '结果'}</h2>{training ? <div className="kv"><span>类型</span><Badge tone="warn">训练样本</Badge><span>数据视图</span><strong>{training.dataset_variant}</strong><span>来源</span><strong>{training.source_kind}</strong><span>评测结果</span><strong>未包含</strong></div> : <><div className="kv"><span>Reward</span><Badge tone={sim.reward_info?.reward === 1 ? 'good' : 'bad'}>{sim.reward_info?.reward ?? '—'}</Badge><span>终止原因</span><strong>{sim.termination_reason}</strong><span>Agent cost</span><strong>{sim.agent_cost ?? '—'}</strong><span>User cost</span><strong>{sim.user_cost ?? '—'}</strong></div><details><summary>原始 Reward Info</summary><Json value={sim.reward_info} /></details></>}</section>
      <section className="panel"><h2>模型与检索配置</h2><Json value={data.model_info} /></section>
      <section className="panel"><h2>错误归因</h2>{training ? <p className="notice">训练导出没有保存本次评测的 reward、verifier 或模型 review，因此不加载评测错误归因。</p> : <><h3>轨迹原始模型 Review</h3><Json value={{ review: sim.review, user_only_review: sim.user_only_review, synthesis_review: data.synthesis_review }} /><FixedTrajectoryDiagnosis key={data.artifact.id} artifact={data.artifact} onResult={setLiveReview} /></>}</section>
      <section className="panel"><h2>人工裁决</h2>{data.human_attributions.length ? data.human_attributions.map((a) => <div className="attribution" key={a.id}><Badge tone={a.severity === 'critical' ? 'bad' : 'warn'}>{a.source} · {a.severity}</Badge><p>{a.reasoning}</p></div>) : <p>尚无人工归因</p>}<AnnotationForm artifact={data.artifact} onSaved={reload} /></section>
    </aside></div><ScrollTopButton /></>}
  </Page>
}

function Documents() {
  const [query, setQuery] = useState(''); const [page, setPage] = useState(1); const { data, error, loading } = useLoad(() => api(`/documents?q=${encodeURIComponent(query)}&page=${page}&page_size=100`), [query, page], 5000)
  return <Page title="种子文档" subtitle="banking_knowledge 全量知识文档" actions={<Search value={query} onChange={(value) => { setQuery(value); setPage(1) }} placeholder="搜索 ID、标题或正文…" />}><ErrorBox error={error} />{loading ? <Loading /> : <><div className="document-grid">{data?.items.map((doc) => <Link className="doc-card" key={doc.id} to={`/documents/${doc.id}`}><Badge>{doc.reference_count} 引用</Badge><h3>{doc.title}</h3><code>{doc.id}</code><p>{doc.content_preview}</p></Link>)}</div><Pager data={data} page={page} onChange={setPage} /></>}</Page>
}

function DocumentDetail() {
  const { documentId } = useParams(); const { data, error, loading, reload } = useLoad(() => api(`/documents/${documentId}`), [documentId])
  if (loading) return <Page title={documentId}><Loading /></Page>
  const editable = data ? { id: data.document.id, title: data.document.title, content: data.document.content } : null
  return <Page title={data?.document.title || documentId} subtitle={documentId}><ErrorBox error={error} />{data && <div className="detail-grid"><div><section className="panel document"><pre>{data.document.content}</pre></section><Editor artifact={{ kind: 'document', id: documentId }} initial={editable} baseHash={data.sha256} onCreated={reload} /></div><aside className="right-column"><section className="panel"><h2>任务引用</h2>{data.references.map((ref) => <Link className="row-link" key={`${ref.bundle_id}-${ref.task_id}`} to={`/bundles/${ref.bundle_id}/tasks/${ref.task_id}`}>{ref.task_id}</Link>)}</section></aside></div>}</Page>
}

function Database() {
  const { data, error, loading } = useLoad(() => api('/seed-db/tables'), []); const [table, setTable] = useState(''); const [query, setQuery] = useState(''); const [page, setPage] = useState(1)
  useEffect(() => { if (!table && data?.items.length) setTable(data.items[0].name) }, [data, table])
  const records = useLoad(() => table ? api(`/seed-db/tables/${encodeURIComponent(table)}/records?q=${encodeURIComponent(query)}&page=${page}&page_size=100`) : Promise.resolve({ items: [] }), [table, query, page])
  const tableDetail = useLoad(() => table ? api(`/seed-db/tables/${encodeURIComponent(table)}`) : Promise.resolve(null), [table])
  return <Page title="种子数据库" subtitle="按表和记录浏览 banking_knowledge 基线 DB" actions={<Search value={query} onChange={(value) => { setQuery(value); setPage(1) }} />}><ErrorBox error={error || records.error || tableDetail.error} />{loading ? <Loading /> : <div className="db-layout"><div className="table-list">{data?.items.map((item) => <button className={table === item.name ? 'active' : ''} key={item.name} onClick={() => { setTable(item.name); setPage(1) }}><span>{item.name}</span><Badge>{item.records}</Badge></button>)}</div><div><section className="panel"><h2>{table}</h2><div className="record-list">{records.data?.items.map((record) => <Link className="row-link" key={record.id} to={`/database/${table}/${record.id}`}><code>{record.id}</code><span>{JSON.stringify(record.value).slice(0, 140)}</span></Link>)}</div><Pager data={records.data} page={page} onChange={setPage} /></section>{tableDetail.data && <details className="panel"><summary>编辑表级数据与 notes</summary><Editor artifact={{ kind: 'seed_table', id: table }} initial={tableDetail.data.value} baseHash={tableDetail.data.sha256} onCreated={tableDetail.reload} /></details>}</div></div>}</Page>
}

function RecordDetail() {
  const { table, recordId } = useParams(); const { data, error, loading, reload } = useLoad(() => api(`/seed-db/tables/${encodeURIComponent(table)}/records/${encodeURIComponent(recordId)}`), [table, recordId])
  if (loading) return <Page title={recordId}><Loading /></Page>
  return <Page title={recordId} subtitle={`DB · ${table}`}><ErrorBox error={error} />{data && <><section className="panel"><Json value={data.value} /></section><Editor artifact={{ kind: 'seed_record', id: recordId, task_id: table }} initial={data.value} baseHash={data.sha256} onCreated={reload} /></>}</Page>
}

function Calibration() {
  const state = useLoad(() => Promise.all([api('/change-sets'), api('/jobs'), api('/bundles'), api('/runs')]).then(([changes, jobs, bundles, runs]) => ({ changes, jobs, bundles, runs })), [], 3000)
  const [job, setJob] = useState({ type: 'check_data', bundle_id: '', run_id: '', offline: false, review_mode: 'full', review_model: '', task_ids: '', max_concurrency: 4 }); const [message, setMessage] = useState(''); const [selectedLog, setSelectedLog] = useState(null)
  async function apply(id) { if (!window.confirm('确认应用这个变更集？源文件将原子写回并创建备份。')) return; try { await api(`/change-sets/${id}/apply`, { method: 'POST' }); setMessage('变更已应用'); state.reload() } catch (e) { setMessage(e.message) } }
  async function startJob() { try { const payload = { ...job, task_ids: job.task_ids.split(',').map((value) => value.trim()).filter(Boolean) }; await api('/jobs', { method: 'POST', body: JSON.stringify(payload) }); setMessage('作业已进入队列'); state.reload() } catch (e) { setMessage(e.message) } }
  async function showLog(id) { try { setSelectedLog({ id, text: (await api(`/jobs/${id}/log`)).log }) } catch (e) { setMessage(e.message) } }
  const d = state.data
  return <Page title="校准与后台作业" subtitle="先审核 diff，再应用；命令严格白名单执行"><ErrorBox error={state.error} />{message && <div className="notice">{message}</div>}{state.loading ? <Loading /> : <div className="detail-grid"><div><section className="panel"><h2>变更集</h2>{d.changes.items.length ? d.changes.items.map((change) => <details className="change" key={change.id}><summary><Badge tone={statusTone(change.status)}>{change.status}</Badge><strong>{change.id}</strong><span>{change.rationale}</span></summary><Json value={change.validation || change.targets} />{change.status !== 'applied' && <button onClick={() => apply(change.id)}>应用变更</button>}</details>) : <p>暂无变更集。请从 task、文档或 DB 记录详情创建。</p>}</section><section className="panel"><h2>作业</h2>{d.jobs.external && <div className="external"><Badge tone={d.jobs.external.process_alive ? 'warn' : ''}>外部 · {d.jobs.external.phase}</Badge><span>{d.jobs.external.bundle}</span></div>}{d.jobs.items.map((item) => <div className="job" key={item.id}><Badge tone={statusTone(item.status)}>{item.status}</Badge><strong>{item.type}</strong><code>{item.id}</code><button onClick={() => showLog(item.id)}>日志</button>{['queued', 'running'].includes(item.status) && <button className="danger" onClick={() => api(`/jobs/${item.id}/cancel`, { method: 'POST' }).then(state.reload)}>取消</button>}</div>)}</section></div><aside className="right-column"><section className="panel job-form"><h2>启动白名单作业</h2><select value={job.type} onChange={(e) => setJob({ ...job, type: e.target.value })}><option value="check_data">check-data</option><option value="bundle_validate">bundle validate</option><option value="bundle_export">bundle export</option><option value="results_review">results review</option></select>{job.type.startsWith('bundle_') && <select value={job.bundle_id} onChange={(e) => setJob({ ...job, bundle_id: e.target.value })}><option value="">选择 bundle</option>{d.bundles.items.map((b) => <option key={b.id} value={b.id}>{b.name}</option>)}</select>}{job.type === 'bundle_validate' && <label className="check"><input type="checkbox" checked={job.offline} onChange={(e) => setJob({ ...job, offline: e.target.checked })} />离线校验</label>}{job.type === 'results_review' && <><select value={job.run_id} onChange={(e) => setJob({ ...job, run_id: e.target.value })}><option value="">选择 run</option>{d.runs.items.map((r) => <option key={r.id} value={r.id}>{r.name}</option>)}</select><select value={job.review_mode} onChange={(e) => setJob({ ...job, review_mode: e.target.value })}><option value="full">Agent + User</option><option value="user">仅 User</option></select><input placeholder="Review model（可选）" value={job.review_model} onChange={(e) => setJob({ ...job, review_model: e.target.value })} /><input placeholder="Task IDs，逗号分隔（可选）" value={job.task_ids} onChange={(e) => setJob({ ...job, task_ids: e.target.value })} /><input aria-label="最大并发" type="number" min="1" max="32" value={job.max_concurrency} onChange={(e) => setJob({ ...job, max_concurrency: Number(e.target.value) })} /></>}<button onClick={startJob}>启动</button><p className="muted">Review 可能产生模型费用，只有点击后才会运行。</p></section>{selectedLog && <section className="panel"><h2>{selectedLog.id}</h2><pre className="log">{selectedLog.text || '暂无输出'}</pre></section>}</aside></div>}</Page>
}

export default Layout
