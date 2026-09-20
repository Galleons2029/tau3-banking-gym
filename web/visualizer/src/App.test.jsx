import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, describe, expect, it, vi } from 'vitest'
import App, { api, parentPagePath } from './App.jsx'

afterEach(() => { cleanup(); vi.restoreAllMocks() })

function response(payload) {
  return { ok: true, json: async () => payload }
}

describe('web console', () => {
  it('renders Chinese navigation', () => {
    vi.stubGlobal('fetch', vi.fn(() => new Promise(() => {})))
    render(<MemoryRouter><App /></MemoryRouter>)
    expect(screen.getByText('运行轨迹')).toBeTruthy()
    expect(screen.getByText('合成任务包')).toBeTruthy()
    expect(screen.getByText('经典 View')).toBeTruthy()
    expect(screen.getByText('种子数据库')).toBeTruthy()
  })

  it('shows background indexing without blocking the overview', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => response({
      index: { status: 'indexing', runs: 2, bundles: 3, completed_at: null },
      runs: 2, bundles: 3, tasks: 0, trajectories: 0,
      successful_trajectories: 0, documents: 0, database_records: 0,
      stale_bundles: 0, failure_types: [],
    })))
    render(<MemoryRouter><App /></MemoryRouter>)
    expect(await screen.findByText(/后台正在更新索引/)).toBeTruthy()
    expect(screen.getByText(/当前已载入 2 个运行结果、3 个 Bundle/)).toBeTruthy()
  })

  it('does not overlap polling while the previous request is pending', async () => {
    vi.useFakeTimers()
    try {
      const fetchMock = vi.fn(() => new Promise(() => {}))
      vi.stubGlobal('fetch', fetchMock)
      render(<MemoryRouter><App /></MemoryRouter>)
      await vi.advanceTimersByTimeAsync(20000)
      expect(fetchMock).toHaveBeenCalledTimes(1)
    } finally {
      vi.useRealTimers()
    }
  })

  it('splits runs into regular results and synthetic task trajectories', async () => {
    const fetchMock = vi.fn(async (url) => {
      if (url.includes('/synthetic-trajectories')) return response({
        items: [{
          id: 'trace_1', bundle_id: 'bundle_1', bundle_name: 'bundle-a',
          bundle_status: 'draft', task_id: 'synthetic_task_1', candidate_id: '0',
          trial: 'bm25::0', reward: 0, status: 'failed',
          termination_reason: 'max_steps', model: 'test-model', has_review: true,
          review_summary: 'wrong action',
        }],
        page: 1, page_size: 100, total: 1,
        summary: { trajectories: 1, successful: 0, failed: 1, errors: 0 },
        bundle_options: [{ id: 'bundle_1', name: 'bundle-a' }],
        filter_options: { bundle_task: ['bundle-a', 'synthetic_task_1'], candidate_trial: ['0', 'bm25::0'], termination: ['max_steps'], model: ['test-model'] },
      })
      return response({
        items: [{ id: 'run_1', name: 'regular-run', file: 'results.json', domain: 'banking_knowledge', success_count: 1, simulation_count: 2 }],
        page: 1, page_size: 100, total: 1,
        summary: { runs: 1, trajectories: 2, successful: 1 },
      })
    })
    vi.stubGlobal('fetch', fetchMock)
    render(<MemoryRouter initialEntries={['/runs']}><App /></MemoryRouter>)

    expect(await screen.findByText('regular-run')).toBeTruthy()
    fireEvent.click(screen.getByRole('tab', { name: '合成任务轨迹' }))

    expect(await screen.findByText('synthetic_task_1')).toBeTruthy()
    expect(screen.getByText('bundle-a · draft')).toBeTruthy()
    expect(screen.getByText('max_steps')).toBeTruthy()
    expect(screen.getByText('synthetic_task_1').closest('a').getAttribute('href')).toBe('/trajectories/trace_1?from=runs')
    expect(fetchMock.mock.calls.some(([url]) => url.includes('/synthetic-trajectories'))).toBe(true)

    fireEvent.change(screen.getByLabelText('按 Bundle 或 Task 筛选'), { target: { value: 'bundle-a' } })
    fireEvent.change(screen.getByLabelText('按 Candidate 或 Trial 筛选'), { target: { value: 'bm25::0' } })
    fireEvent.change(screen.getByLabelText('按终止原因筛选'), { target: { value: 'max_steps' } })
    fireEvent.change(screen.getByLabelText('按模型筛选'), { target: { value: 'test-model' } })
    fireEvent.change(screen.getByLabelText('按 Review 筛选'), { target: { value: 'reviewed' } })
    await waitFor(() => expect(fetchMock.mock.calls.some(([url]) => url.includes('bundle_task=bundle-a') && url.includes('candidate_trial=bm25%3A%3A0') && url.includes('termination=max_steps') && url.includes('model=test-model') && url.includes('review=reviewed'))).toBe(true))

    fireEvent.click(screen.getByText('清空筛选'))
    expect(screen.getByLabelText('按 Bundle 或 Task 筛选').value).toBe('')
    expect(screen.getByLabelText('按 Review 筛选').value).toBe('')
    await act(async () => { await Promise.resolve() })
  })

  it('shows temporary data/synthesis rounds and loads selected tasks on demand', async () => {
    const fetchMock = vi.fn(async (url) => {
      if (url.includes('/synthesis-samples/rounds/round_1/tasks')) return response({
        round: { id: 'round_1', name: 'targeted-native-test' },
        items: [{
          id: 'native_pilot_00000', stage: 'pilot', slot: '00000',
          family: 'credit_limit', difficulty: '1-4',
          purpose: 'Temporary credit limit sample',
          target_operations: ['lookup', 'update'], sample_count: 2, error: null,
        }],
        page: 1, page_size: 100, total: 1,
        stages: ['pilot'], families: ['credit_limit'],
        summary: { tasks: 1, training_samples: 2, errors: 0 },
      })
      return response({
        items: [{
          id: 'round_1', name: 'targeted-native-test', relative_path: 'synthesis/targeted-native-test',
          active: true, status: 'INCOMPLETE', stage: 'pilot', parent: null,
          has_tasks: true, reports: { pilot: { valid_tasks: 1, expected_tasks: 1, qualified_rows: 2 } },
        }],
        page: 1, page_size: 100, total: 1,
        summary: { rounds: 1, active: 1, with_tasks: 1 },
      })
    })
    vi.stubGlobal('fetch', fetchMock)
    render(<MemoryRouter initialEntries={['/runs?view=synthesis']}><App /></MemoryRouter>)

    expect(await screen.findByText('targeted-native-test')).toBeTruthy()
    expect(screen.getAllByText('活动轮次').length).toBeGreaterThan(0)
    fireEvent.click(screen.getByText('targeted-native-test'))

    expect(await screen.findByText('native_pilot_00000')).toBeTruthy()
    expect(screen.getByText('Temporary credit limit sample')).toBeTruthy()
    expect(screen.getByText('native_pilot_00000').closest('a').getAttribute('href')).toBe('/synthesis-samples/round_1/tasks/native_pilot_00000')
    expect(fetchMock.mock.calls.some(([url]) => url.includes('/rounds/round_1/tasks'))).toBe(true)
  })

  it('drills from a temporary task into its exported training trajectory', async () => {
    const fetchMock = vi.fn(async (url) => {
      if (url.includes('/rows/row_1')) return response({
        artifact: { kind: 'trajectory', id: 'sample_1', package_id: 'round_1', task_id: 'native_pilot_00000', trial: 42 },
        simulation: {
          task_id: 'native_pilot_00000', trial: 42, termination_reason: 'training_sample',
          reward_info: { reward: null },
          messages: [{ role: 'assistant', turn_idx: 2, content: 'Checked eligibility', training_loss: true, reasoning: [{ available: true, source: 'message.reasoning_content', text: 'Verify first' }] }],
        },
        task: { id: 'native_pilot_00000' }, prompts: [], model_info: {},
        verifier_diagnostics: null, synthesis_review: null, human_attributions: [],
        training_sample: {
          dataset_variant: 'pilot', source_kind: 'temporary_synthesis', line: 1, slot: '00000', shard_line: null,
          sample_hash: 'hash', retrieval_config: 'bm25_grep', loss_mask: [0, 0, 1],
          note: '这是 data/synthesis 的临时训练样本。', audits: [{ tokens: 120 }],
          representations: { sft: { row_id: 'row_1', seed: 42 }, general_agent: { messages: [] } },
        },
      })
      return response({
        round: { id: 'round_1', name: 'targeted-native-test' }, stage: 'pilot', slot: '00000',
        task: { id: 'native_pilot_00000', description: { purpose: 'Temporary sample' } },
        candidate: { checks: { reference: 'PASS' } }, provenance: { slot: { family: 'credit_limit' } },
        samples: [{ row_id: 'row_1', stage: 'pilot', line: 1, seed: 42, weight: 1, token_audit: { tokens: 120 } }],
      })
    })
    vi.stubGlobal('fetch', fetchMock)
    render(<MemoryRouter initialEntries={['/synthesis-samples/round_1/tasks/native_pilot_00000']}><App /></MemoryRouter>)

    fireEvent.click(await screen.findByText('pilot · line 1'))
    expect(await screen.findByText('Checked eligibility')).toBeTruthy()
    expect(screen.getByText('Verify first')).toBeTruthy()
    expect(screen.getByText('参与监督')).toBeTruthy()
    expect(screen.getByText(/临时训练样本/)).toBeTruthy()
  })

  it('keeps filters visible for empty synthetic results and offers existing values', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => response({
      items: [], page: 1, page_size: 100, total: 0,
      summary: { trajectories: 0, successful: 0, failed: 0, errors: 0 },
      bundle_options: [{ id: 'bundle_1', name: 'bundle-a' }],
      filter_options: { bundle_task: ['bundle-a', 'task_1'], candidate_trial: ['candidate_0'], termination: ['max_steps'], model: ['model-a'] },
    })))
    render(<MemoryRouter initialEntries={['/runs?view=synthetic']}><App /></MemoryRouter>)

    expect(await screen.findByText('没有符合条件的合成轨迹或训练样本。')).toBeTruthy()
    expect(screen.getByLabelText('按 Bundle 或 Task 筛选').getAttribute('list')).toBe('bundle-task-options')
    expect(screen.getByLabelText('按终止原因筛选').getAttribute('list')).toBe('termination-options')
    expect(document.querySelector('#bundle-task-options option[value="task_1"]')).toBeTruthy()
    expect(document.querySelector('#model-options option[value="model-a"]')).toBeTruthy()
  })

  it('groups targeted packages and loads the selected package on demand', async () => {
    const packageId = 'package_72eecdbca86914e5'
    const fetchMock = vi.fn(async (url) => {
      const selected = url.includes(`bundle_id=${packageId}`)
      return response({
        items: selected ? [{
          id: 'trace_targeted', bundle_id: packageId,
          bundle_name: 'targeted-v1-round-000-independent-formal-budget',
          bundle_status: 'finished', package_prefix: 'targeted-v1',
          task_id: 'task_case_003e37eabe90_review', task_prefix: 'task_case_review',
          candidate_id: '0', trial: '0', reward: 0, status: 'failed',
          termination_reason: 'max_steps', model: 'teacher-model', has_review: false,
        }] : [],
        page: 1, page_size: 100, total: selected ? 1 : 0,
        summary: { trajectories: selected ? 1 : 0, successful: 0, failed: selected ? 1 : 0, errors: 0 },
        package_groups: [{ prefix: 'targeted-v1', packages: 8, tasks: 3017, trajectories: selected ? 11854 : 0 }],
        bundle_options: [{ id: packageId, name: 'targeted-v1-round-000-independent-formal-budget', prefix: 'targeted-v1', format: 'targeted_round', loaded: selected, task_count: 2964, trajectory_count: selected ? 11854 : 0 }],
        task_prefixes: selected ? ['task_case_review'] : [],
        matching_packages: [],
        filter_options: { bundle_task: [], candidate_trial: [], termination: [], model: [] },
      })
    })
    vi.stubGlobal('fetch', fetchMock)
    render(<MemoryRouter initialEntries={['/runs?view=synthetic']}><App /></MemoryRouter>)

    expect(await screen.findByText(/targeted-v1.*8 包/)).toBeTruthy()
    fireEvent.change(screen.getByLabelText('任务包系列'), { target: { value: 'targeted-v1' } })
    fireEvent.change(screen.getByLabelText('具体任务包'), { target: { value: packageId } })

    expect(await screen.findByText('task_case_003e37eabe90_review')).toBeTruthy()
    expect(screen.getAllByText('task_case_review').length).toBeGreaterThan(0)
    expect(fetchMock.mock.calls.some(([url]) => url.includes(`bundle_id=${packageId}`))).toBe(true)
  })

  it('maps every nested resource page to its parent page', () => {
    expect(parentPagePath('/classic-view/run_1')).toBe('/classic-view')
    expect(parentPagePath('/runs/run_1')).toBe('/runs')
    expect(parentPagePath('/synthetic-packages/package_1')).toBe('/synthetic-packages')
    expect(parentPagePath('/synthetic-packages/package_1/tasks/task_1')).toBe('/synthetic-packages/package_1')
    expect(parentPagePath('/bundles/bundle_1')).toBe('/bundles')
    expect(parentPagePath('/bundles/bundle_1/tasks/task_1')).toBe('/bundles/bundle_1')
    expect(parentPagePath('/documents/doc_1')).toBe('/documents')
    expect(parentPagePath('/database/users/user_1')).toBe('/database')
    expect(parentPagePath('/documents')).toBeNull()
  })

  it('navigates from a normalized synthetic package to its child task', async () => {
    vi.stubGlobal('fetch', vi.fn(async (url) => {
      if (url.includes('/synthetic-packages/package_1/tasks/world_task_1')) return response({
        package: { id: 'package_1', name: 'natural', format: 'world_package', status: 'published', editable: false },
        task: { id: 'world_task_1', description: { purpose: 'Grounding task' }, evaluation_criteria: { nl_assertions: ['回答必须基于文档'], reward_basis: ['NL_ASSERTION'] }, required_documents: ['doc_1'] },
        source: 'world_task', splits: ['test'], candidate: null,
        validation: { package: { status: 'PASS', files: ['validation/certificate.json'] }, task_files: [] },
        trajectories: [], run_simulations: [], sha256: 'abc',
      })
      if (url.includes('/synthetic-packages/package_1/tasks')) return response({
        package: { id: 'package_1', name: 'natural', format: 'world_package', status: 'published', relative_path: 'worldgen/natural', validation: { status: 'PASS' } },
        items: [{ id: 'world_task_1', family: 'grounding', purpose: 'Grounding task', source: 'world_task', status: 'published', accepted: null, splits: ['test'], trajectory_count: 0, required_document_count: 1 }],
        page: 1, page_size: 100, total: 1,
      })
      return response({
        items: [{ id: 'package_1', name: 'natural', relative_path: 'worldgen/natural', format: 'world_package', status: 'published', task_count: 1, trajectory_count: 0, validation: { status: 'PASS' } }],
        page: 1, page_size: 100, total: 1,
        summary: { packages: 1, tasks: 1, trajectories: 0 }, formats: ['world_package'],
      })
    }))
    render(<MemoryRouter initialEntries={['/synthetic-packages']}><App /></MemoryRouter>)

    fireEvent.click(await screen.findByText('natural'))
    fireEvent.click(await screen.findByText('world_task_1'))

    expect(await screen.findByText('完整任务定义')).toBeTruthy()
    expect(screen.getByText('回答必须基于文档')).toBeTruthy()
    expect(screen.getByText(/"status": "PASS"/)).toBeTruthy()
    expect(screen.getByText('← 返回上一级').closest('a').getAttribute('href')).toBe('/synthetic-packages/package_1')
  })

  it('shows the parent button immediately while a detail page is loading', () => {
    vi.stubGlobal('fetch', vi.fn(() => new Promise(() => {})))
    render(<MemoryRouter initialEntries={['/bundles/bundle_1/tasks/task_1']}><App /></MemoryRouter>)
    expect(screen.getByText('← 返回上一级').closest('a').getAttribute('href')).toBe('/bundles/bundle_1')
  })

  it('surfaces API errors', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => ({ ok: false, status: 409, statusText: 'Conflict', json: async () => ({ detail: 'hash conflict' }) })))
    await expect(api('/change-sets')).rejects.toThrow('hash conflict')
  })

  it('explains when an old backend returns the SPA HTML for an API request', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => ({
      ok: true,
      headers: { get: () => 'text/html; charset=utf-8' },
      text: async () => '<!doctype html><html></html>',
    })))
    await expect(api('/classic-view/runs')).rejects.toThrow('请重启 tau3 web')
  })

  it('renders reasoning availability and highlights reviewed turns', async () => {
    vi.stubGlobal('fetch', vi.fn(async (url) => {
      if (url.includes('/jobs?')) return response({ items: [], external: null })
      return response({
      artifact: { kind: 'trajectory', id: 'trace_1', trial: 0, bundle_id: 'bundle_1', task_id: 'task__1' },
      simulation: {
        task_id: 'task__1', termination_reason: 'max_steps',
        reward_info: {
          reward: 0, reward_basis: ['ACTION', 'NL_ASSERTION'], reward_breakdown: { ACTION: 0, NL_ASSERTION: 1 },
          action_checks: [{ action: { action_id: 'a1', requestor: 'assistant', name: 'lookup', arguments: { id: '1' } }, action_match: false, action_reward: 0, tool_type: 'read' }],
          nl_assertions: [{ nl_assertion: '回答必须准确', met: true, justification: '回答与资料一致' }],
        },
        review: { errors: [{ turn_idx: 2, reason: 'wrong tool' }] },
        messages: [{ role: 'assistant', turn_idx: 2, content: '回答', reasoning: [{ participant: 'agent', available: false }] }],
      },
      task: { evaluation_criteria: { actions: [{ action_id: 'a1', requestor: 'assistant', name: 'lookup', arguments: { id: '1' } }], nl_assertions: ['回答必须准确'], reward_basis: ['ACTION', 'NL_ASSERTION'] } },
      verifier_diagnostics: {
        premature_termination: true, termination_reason: 'max_steps', reward_info_note: null,
        summary: { expected: 1, matched: 0, argument_mismatch: 1, missing: 0, not_evaluated: 0, extra: 1 },
        failed_turns: [2],
        actions: [{
          index: 0, action_id: 'a1', status: 'argument_mismatch',
          expected: { action_id: 'a1', requestor: 'assistant', name: 'lookup', arguments: { id: '1' }, compare_args: ['id'] },
          check: { action_match: false, action_reward: 0, tool_type: 'read' }, matched_call: null,
          observed_candidates: [{ key: '0:0', turn_idx: 2, requestor: 'assistant', name: 'lookup', arguments: { id: '2' }, argument_diffs: [{ field: 'id', expected: '1', actual: '2', expected_present: true, actual_present: true }] }],
        }],
        extra_calls: [{ key: '0:0', turn_idx: 2, requestor: 'assistant', name: 'lookup', arguments: { id: '2' } }],
      },
      prompts: [{ participant: 'user', origin: 'reconstructed', content: 'prompt' }],
      human_attributions: [], synthesis_review: null,
      })
    }))
    render(<MemoryRouter initialEntries={['/trajectories/trace_1']}><App /></MemoryRouter>)
    expect(await screen.findByText('回答')).toBeTruthy()
    expect(screen.getByText('思考过程 · 未保存')).toBeTruthy()
    expect(screen.getByText('Verifier Rubric')).toBeTruthy()
    expect(screen.getByText('工具调用与动作对照')).toBeTruthy()
    expect(screen.getByText('参数不匹配')).toBeTruthy()
    expect(screen.getByText('额外或重复调用 · 1')).toBeTruthy()
    expect(document.querySelector('.argument-diff')).toBeTruthy()
    expect(document.querySelector('a[href="#turn-2"]')).toBeTruthy()
    expect(screen.getByText('回答与资料一致')).toBeTruthy()
    expect(document.querySelector('.message.flagged')).toBeTruthy()
    expect(screen.getByText('← 返回上一级').closest('a').getAttribute('href')).toBe('/bundles/bundle_1/tasks/task__1')

    const scrollTo = vi.spyOn(window, 'scrollTo').mockImplementation(() => {})
    Object.defineProperty(window, 'scrollY', { value: 700, configurable: true })
    fireEvent.scroll(window)
    fireEvent.click(await screen.findByLabelText('回到页面顶部'))
    expect(scrollTo).toHaveBeenCalledWith({ top: 0, behavior: 'smooth' })
    Object.defineProperty(window, 'scrollY', { value: 0, configurable: true })
  })

  it('loads a fixed trajectory diagnosis and highlights diagnosed turns', async () => {
    const detail = {
      artifact: { kind: 'trajectory', id: 'trace_1', trial: 0, bundle_id: 'bundle_1', task_id: 'task__1' },
      simulation: {
        task_id: 'task__1', termination_reason: 'max_steps', reward_info: { reward: 0 },
        review: null, user_only_review: null,
        messages: [{ role: 'assistant', turn_idx: 2, content: '错误回答', reasoning: [] }],
      },
      prompts: [], human_attributions: [], synthesis_review: null, model_info: {},
    }
    const fetchMock = vi.fn(async (url, options = {}) => {
      if (url.endsWith('/fixed-diagnoses/resolve') && options.method === 'POST') {
        return response({
          source: 'fixed_file', status: 'matched', artifact: detail.artifact,
          package: { kind: 'synthetic_package', id: 'bundle_1', name: 'bundle' },
          task_id: 'task__1', trial: 0,
          expected_file: 'calibrations/failure-attributions/synthetic/bundle_1/task__1.json',
          diagnosis: {
            artifact_id: 'trace_1', trial: 0,
            summary: 'Agent 使用了错误工具', has_errors: true,
            errors: [{ source: 'agent', severity: 'critical', turn_idx: 2, error_tags: ['incorrect_tool'], reasoning: '工具与请求无关', correct_behavior: '应先读取账户信息' }],
          },
        })
      }
      return response(detail)
    })
    vi.stubGlobal('fetch', fetchMock)
    render(<MemoryRouter initialEntries={['/trajectories/trace_1']}><App /></MemoryRouter>)

    expect(await screen.findByText('Agent 使用了错误工具')).toBeTruthy()
    expect(screen.getByText('incorrect_tool')).toBeTruthy()
    expect(screen.getByText('来源 · 固定文件')).toBeTruthy()
    expect(document.querySelector('.message.flagged')).toBeTruthy()
    fireEvent.click(screen.getByText('重新加载固定诊断'))
    await waitFor(() => expect(fetchMock.mock.calls.filter(([url]) => url.endsWith('/fixed-diagnoses/resolve')).length).toBe(2))
    const post = fetchMock.mock.calls.find(([url, options]) => url.endsWith('/fixed-diagnoses/resolve') && options?.method === 'POST')
    expect(JSON.parse(post[1].body)).toMatchObject({
      kind: 'trajectory', id: 'trace_1', bundle_id: 'bundle_1', task_id: 'task__1', trial: 0,
    })
    expect(fetchMock.mock.calls.some(([url]) => url.endsWith('/jobs'))).toBe(false)
  })

  it('renders training JSONL samples without pretending they were evaluated', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => response({
      artifact: { kind: 'trajectory', id: 'training_1', package_id: 'package_1', task_id: 'task_case_review', trial: 42 },
      simulation: {
        task_id: 'task_case_review', trial: 0, termination_reason: 'training_sample',
        reward_info: { reward: null, info: { note: '未包含评测结果' } },
        messages: [
          { role: 'user', turn_idx: 1, content: '请求', training_loss: false, reasoning: [] },
          { role: 'assistant', turn_idx: 2, content: '训练回答', training_loss: true, reasoning: [] },
        ],
      },
      task: { id: 'task_case_review', evaluation_criteria: { actions: [{ name: 'lookup' }], reward_basis: ['ACTION'] } },
      prompts: [{ participant: 'agent', origin: 'saved', content: 'system' }],
      model_info: { agent: { llm: 'teacher-model' }, retrieval_config: 'bm25_grep' },
      verifier_diagnostics: null, synthesis_review: null, human_attributions: [],
      training_sample: {
        dataset_variant: 'balanced', source_kind: 'full', line: 42, slot: '000001', shard_line: 3,
        sample_hash: 'sample-hash', retrieval_config: 'bm25_grep', loss_mask: [0, 0, 1],
        note: '这是训练样本而非评测运行。', audits: [{ replay: { status: 'PASS' } }],
        representations: {
          sft: { schema_version: 2, task_id: 'task_case_review', tools: [] },
          general_agent: { messages: [{ role: 'assistant', content: '训练回答', loss: true }], tools: [] },
        },
      },
    })))
    render(<MemoryRouter initialEntries={['/trajectories/training_1']}><App /></MemoryRouter>)

    expect(await screen.findByRole('heading', { name: '训练样本' })).toBeTruthy()
    expect(screen.getByText('训练回答')).toBeTruthy()
    expect(screen.getByText('参与监督')).toBeTruthy()
    expect(screen.getByText('Audit / Replay 证据 · 1')).toBeTruthy()
    expect(screen.getByText(/没有保存本次评测的 reward/)).toBeTruthy()
    expect(screen.queryByText('重新加载固定诊断')).toBeNull()
    expect(screen.queryByText('Verifier Rubric')).toBeNull()
  })

  it('paginates the complete document collection', async () => {
    const fetchMock = vi.fn(async () => response({ items: [], page: 1, page_size: 100, total: 698 }))
    vi.stubGlobal('fetch', fetchMock)
    render(<MemoryRouter initialEntries={['/documents']}><App /></MemoryRouter>)
    fireEvent.click(await screen.findByText('下一页'))
    await waitFor(() => expect(fetchMock.mock.calls.some(([url]) => url.includes('page=2'))).toBe(true))
  })

  it('recreates the tau3 view summary and failure filters', async () => {
    const fetchMock = vi.fn(async (url) => {
      if (url.endsWith('/tasks')) return response({ items: [] })
      return response({ items: [{ index: 1, id: 'sim_1', task_id: 'task_1', trial: 0, reward: 0, db_match: false, read_actions: { correct: 1, count: 2 }, write_actions: null, auth_status: 'failed', termination_reason: 'max_steps', had_unresponsive_period: true, agent_error_count: 1, user_error_count: 0, agent_critical: true, user_critical: false, agent_tags: ['hallucination'], user_tags: [], has_review: true }], page: 1, page_size: 50, total: 1, failed_task_count: 1 })
    })
    vi.stubGlobal('fetch', fetchMock)
    render(<MemoryRouter initialEntries={['/classic-view/run_1']}><App /></MemoryRouter>)
    expect(await screen.findByText('task_1')).toBeTruthy()
    expect(screen.getByText('hallucination')).toBeTruthy()
    expect(screen.getByText('← 返回上一级').closest('a').getAttribute('href')).toBe('/classic-view')
    fireEvent.change(screen.getByLabelText('轨迹筛选'), { target: { value: 'failed' } })
    await waitFor(() => expect(fetchMock.mock.calls.some(([url]) => url.includes('mode=failed'))).toBe(true))
  })

  it('truncates classic tool results until full output is requested', async () => {
    const longResult = 'x'.repeat(600)
    vi.stubGlobal('fetch', vi.fn(async (url) => {
      if (url.endsWith('/fixed-diagnoses/resolve')) return response({ status: 'file_missing', expected_file: 'calibrations/failure-attributions/simulations/run_1/task_1.json', diagnosis: null })
      return response({
        artifact: { kind: 'simulation', id: 'sim_1', run_id: 'run_1', task_id: 'task_1', trial: 0 },
        task: { id: 'task_1' },
        simulation: { id: 'sim_1', task_id: 'task_1', trial: 0, duration: 1, termination_reason: 'user_stop', reward_info: { reward: 1 }, messages: [{ role: 'tool', id: 'tool_1', turn_idx: 1, content: longResult }] },
      })
    }))
    render(<MemoryRouter initialEntries={['/classic-view/run_1/simulations/sim_1']}><App /></MemoryRouter>)
    expect(await screen.findByText(/已截断 100 字符/)).toBeTruthy()
    expect(await screen.findByText('重新加载固定诊断')).toBeTruthy()
    fireEvent.click(screen.getByLabelText('完整 Tool Results'))
    expect(screen.getByText(longResult)).toBeTruthy()
  })
})
