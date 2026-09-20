# Local visualization and calibration console

The console indexes local simulation results, synthetic task bundles and the
`banking_knowledge` seed database/documents. It does not upload data and binds
to localhost by default.

```bash
cd web/visualizer
npm install
npm run build
cd ../..
tau3 web
```

Open `http://127.0.0.1:8001`. Use `--port`, `--refresh-seconds` or `--open` to
change the launch behavior. For frontend development, run `npm run dev`; Vite
proxies `/api` to the backend on port 8001.

The production server starts before the filesystem index is complete. The
overview reports the background indexing state and progressively switches to
the completed snapshot without blocking HTTP requests. Manifest discovery
ignores generation-only trees such as LLM calls, audits, teachers and nested
world workspaces; published world-package roots remain indexable. The bounded
catalog is cached at
`data/calibrations/web-manifest-index.json`. Automatic full metadata refreshes
are coalesced and run at most once every five minutes, while the browser's
five-second status polling waits for each previous request to finish.

The **Synthetic Packages** page normalizes the heterogeneous artifacts under
`data/synthetic` into a package → task hierarchy. τ3-AA bundles, world packages,
exported task batches and active streaming batches use separate read-only
adapters but share stable `package_id + task_id` namespacing. SFT exports and
pure job-state manifests are excluded. World/task-batch validation files are
shown as validation evidence and are never presented as conversation
trajectories; only explicit τ3-AA trial trajectories or unambiguous linked
simulation results receive trajectory links. The legacy `/bundles` API remains
available for calibration and synthesis jobs that require the τ3-AA layout.

The **Classic View** navigation item provides a read-only web counterpart to
`tau3 view`. It prefers `results_reviewed.json` when present, shows the same
reward, DB/action/authentication, termination, responsiveness, error-count and
review-tag columns, and supports both failure filters. Opening a row shows
the embedded task, simulation overview, complete message table, reward checks
and reviews. Tool results are truncated to 500 characters by default and can
be expanded in full.

Edits are stored as reviewable change sets under `data/calibrations/` before
they are applied. Seed edits are backed up, while edits to published synthetic
tasks create a timestamped draft fork and leave the published bundle intact.
Trajectory messages, model reasoning, rewards and model reviews are immutable;
human adjudications are stored separately.

Every regular or synthetic trajectory detail page includes a one-click failure
diagnosis action. It creates a temporary one-simulation Results snapshot under
`data/calibrations/reviews/`, then runs the existing full Agent + User review
pipeline with concurrency fixed to one. The latest persisted diagnosis is
restored after a page refresh, and errors with a `turn_idx` highlight the
corresponding timeline message. The page also exposes job status, cancellation,
logs, review/authentication costs and the structured expected behavior. Original
results and synthetic trajectory files remain unchanged.

The Jobs panel exposes only `check-data`, synthesis validate/export and explicit
conversation review. Review is never started automatically and may incur model
costs. Processes not started by the current console instance are visible but
cannot be cancelled.
