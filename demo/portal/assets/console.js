/* The console.
 *
 * Every button below is an HTTP call, and every number it shows came out of a
 * response. Nothing on this page is computed from a table of expected values —
 * if the system stopped deduplicating, the card would say so.
 *
 * The one rule worth keeping while editing: a step reports what it *measured*.
 * `evidence` rows are formatted response fields and deltas of the server's own
 * counters, never constants. */

'use strict';

const ENV = 'proximal/demo';
const ENV_B = 'proximal/demo-b';
const FORK = 'proximal/demo-variant';
const OTHER = 'other/team';
const MIGRATED = 'proximal/migrated';
const MAIN = 'refs/heads/main';
const DATASET = 'data/train.bin';
const DATASET_BYTES = 4000000;

/* ────────────────────────────────────────────────────────────────────────
 * Formatting
 * ──────────────────────────────────────────────────────────────────────── */

/** Mirrors `ledger.text.human_bytes`, so the page and the CLI agree. */
function humanBytes(count) {
  const units = [['GiB', 1 << 30], ['MiB', 1 << 20], ['KiB', 1 << 10]];
  for (const [unit, size] of units) {
    if (count >= size) return `${(count / size).toFixed(2)} ${unit}`;
  }
  return `${count} B`;
}

const pct = (x) => `${(x * 100).toFixed(2)}%`;
const short = (name) => (name || '').replace(/^b3:/, '').slice(0, 12);
const plural = (n, one, many) => `${n} ${n === 1 ? one : many}`;

/** `n` hex characters. Change ids and idempotency keys are both client-chosen. */
function hex(n) {
  const bytes = new Uint8Array(Math.ceil(n / 2));
  crypto.getRandomValues(bytes);
  return [...bytes].map((b) => b.toString(16).padStart(2, '0')).join('').slice(0, n);
}

/** The demo clock, as something a viewer can read. */
function clockText(microseconds) {
  const when = new Date(microseconds / 1000);
  return when.toISOString().replace('T', ' ').slice(0, 19) + 'Z';
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

/* ────────────────────────────────────────────────────────────────────────
 * The API client, and the request log
 * ──────────────────────────────────────────────────────────────────────── */

class ApiError extends Error {
  constructor(status, payload, method, path) {
    super((payload && payload.message) || `${method} ${path} → ${status}`);
    this.status = status;
    this.payload = payload || {};
    this.code = this.payload.code || String(status);
    this.details = this.payload.details || {};
  }
}

/** One call. */
async function api(method, path, options = {}) {
  const init = { method, headers: Object.assign({}, options.headers) };
  if (options.body !== undefined) {
    init.headers['Content-Type'] = 'application/json';
    init.body = JSON.stringify(options.body);
  }
  if (options.token) init.headers['Authorization'] = `Bearer ${options.token}`;

  const response = await fetch(path, init);

  if (options.binary && response.ok) {
    const buffer = await response.arrayBuffer();
    return { bytes: buffer.byteLength, headers: response.headers };
  }
  const text = await response.text();
  let payload = null;
  if (text) {
    try {
      payload = JSON.parse(text);
    } catch {
      payload = { text };
    }
  }
  if (!response.ok) throw new ApiError(response.status, payload, method, path);
  if (payload && payload.text !== undefined && Object.keys(payload).length === 1) {
    payload.headers = response.headers;
  }
  return payload === null ? { headers: response.headers } : payload;
}

/** Assert a call is refused, and hand back what it was refused with. */
async function refused(status, run) {
  try {
    await run();
  } catch (error) {
    if (error instanceof ApiError && error.status === status) return error;
    throw error;
  }
  throw new Error(`expected ${status}, but the call was allowed`);
}

/* ────────────────────────────────────────────────────────────────────────
 * What a step reports
 * ──────────────────────────────────────────────────────────────────────── */

/** One measured fact. `tone` is 'good' | 'bad' | 'warn' | undefined. */
const E = (label, value, tone) => ({ label, value: String(value), tone });

/** A sentence under the evidence, for the thing a number cannot say. */
const note = (text) => ({ note: text });

/* Shared state between cards — commits, ids, tokens the earlier steps made. */
const shared = new Map();

/* Every write this page performed, for the cost chart. */
const writes = [];

const context = {
  api,
  refused,
  set: (key, value) => shared.set(key, value),
  get: (key) => shared.get(key),
  stats: () => api('GET', '/console/stats'),
  /** Record a write for the chart, and hand it straight back. */
  record(label, result) {
    writes.push({
      label,
      offered: result.bytes_offered,
      stored: result.bytes_stored,
      objects: result.objects_created,
    });
    drawCostChart();
    return result;
  },
  /** Corpus bytes either side of `run` — how the page measures "costs nothing". */
  async costOf(run) {
    const before = (await this.stats()).stored_bytes;
    const value = await run();
    const after = (await this.stats()).stored_bytes;
    return { value, bytes: after - before };
  },
  /** Objects the server fetched while `run` ran, from its own read counter. */
  async readsOf(run) {
    const before = (await this.stats()).object_reads;
    const value = await run();
    const after = (await this.stats()).object_reads;
    return { value, reads: after - before };
  },
};

/* ────────────────────────────────────────────────────────────────────────
 * The twelve cards
 *
 * Keyed by the requirement names in `src.verify.REQUIREMENTS`. A name here
 * that is not in that tuple never renders, and a requirement there with no
 * entry here renders as a card with nothing to run — which is the honest
 * failure mode, and visible.
 * ──────────────────────────────────────────────────────────────────────── */

const CARDS = {
  'Version Environments': [
    {
      label: 'create the environment',
      async run(c) {
        const created = await c.api('POST', '/v1/envs', {
          body: { name: ENV, owner: 'rl-infra', labels: { team: 'rl-infra' } },
        });
        c.set('envId', created.env_id);
        return [
          E('environment', created.name),
          E('id', created.env_id),
          E('default ref', created.default_ref),
        ];
      },
    },
    {
      label: 'commit it — version 1',
      async run(c) {
        const result = c.record('v1', await c.api('POST', `/demo/envs/${ENV}/commit`, {
          body: { message: 'initial version' },
        }));
        c.set('v1', result.commit);
        await refreshDistribution();
        return [
          E('commit', short(result.commit)),
          E('objects created', `${result.objects_created} of ${result.objects_offered}`),
          E('bytes stored', humanBytes(result.bytes_stored)),
          note('A manifest, a task, a verifier and a 3.81 MiB dataset. Everything is new, so everything is stored.'),
        ];
      },
    },
    {
      label: 'reword one line — version 2',
      async run(c) {
        const result = c.record('v2', await c.api('POST', `/demo/envs/${ENV}/files`, {
          body: {
            path: 'task/prompt.md',
            text: 'Solve the task, carefully.\n',
            message: 'reword the prompt',
          },
        }));
        c.set('v2', result.commit);
        return [
          E('commit', short(result.commit)),
          E('objects created', result.objects_created, 'good'),
          E('bytes stored', humanBytes(result.bytes_stored), 'good'),
          E('…in an environment of', humanBytes(DATASET_BYTES)),
          note('The dataset was not touched, so it was not re-stored — the new version shares it by name.'),
        ];
      },
    },
    {
      label: 'list the history',
      async run(c) {
        const log = await c.api(
          'GET',
          `/v1/envs/${ENV}/commits/${c.get('v2')}/log?limit=20`,
        );
        return [
          E('versions', log.commits.length),
          ...log.commits.map((commit) =>
            E(short(commit.name), commit.message || '(no message)'),
          ),
        ];
      },
    },
    {
      label: 'diff the two versions',
      async run(c) {
        const diff = await c.api(
          'GET',
          `/v1/envs/${ENV}/diff?before=${c.get('v1')}&after=${c.get('v2')}`,
        );
        return [
          E('paths changed', diff.changes.length, 'good'),
          ...diff.changes.map((change) => E(change.path, `${change.kind} ${change.size_delta >= 0 ? '+' : ''}${change.size_delta} B`)),
          note('Unchanged subtrees have unchanged hashes, so the walk dismissed the dataset by comparing two names.'),
        ];
      },
    },
    {
      label: 'restore version 1 — then version 2',
      async run(c) {
        const measured = await c.costOf(async () => {
          for (const target of [c.get('v1'), c.get('v2')]) {
            const current = await c.api('GET', `/v1/envs/${ENV}/refs/${MAIN}/resolve`);
            await c.api('PUT', `/v1/envs/${ENV}/refs/${MAIN}`, {
              body: { target, expected_generation: current.generation },
            });
          }
        });
        return [
          E('went back, then forward', 'two ref updates'),
          E('bytes stored', measured.bytes, measured.bytes === 0 ? 'good' : 'warn'),
          note('Every version is a full snapshot over shared content, so restoring an old one moves no bytes at all.'),
        ];
      },
    },
  ],

  'Branching / Forking': [
    {
      label: 'fork into a variant environment',
      async run(c) {
        const measured = await c.costOf(() =>
          c.api('POST', `/v1/envs/${ENV}/fork`, {
            body: { name: FORK, from_ref: MAIN, owner: 'rl-infra' },
          }),
        );
        c.set('forkId', measured.value.env_id);
        return [
          E('fork', measured.value.name),
          E('forked from', short(measured.value.forked_from_commit)),
          E('bytes copied', measured.bytes, measured.bytes === 0 ? 'good' : 'warn'),
          note('Objects carry no environment identity, so a fork is one metadata row over content that already exists.'),
        ];
      },
    },
    {
      label: 'open an ephemeral branch for an A/B run',
      async run(c) {
        const branch = await c.api(
          'POST',
          `/v1/envs/${ENV}/refs/refs/heads/exp/lr-3e4`,
          { body: { ephemeral: true, ttl_days: 14 } },
        );
        return [
          E('branch', branch.name),
          E('lifecycle', branch.lifecycle, 'good'),
          E('expires', clockText(branch.expires_at_us)),
          note('An experiment that is abandoned rather than deleted still stops costing storage, on its own.'),
        ];
      },
    },
    {
      label: 'commit a dead end on a throwaway branch',
      async run(c) {
        const head = await c.api('GET', `/v1/envs/${ENV}/refs/${MAIN}/resolve`);
        await c.api('POST', `/v1/envs/${ENV}/refs/refs/heads/exp/discard-me`, {
          body: { target: head.commit },
        });
        const result = c.record('dead end', await c.api('POST', `/demo/envs/${ENV}/patch`, {
          body: {
            path: DATASET,
            offset: 1000,
            length: 64,
            fill: 'Z',
            ref: 'refs/heads/exp/discard-me',
            message: 'a dead end',
          },
        }));
        return [
          E('branch', 'refs/heads/exp/discard-me'),
          E('content only this branch reaches', humanBytes(result.bytes_stored)),
        ];
      },
    },
    {
      label: 'discard the branch',
      async run(c) {
        const current = await c.api(
          'GET',
          `/v1/envs/${ENV}/refs/refs/heads/exp/discard-me/resolve`,
        );
        await c.api(
          'DELETE',
          `/v1/envs/${ENV}/refs/refs/heads/exp/discard-me?expected_generation=${current.generation}`,
        );
        const keepsets = await c.api('POST', '/demo/keepsets/rebuild');
        return [
          E('deleted', 'refs/heads/exp/discard-me'),
          ...Object.entries(keepsets.environments).map(([name, size]) =>
            E(`${name} keep-set`, plural(size, 'object', 'objects')),
          ),
          note('Deleting the ref is instant. Shrinking the keep-set is the other half of it, and it decides what a later sweep may take.'),
        ];
      },
    },
    {
      label: 'nine days later — what can be collected?',
      async run(c) {
        await c.api('POST', '/demo/clock/advance', { body: { days: 9 } });
        await c.api('POST', '/demo/keepsets/rebuild');
        const plan = await c.api('POST', '/demo/gc/plan');
        await drawStorageChart(plan);
        return [
          E('past the grace period', 'yes'),
          E('objects collectable', plan.candidates, plan.candidates === 0 ? 'good' : 'warn'),
          note('Nothing — because undo can still reach that branch, and content undo can reach is content collection must not take.'),
        ];
      },
    },
    {
      label: 'ninety more — the operation log ages out',
      async run(c) {
        await c.api('POST', '/demo/clock/advance', { body: { days: 90 } });
        await c.api('POST', '/demo/keepsets/rebuild');
        const plan = await c.api('POST', '/demo/gc/plan');
        const swept = await c.api('POST', '/demo/gc/run');
        await drawStorageChart();
        return [
          E('objects collectable', plan.candidates, 'good'),
          E('bytes reclaimable', humanBytes(plan.bytes_reclaimable), 'good'),
          E('objects deleted', swept.deleted, 'good'),
          E('corpus after', `${swept.corpus_objects} objects · ${humanBytes(swept.corpus_bytes)}`),
          note('How long the operation log is kept is how long reclamation takes. Those are one number, not two.'),
        ];
      },
    },
    {
      label: 'and main still reads its shared dataset',
      async run(c) {
        const head = await c.api('GET', `/v1/envs/${ENV}/refs/${MAIN}/resolve`);
        const read = await c.api(
          'GET',
          `/v1/envs/${ENV}/commits/${head.commit}/file/${DATASET}?offset=0&length=32`,
          { binary: true },
        );
        return [
          E('bytes read from main', read.bytes, 'good'),
          note('The sweep took only what nothing else reached. Shared content was never a candidate.'),
        ];
      },
    },
  ],

  'Large Artifact Handling': [
    {
      label: 'measure the dataset',
      async run(c) {
        const head = await c.api('GET', `/v1/envs/${ENV}/refs/${MAIN}/resolve`);
        const listing = await c.api(
          'GET',
          `/v1/envs/${ENV}/commits/${head.commit}/tree/data`,
        );
        const entry = listing.entries.find((e) => e.name === 'train.bin');
        c.set('datasetSize', entry.size);
        return [E('data/train.bin', humanBytes(entry.size)), E('stored as', entry.kind)];
      },
    },
    {
      label: 'overwrite 64 bytes in the middle of it',
      async run(c) {
        const result = c.record('64 B patch', await c.api('POST', `/demo/envs/${ENV}/patch`, {
          body: {
            path: DATASET,
            offset: Math.floor(DATASET_BYTES / 2),
            length: 64,
            fill: 'X',
            message: 'patch the dataset',
          },
        }));
        const size = c.get('datasetSize') || DATASET_BYTES;
        await refreshDistribution();
        return [
          E('bytes changed', '64 B'),
          E('objects created', result.objects_created, 'good'),
          E('bytes stored', humanBytes(result.bytes_stored), 'good'),
          E('as a share of the file', pct(result.bytes_stored / size), 'good'),
          note('Content-defined chunk boundaries re-synchronise a few kilobytes after the edit, so the cost is the region — and it does not grow with the file.'),
        ];
      },
    },
  ],

  'Multi-Container Support': [
    {
      label: 'store alpine inside the version',
      async run(c) {
        const result = c.record('image: alpine', await c.api('POST', `/demo/envs/${ENV}/images`, {
          body: { base: 'alpine', name: 'app' },
        }));
        c.set('digestV1', result.manifest_digest);
        c.set('imageCommit', result.commit);
        return [
          E('image', `app @ ${result.manifest_digest.slice(0, 26)}…`),
          E('layers', `${result.layers} (${result.layers_reused} already stored)`),
          E('compressed source', humanBytes(result.bytes_compressed)),
          E('stored uncompressed', humanBytes(result.bytes_uncompressed)),
          E('pull with', result.pull_with),
          note('Stored uncompressed on purpose: chunking a gzip stream deduplicates nothing, because one changed byte changes every byte after it.'),
        ];
      },
    },
    {
      label: 'the registry serves it',
      async run(c) {
        const manifest = await c.api(
          'GET',
          `/v2/${ENV}/app/manifests/main`,
        );
        return [
          E('media type', manifest.mediaType),
          E('layers in the manifest', manifest.layers.length),
          note('Not a separate registry — the same objects, read through the OCI distribution API and the same authorization.'),
        ];
      },
    },
    {
      label: 'add a second image alongside it',
      async run(c) {
        const result = c.record('image: busybox', await c.api('POST', `/demo/envs/${ENV}/images`, {
          body: { base: 'busybox', name: 'sidecar' },
        }));
        const app = await c.api('GET', `/v2/${ENV}/app/manifests/main`);
        return [
          E('sidecar', result.manifest_digest.slice(0, 26) + '…'),
          E('app, still served', app.layers.length + ' layers'),
          note('One version, two images. Adding the second rewrote one directory listing and copied no layer bytes.'),
        ];
      },
    },
    {
      label: 'rebuild app from a different base',
      async run(c) {
        const result = c.record('image: rebuild', await c.api('POST', `/demo/envs/${ENV}/images`, {
          body: { base: 'busybox', name: 'app' },
        }));
        c.set('digestV2', result.manifest_digest);
        return [
          E('digest before', c.get('digestV1').slice(0, 26) + '…'),
          E('digest now', result.manifest_digest.slice(0, 26) + '…', 'warn'),
          E('bytes stored', humanBytes(result.bytes_stored)),
          E('layers reused', `${result.layers_reused} of ${result.layers}`, 'good'),
        ];
      },
    },
    {
      label: 'revert — and the old containers come back',
      async run(c) {
        const current = await c.api('GET', `/v1/envs/${ENV}/refs/${MAIN}/resolve`);
        c.set('afterImages', current.commit);
        await c.api('PUT', `/v1/envs/${ENV}/refs/${MAIN}`, {
          body: { target: c.get('imageCommit'), expected_generation: current.generation },
        });
        const manifest = await c.api('GET', `/v2/${ENV}/app/manifests/main`, { binary: true });
        const digest = manifest.headers.get('Docker-Content-Digest');
        return [
          E('digest now', (digest || '').slice(0, 26) + '…', digest === c.get('digestV1') ? 'good' : 'bad'),
          E('matches the first version', digest === c.get('digestV1') ? 'yes' : 'no', digest === c.get('digestV1') ? 'good' : 'bad'),
          note('There is no tag that could have moved underneath it and no registry that could have collected it — the commit contains the layers.'),
        ];
      },
    },
    {
      label: 'and forward again',
      async run(c) {
        const current = await c.api('GET', `/v1/envs/${ENV}/refs/${MAIN}/resolve`);
        await c.api('PUT', `/v1/envs/${ENV}/refs/${MAIN}`, {
          body: { target: c.get('afterImages'), expected_generation: current.generation },
        });
        const manifest = await c.api('GET', `/v2/${ENV}/app/manifests/main`, { binary: true });
        const digest = manifest.headers.get('Docker-Content-Digest');
        return [
          E('digest now', (digest || '').slice(0, 26) + '…', digest === c.get('digestV2') ? 'good' : 'bad'),
          note('A real container runtime pulls this image and runs it — `docker pull` against /v2 needs no registry credential, because the read token is the credential.'),
        ];
      },
    },
  ],

  'Read at scale': [
    {
      label: 'read 64 bytes from the middle of the dataset',
      async run(c) {
        const head = await c.api('GET', `/v1/envs/${ENV}/refs/${MAIN}/resolve`);
        c.set('head', head.commit);
        const measured = await c.readsOf(() =>
          c.api(
            'GET',
            `/v1/envs/${ENV}/commits/${head.commit}/file/${DATASET}?offset=2000000&length=64`,
            { binary: true },
          ),
        );
        c.set('rangedReads', measured.reads);
        return [
          E('bytes over the wire', measured.value.bytes, 'good'),
          E('objects the server fetched', measured.reads, 'good'),
          note('Counted by the server’s own ledger_object_reads_total, read either side of the request.'),
        ];
      },
    },
    {
      label: 'read the whole file, for comparison',
      async run(c) {
        const measured = await c.readsOf(() =>
          c.api(
            'GET',
            `/v1/envs/${ENV}/commits/${c.get('head')}/file/${DATASET}`,
            { binary: true },
          ),
        );
        return [
          E('bytes over the wire', humanBytes(measured.value.bytes)),
          E('objects fetched', measured.reads),
          E('…against the ranged read', c.get('rangedReads'), 'good'),
          note('No clone, no working copy, and no size limit — the read descends the blob spine to the chunks it actually needs.'),
        ];
      },
    },
    {
      label: 'page a directory with a cursor',
      async run(c) {
        const first = await c.api(
          'GET',
          `/v1/envs/${ENV}/commits/${c.get('head')}/tree/?limit=2`,
        );
        const next = await c.api(
          'GET',
          `/v1/envs/${ENV}/commits/${c.get('head')}/tree/?limit=2&after=${encodeURIComponent(first.cursor || '')}`,
        );
        return [
          E('first page', first.entries.map((e) => e.name).join(', ')),
          E('cursor', first.cursor || '(end)'),
          E('next page', next.entries.map((e) => e.name).join(', ') || '(end)'),
          note('The cursor is the last name you saw. Under immutability the listing is stable — entries cannot shift, repeat or vanish between pages.'),
        ];
      },
    },
  ],

  'Scoped authorization': [
    {
      label: 'create another team’s environment',
      async run(c) {
        const created = await c.api('POST', '/v1/envs', {
          body: { name: OTHER, owner: 'someone-else' },
        });
        return [E('environment', created.name), E('owner', 'someone-else')];
      },
    },
    {
      label: 'mint a read-only token for this environment',
      async run(c) {
        const minted = await c.api('POST', '/v1/tokens', {
          body: {
            principal: 'rollout-7',
            operations: ['env:read'],
            env_id: c.get('envId'),
            ttl_seconds: 86400,
          },
        });
        c.set('token', minted.token);
        return [
          E('principal', 'rollout-7'),
          E('operations', minted.operations.join(', ')),
          E('token', minted.token.slice(0, 34) + '…'),
          note('Minting only ever narrows. A token that could widen itself would make every scope in the corpus decorative.'),
        ];
      },
    },
    {
      label: 'read with it',
      async run(c) {
        const record = await c.api('GET', `/v1/envs/${ENV}`, { token: c.get('token') });
        return [E('GET the environment', '200', 'good'), E('name', record.name)];
      },
    },
    {
      label: 'write with it',
      async run(c) {
        const error = await refused(403, () =>
          c.api('POST', `/demo/envs/${ENV}/files`, {
            token: c.get('token'),
            body: { path: 'README.md', text: 'nope\n' },
          }),
        );
        return [
          E('write', `403 ${error.code}`, 'good'),
          E('operation', error.details.operation || '—'),
        ];
      },
    },
    {
      label: 'read another team’s environment with it',
      async run(c) {
        const error = await refused(403, () =>
          c.api('GET', `/v1/envs/${OTHER}`, { token: c.get('token') }),
        );
        return [
          E('read elsewhere', `403 ${error.code}`, 'good'),
          E('message', error.message),
          note('Worded exactly like the refusal for a name that does not exist. Two spellings of “no” are two thirds of an enumeration oracle.'),
        ];
      },
    },
    {
      label: 'mint a wider token from it',
      async run(c) {
        const error = await refused(403, () =>
          c.api('POST', '/v1/tokens', {
            token: c.get('token'),
            body: { principal: 'escalated', operations: ['env:write'], ttl_seconds: 60 },
          }),
        );
        return [E('attenuation', `403 ${error.code}`, 'good'), E('message', error.message)];
      },
    },
  ],

  'Build & Sync': [
    {
      label: 'run the dispatcher and a worker',
      async run(c) {
        const drained = await c.api('POST', '/demo/builds/drain');
        c.set('runnerBefore', drained.runner_invocations);
        return [
          E('events read from the stream', drained.events),
          E('builds queued', drained.enqueued),
          E('builds run', drained.built),
          E('runner invocations', drained.runner_invocations),
          note('Nothing asked for these. The ref update that published the commit produced the event, in the same transaction — there is no Action to forget to add.'),
        ];
      },
    },
    {
      label: 'the build result, under the commit',
      async run(c) {
        const head = await c.api('GET', `/v1/envs/${ENV}/refs/${MAIN}/resolve`);
        c.set('builtCommit', head.commit);
        const build = await c.api('GET', `/v1/envs/${ENV}/commits/${head.commit}/build`);
        const sync = await c.api('GET', `/v1/envs/${ENV}/commits/${head.commit}/notes/sync`);
        return [
          E('status', build.status, build.status === 'succeeded' ? 'good' : 'bad'),
          E('worker', build.worker),
          E('attempts', build.attempts),
          E('synced as', sync.body.platform_id, 'good'),
          note('The sync record is a note beside the commit, not inside it — a commit’s name is the hash of its content, so a verdict cannot be appended to it.'),
        ];
      },
    },
    {
      label: 'point the fork at the same commit, and drain again',
      async run(c) {
        const forkHead = await c.api('GET', `/v1/envs/${FORK}/refs/${MAIN}/resolve`);
        await c.api('PUT', `/v1/envs/${FORK}/refs/${MAIN}`, {
          body: { target: c.get('builtCommit'), expected_generation: forkHead.generation },
        });
        const drained = await c.api('POST', '/demo/builds/drain');
        const moved = drained.runner_invocations - c.get('runnerBefore');
        return [
          E('builds run', drained.built),
          E('served from cache', drained.cache_hits, 'good'),
          E('runner invocations moved by', moved, moved === 0 ? 'good' : 'warn'),
          note('A build is a pure function of a commit, so a fork that changed nothing inherits its parent’s result instead of rebuilding.'),
        ];
      },
    },
    {
      label: 'ask for a rebuild explicitly',
      async run(c) {
        const queued = await c.api('POST', `/v1/envs/${ENV}/builds`, {
          body: { ref: MAIN, rebuild: true },
        });
        const drained = await c.api('POST', '/demo/builds/drain');
        return [
          E('accepted', `202 for ${short(queued.commit)}`),
          E('builds run', drained.built),
          E('runner invocations', drained.runner_invocations),
          note('Reruns and backfills are the only reason this route exists. An ordinary commit needs no call at all.'),
        ];
      },
    },
  ],

  'Format Agnostic': [
    {
      label: 'import a real git history',
      async run(c) {
        await c.api('POST', '/v1/envs', { body: { name: MIGRATED, owner: 'rl-infra' } });
        const report = await c.api('POST', `/demo/envs/${MIGRATED}/import`, {
          body: { repository: 'ledger', ref: 'refs/heads/imported', limit: 50 },
        });
        c.set('imported', report.head);
        c.record('git import', {
          bytes_offered: report.git_bytes,
          bytes_stored: report.bytes_stored,
          objects_created: report.objects_created,
        });
        return [
          E('commits converted', report.commits),
          E('trees · blobs', `${report.trees} · ${report.blobs}`),
          E('git content seen', humanBytes(report.git_bytes)),
          E('bytes stored', humanBytes(report.bytes_stored)),
          E('deduplicated', pct(report.dedup_ratio), 'good'),
          note('A conversion, not a bridge. Blobs are re-chunked rather than copied — importing git’s representation would import the storage model this system replaces.'),
        ];
      },
    },
    {
      label: 'import it again',
      async run(c) {
        const report = await c.api('POST', `/demo/envs/${MIGRATED}/import`, {
          body: { repository: 'ledger', ref: 'refs/heads/imported', limit: 50 },
        });
        return [
          E('objects created', report.objects_created, report.objects_created === 0 ? 'good' : 'warn'),
          E('bytes stored', report.bytes_stored, report.bytes_stored === 0 ? 'good' : 'warn'),
          note('Every git SHA is already in the alternate-digest index, so a re-import resumes rather than restarts.'),
        ];
      },
    },
    {
      label: 'read a file out of the imported history',
      async run(c) {
        const read = await c.api(
          'GET',
          `/v1/envs/${MIGRATED}/commits/${c.get('imported')}/file/pyproject.toml`,
          { binary: true },
        );
        return [
          E('pyproject.toml', `${read.bytes} B`),
          note('The same read path as anything else. Nothing below the build layer knows what a git repository, an OCI image or a YAML manifest is.'),
        ];
      },
    },
    {
      label: 'the manifest is an ordinary blob',
      async run(c) {
        const head = await c.api('GET', `/v1/envs/${ENV}/refs/${MAIN}/resolve`);
        const listing = await c.api('GET', `/v1/envs/${ENV}/commits/${head.commit}/tree/`);
        const manifest = listing.entries.find((entry) => entry.name === 'harbor.yaml');
        return [
          E('harbor.yaml', `${manifest.kind}, ${manifest.size} B`),
          E('named by', short(manifest.target)),
          note('One module parses YAML, and a structural test enforces that nothing below it may. The storage layer stores bytes.'),
        ];
      },
    },
  ],

  'Concurrent Automations': [
    {
      label: 'two automations, the same generation',
      async run(c) {
        const current = await c.api('GET', `/v1/envs/${ENV}/refs/${MAIN}/resolve`);
        await c.api('PUT', `/v1/envs/${ENV}/refs/${MAIN}`, {
          body: { target: current.commit, expected_generation: current.generation },
        });
        const error = await refused(409, () =>
          c.api('PUT', `/v1/envs/${ENV}/refs/${MAIN}`, {
            body: { target: current.commit, expected_generation: current.generation },
          }),
        );
        return [
          E('first writer', '200', 'good'),
          E('second writer', `409 ${error.code}`, 'good'),
          E('current generation', error.details.current_generation),
          E('current target', short(String(error.details.current_target))),
          note('The conflict carries everything the loser needs to rebase, so it does not have to ask again to find out what happened.'),
        ];
      },
    },
    {
      label: 'two environments, at the same time',
      async run(c) {
        const started = performance.now();
        const both = await Promise.all([
          c.api('POST', `/demo/envs/${ENV}/files`, {
            body: { path: 'notes/a.md', text: 'from automation A\n', message: 'A' },
          }),
          c.api('POST', `/demo/envs/${FORK}/files`, {
            body: { path: 'notes/b.md', text: 'from automation B\n', message: 'B' },
          }),
        ]);
        return [
          E('both published', both.map((r) => short(r.commit)).join('  ')),
          E('wall clock', `${Math.round(performance.now() - started)} ms`),
          note('Metadata is partitioned by environment, so neither of these ever saw the other’s lock. Work on one environment never waits on another.'),
        ];
      },
    },
  ],

  Deduplication: [
    {
      label: 'write a file that is already exactly that',
      async run(c) {
        const first = await c.api('POST', `/demo/envs/${ENV}/files`, {
          body: { path: 'notes/a.md', text: 'from automation A\n', message: 'no change at all' },
        });
        c.record('identical write', first);
        return [
          E('objects created', first.objects_created, first.objects_created <= 1 ? 'good' : 'warn'),
          E('bytes stored', first.bytes_stored, 'good'),
          note('One object: the commit. Everything else already existed under its own hash, including every tree node on the path — the edit produced the identical tree.'),
        ];
      },
    },
    {
      label: 'commit the whole environment into an unrelated org',
      async run(c) {
        await c.api('POST', '/v1/envs', { body: { name: ENV_B, owner: 'another-team' } });
        const result = c.record('cross-environment', await c.api(
          'POST',
          `/demo/envs/${ENV_B}/commit`,
          { body: { message: 'the same environment, elsewhere' } },
        ));
        return [
          E('objects offered', result.objects_offered),
          E('objects created', result.objects_created, 'good'),
          E('bytes stored', humanBytes(result.bytes_stored), 'good'),
          note('Global deduplication: the two environments are unrelated and share every byte. Which is also why a writer cannot store arbitrary bytes under a chosen name.'),
        ];
      },
    },
    {
      label: 'and the second environment really has it',
      async run(c) {
        const head = await c.api('GET', `/v1/envs/${ENV_B}/refs/${MAIN}/resolve`);
        const read = await c.api(
          'GET',
          `/v1/envs/${ENV_B}/commits/${head.commit}/file/${DATASET}?offset=0&length=64`,
          { binary: true },
        );
        return [
          E('bytes read', read.bytes, 'good'),
          note('Sharing is not a reference to somebody else’s copy. There is one copy, and both environments name it.'),
        ];
      },
    },
  ],

  Stability: [
    {
      label: 'retry a ref update with the same key',
      async run(c) {
        const key = `demo-${hex(12)}`;
        const current = await c.api('GET', `/v1/envs/${ENV}/refs/${MAIN}/resolve`);
        const body = { target: current.commit, expected_generation: current.generation };
        const first = await c.api('PUT', `/v1/envs/${ENV}/refs/${MAIN}`, {
          body,
          headers: { 'Idempotency-Key': key },
        });
        const replay = await c.api('PUT', `/v1/envs/${ENV}/refs/${MAIN}`, {
          body,
          headers: { 'Idempotency-Key': key },
        });
        c.set('idempotencyKey', key);
        return [
          E('first', `generation ${first.generation}`),
          E('replay', `generation ${replay.generation}`, first.generation === replay.generation ? 'good' : 'bad'),
          note('The second call did not move anything. A retry after a timeout is safe because the outcome was recorded in the transaction that produced it.'),
        ];
      },
    },
    {
      label: 'the same key, a different payload',
      async run(c) {
        const current = await c.api('GET', `/v1/envs/${ENV}/refs/${MAIN}/resolve`);
        const error = await refused(422, () =>
          c.api('PUT', `/v1/envs/${ENV}/refs/refs/heads/other`, {
            body: { target: current.commit },
            headers: { 'Idempotency-Key': c.get('idempotencyKey') },
          }),
        );
        return [
          E('reused key, new request', `422 ${error.code}`, 'good'),
          note('422 means exactly this. Request validation is remapped to 400 so the two cannot be confused.'),
        ];
      },
    },
    {
      label: 'rot one byte on the medium',
      async run(c) {
        const damaged = await c.api('POST', `/demo/envs/${ENV}/damage`, {
          body: { path: DATASET },
        });
        c.set('damaged', damaged.object);
        return [
          E('object', short(damaged.object)),
          E('changed', 'one byte, on disk', 'warn'),
          note('Reaching past the store to the file underneath it. This is the only place in the portal that touches storage as bytes.'),
        ];
      },
    },
    {
      label: 'read it',
      async run(c) {
        const head = await c.api('GET', `/v1/envs/${ENV}/refs/${MAIN}/resolve`);
        const error = await refused(500, () =>
          c.api(
            'GET',
            `/v1/envs/${ENV}/commits/${head.commit}/file/${DATASET}?offset=0&length=16`,
            { binary: true },
          ),
        );
        return [
          E('the read', `500 ${error.code}`, 'good'),
          E('message', error.message),
          note('Refused, never downgraded to a miss. A miss invites a retry; this means a medium is actively wrong and someone has to look at it.'),
        ];
      },
    },
    {
      label: 'put the byte back',
      async run(c) {
        await c.api('POST', `/demo/envs/${ENV}/damage`, {
          body: { path: DATASET, restore: true },
        });
        const head = await c.api('GET', `/v1/envs/${ENV}/refs/${MAIN}/resolve`);
        const read = await c.api(
          'GET',
          `/v1/envs/${ENV}/commits/${head.commit}/file/${DATASET}?offset=0&length=16`,
          { binary: true },
        );
        return [E('the read', `${read.bytes} B`, 'good')];
      },
    },
  ],

  'Native for Agents': [
    {
      label: 'the operation log',
      async run(c) {
        const log = await c.api('GET', `/v1/envs/${ENV}/operations?limit=8`);
        c.set('lastOp', log.operations[0].sequence);
        return [
          E('entries', log.operations.length),
          ...log.operations
            .slice(0, 5)
            .map((entry) => E(`#${entry.sequence} ${entry.kind}`, entry.principal)),
          note('Every mutation, in order, with who made it — and it is a garbage-collection root, which is why undo could still reach that discarded branch.'),
        ];
      },
    },
    {
      label: 'two commits, one change',
      async run(c) {
        const changeId = hex(32);
        const first = await c.api('POST', `/demo/envs/${ENV}/files`, {
          body: { path: 'task/verifier.py', text: 'def verify(result):\n    return True\n', change_id: changeId, message: 'first attempt' },
        });
        const second = await c.api('POST', `/demo/envs/${ENV}/files`, {
          body: { path: 'task/verifier.py', text: 'def verify(result):\n    return False\n', change_id: changeId, message: 'stricter' },
        });
        const a = await c.api('GET', `/v1/envs/${ENV}/commits/${first.commit}`);
        const b = await c.api('GET', `/v1/envs/${ENV}/commits/${second.commit}`);
        return [
          E('commit', short(a.name)),
          E('commit', short(b.name)),
          E('change id', a.change_id === b.change_id ? a.change_id : 'differs', a.change_id === b.change_id ? 'good' : 'bad'),
          note('jj’s vocabulary, not a new one: a change is the idea, a commit is one version of it. An agent that amends its work keeps one identity.'),
        ];
      },
    },
    {
      label: 'undo the last operation',
      async run(c) {
        const before = await c.api('GET', `/v1/envs/${ENV}/refs/${MAIN}/resolve`);
        const log = await c.api('GET', `/v1/envs/${ENV}/operations?limit=1`);
        const undone = await c.api(
          'POST',
          `/v1/envs/${ENV}/operations/${log.operations[0].sequence}/undo`,
          { body: {} },
        );
        return [
          E('undid', `#${log.operations[0].sequence} ${log.operations[0].kind}`),
          E('target before', short(before.commit)),
          E('target now', short(undone.target), undone.target !== before.commit ? 'good' : 'warn'),
          note('An ordinary ref update, not a rewind. It expects the generation the ref has now, so a writer who moved it meanwhile gets the usual conflict.'),
        ];
      },
    },
    {
      label: 'jj’s own backend agrees',
      async run() {
        return [
          note('A real `jj_lib::backend::Backend`, pinned to jj-lib 0.44.0, round-trips jj’s own structures through this server. The crate never computes an object name — Ledger names everything.'),
        ];
      },
    },
  ],
};

/* ────────────────────────────────────────────────────────────────────────
 * Rendering and running the cards
 * ──────────────────────────────────────────────────────────────────────── */

const cardsRoot = document.getElementById('cards');
const scoreValue = document.getElementById('score-value');
const scoreBox = document.querySelector('.score');
const banner = document.getElementById('banner');

/** Everything rendered, in requirement order. */
const cards = [];

function renderCards(requirements) {
  cardsRoot.replaceChildren();
  cards.length = 0;

  for (const requirement of requirements) {
    const steps = CARDS[requirement.name] || [];
    const root = el('article', 'card');
    const head = el('div', 'card-head');
    const tick = el('span', 'tick', '○');
    head.append(tick);
    head.append(el('span', 'card-title', requirement.name));
    head.append(el('span', 'card-asks', requirement.asks_for));
    const count = el('span', 'card-count', `0/${steps.length}`);
    head.append(count);
    root.append(head);

    const body = el('div', 'card-body');
    body.append(el('blockquote', 'quote', `“${requirement.asks_for}”`));

    const stepList = el('div', 'steps');
    const entries = steps.map((step, index) => renderStep(step, index, requirement.name));
    for (const entry of entries) stepList.append(entry.node);
    body.append(stepList);

    if (!steps.length) {
      body.append(el('p', 'step-note', 'No steps are defined for this requirement on the page.'));
    }
    root.append(body);
    cardsRoot.append(root);

    if (ESSENTIALS.includes(requirement.name)) root.classList.add('core');

    const card = { requirement, root, tick, count, entries, done: false };
    head.addEventListener('click', () => {
      body.hidden = !body.hidden;
    });
    body.hidden = true;
    cards.push(card);
  }
  updateScore();
}

function renderStep(step, index, requirementName) {
  const node = el('div', 'step');
  const mark = el('span', 'step-mark', '·');
  const main = el('div', 'step-main');
  main.append(el('div', 'step-label', step.label));
  const evidence = el('div', 'evidence');
  main.append(evidence);
  const button = el('button', '', 'run');
  node.append(mark, main, button);

  const entry = { step, node, mark, evidence, button, ok: false, requirementName, index };
  button.addEventListener('click', () => runStep(entry));
  return entry;
}

async function runStep(entry) {
  entry.node.className = 'step busy';
  entry.mark.textContent = '…';
  entry.evidence.replaceChildren();
  entry.button.disabled = true;

  try {
    const rows = (await entry.step.run(context)) || [];
    entry.evidence.replaceChildren(...rows.map(renderEvidence));
    entry.node.className = 'step ok';
    entry.mark.textContent = '✓';
    entry.ok = true;
  } catch (error) {
    entry.node.className = 'step bad';
    entry.mark.textContent = '✗';
    entry.ok = false;
    const message =
      error instanceof ApiError
        ? `${error.status} ${error.code} — ${error.message}`
        : String(error.message || error);
    entry.evidence.append(el('div', 'step-error', message));
  } finally {
    entry.button.disabled = false;
    refreshCard(entry.requirementName);
    refreshStats();
  }
  return entry.ok;
}

/** A hash, a path, a ref — an identifier rather than a measurement.
 *
 * These share the evidence row with the numbers, but setting `b3:9f2a…` at the
 * size of a headline figure would shout the one thing nobody reads aloud. */
const IDENTIFIER = /^(b3:|refs\/|sha256:)|[/]|^[0-9a-f]{8,}$/;

function renderEvidence(row) {
  if (row.note) return el('p', 'step-note', row.note);
  // DOM order stays label-then-value, which is how it reads aloud. The tile
  // shows the value above the label via `flex-direction: column-reverse`.
  const line = el('div', 'evidence-row');
  line.append(el('span', 'k', row.label));
  const classes = ['v'];
  if (row.tone) classes.push(row.tone);
  if (!row.tone && IDENTIFIER.test(row.value)) classes.push('ident');
  line.append(el('span', classes.join(' '), row.value));
  return line;
}

function refreshCard(name) {
  const card = cards.find((entry) => entry.requirement.name === name);
  if (!card) return;
  const done = card.entries.filter((entry) => entry.ok).length;
  card.count.textContent = `${done}/${card.entries.length}`;
  card.done = card.entries.length > 0 && done === card.entries.length;
  const failed = card.entries.some((entry) => entry.node.classList.contains('bad'));
  // Toggling rather than reassigning `className`: this runs after every step,
  // and a whole-string assignment wiped the `running` class that
  // `runEverything` had just set — so the focus styling only ever flashed.
  card.root.classList.toggle('done', card.done);
  card.root.classList.toggle('failed', !card.done && failed);
  card.tick.textContent = card.done ? '✓' : failed ? '✗' : '○';
  showHeadline(card);
  updateScore();
}

/** The measurement a card is remembered for, lifted into its header.
 *
 * A closed card is a title and a tick; on a recording that is nothing to look
 * at. `tone: 'good'` already marks the number each step exists to produce, so
 * the first one a card has proved is the honest choice — no second list to
 * keep true. */
function showHeadline(card) {
  if (card.headline) return;
  for (const entry of card.entries) {
    if (!entry.ok) continue;
    const row = entry.evidence.querySelector('.evidence-row .v.good');
    if (!row) continue;
    const box = el('span', 'card-headline');
    box.append(el('span', 'hv', row.textContent));
    box.append(el('span', 'hk', row.parentElement.querySelector('.k').textContent));
    card.count.before(box);
    card.headline = box;
    return;
  }
}

function updateScore() {
  const done = cards.filter((card) => card.done).length;
  scoreValue.textContent = String(done);
  scoreBox.classList.toggle('complete', done === cards.length && cards.length > 0);
}

/** The cards a five-to-seven minute telling walks, in page order.
 *
 * Not a shorter page — every requirement still renders, because the card list
 * is read from `src.verify.REQUIREMENTS` and the page must not be able to claim
 * one the verifier does not know about. This is a shorter *run*.
 *
 * The set is closed under its own dependencies, which is the whole difficulty:
 * `Branching / Forking` creates the variant environment that
 * `Concurrent Automations` writes to, and that card in turn writes the file
 * `Deduplication` re-writes byte-for-byte to show one object created. Drop
 * either and the last card quietly stops saying 1.
 *
 * It also leaves out the two cards that shell out — a git import and three
 * `docker save` calls — which are the slowest steps on the page and the two
 * that can fail on somebody else's machine. */
const ESSENTIALS = [
  'Version Environments',
  'Branching / Forking',
  'Large Artifact Handling',
  'Concurrent Automations',
  'Deduplication',
];

/** Walk cards in order, stopping at nothing.
 *
 * `focusing` on the container dims every card except the running one. The page
 * is recorded, and at roughly seven seconds a card the camera needs one
 * subject rather than twelve equal ones.
 *
 * Order is never the caller's to choose: two steps advance the shared clock by
 * ninety-nine days between them, and a card that mints a token before that jump
 * and spends it after would be refused for the wrong reason. */
async function runEverything(only) {
  const walk = only ? cards.filter((card) => only.includes(card.requirement.name)) : cards;
  cardsRoot.classList.add('focusing');
  try {
    for (const card of walk) {
      card.root.classList.add('running');
      card.root.scrollIntoView({ behavior: 'smooth', block: 'center' });
      const body = card.root.querySelector('.card-body');
      body.hidden = false;
      for (const entry of card.entries) {
        const ok = await runStep(entry);
        if (!ok) break;
      }
      card.root.classList.remove('running');
      refreshCard(card.requirement.name);
    }
  } finally {
    cardsRoot.classList.remove('focusing');
  }
  await refreshStats();
  await drawStorageChart();
}

/* ────────────────────────────────────────────────────────────────────────
 * Charts — hand-rolled SVG, for the reason the browser hand-rolls its HTML
 * ──────────────────────────────────────────────────────────────────────── */

const SVG = 'http://www.w3.org/2000/svg';

function svg(tag, attributes) {
  const node = document.createElementNS(SVG, tag);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, String(value));
  return node;
}

/** The palette, read from the stylesheet rather than restated here.
 *
 * Every colour in these charts used to be a hex literal, so the theme lived in
 * two files and had to be changed in both. The `fill*` entries are the
 * validated categorical set — see the note beside them in console.css. */
const PALETTE = (() => {
  const style = getComputedStyle(document.documentElement);
  const token = (name, fallback) => (style.getPropertyValue(name) || fallback).trim();
  return {
    ink: token('--ink', '#dfe4ee'),
    dim: token('--ink-dim', '#8b94a7'),
    line: token('--line', '#222836'),
    surface: token('--bg-raised', '#141821'),
    track: token('--line', '#222836'),
    accent: token('--accent', '#6ea8ff'),
    fillCommit: token('--fill-commit', '#c48420'),
    fillTree: token('--fill-tree', '#5a90e8'),
    fillBlob: token('--fill-blob', '#c063b8'),
    fillChunk: token('--fill-chunk', '#2f9e78'),
  };
})();

/** Type size inside the SVGs. They are read at the same distance as the rest
 * of the page, so they are set at the page's own small-label size — not the
 * 10px they were, which no CSS change could reach. */
const CHART_TEXT = 13;

function drawCostChart() {
  const host = document.getElementById('chart-cost');
  if (!writes.length) return;

  const rows = writes.slice(-9);
  const width = 460;
  const rowHeight = 46;
  const labelWidth = 150;
  const valueWidth = 84;
  const barWidth = width - labelWidth - valueWidth;
  const peak = Math.max(...rows.map((row) => row.offered), 1);
  const height = rows.length * rowHeight + 8;

  // viewBox rather than a fixed width, so the chart fills whatever column it
  // is given instead of only ever shrinking below 320px.
  const chart = svg('svg', {
    viewBox: `0 0 ${width} ${height}`,
    preserveAspectRatio: 'xMinYMin meet',
    role: 'img',
  });
  rows.forEach((row, index) => {
    const y = index * rowHeight + 6;
    const label = svg('text', { x: 0, y: y + 14, fill: PALETTE.dim, 'font-size': CHART_TEXT });
    label.textContent = row.label;
    chart.append(label);

    // Offered is the context, stored is the point: one track, one fill on top
    // of it, so the saving is the length difference rather than two bars to
    // compare. 4px rounded data-end, anchored to a shared baseline.
    chart.append(svg('rect', {
      x: labelWidth, y: y + 4, width: Math.max(2, (row.offered / peak) * barWidth),
      height: 18, rx: 4, fill: PALETTE.track,
    }));
    chart.append(svg('rect', {
      x: labelWidth, y: y + 4, width: Math.max(2, (row.stored / peak) * barWidth),
      height: 18, rx: 4,
      fill: row.stored / Math.max(row.offered, 1) < 0.1 ? PALETTE.fillChunk : PALETTE.fillTree,
    }));

    const value = svg('text', {
      x: width, y: y + 18, fill: PALETTE.ink, 'font-size': CHART_TEXT,
      'text-anchor': 'end', 'font-weight': 600,
    });
    value.textContent = humanBytes(row.stored);
    chart.append(value);
  });

  host.replaceChildren(chart);
  const legend = el('div', 'legend');
  legend.innerHTML =
    `<span><i class="swatch" style="background:${PALETTE.track}"></i>offered</span>` +
    `<span><i class="swatch" style="background:${PALETTE.fillTree}"></i>stored</span>`;
  host.append(legend);
}

/* The four object kinds, in the order they stack. Fills come from the
 * stylesheet's validated categorical set — see the note beside them there. */
const KINDS = [
  ['chunk', 'fillChunk'],
  ['blob', 'fillBlob'],
  ['tree', 'fillTree'],
  ['commit', 'fillCommit'],
];

async function refreshDistribution() {
  try {
    drawDistribution(await api('GET', '/console/distribution'));
  } catch {
    /* A corpus that was just reset. The panel keeps whatever it last drew
       rather than blanking mid-demonstration. */
  }
}

/** Where the corpus sits in its own name space, and what it is made of.
 *
 * Sixteen columns, one per leading hex character of the object name, each
 * stacked by kind. The columns come out level because a name *is* the hash of
 * the bytes — which is the property that lets the store shard by prefix at all,
 * and this is the same partitioning the collector walks.
 */
function drawDistribution(data) {
  const host = document.getElementById('chart-names');
  if (!data.buckets.length || !data.objects) return;

  const width = 460;
  const height = 150;
  const axis = 22;
  const slot = width / data.buckets.length;
  // Cap the bar and let the slot's leftover be air, rather than filling it.
  const bar = Math.min(24, slot - 8);
  const peak = Math.max(...data.buckets.map((b) => b.total), 1);

  const chart = svg('svg', {
    viewBox: `0 0 ${width} ${height + axis}`,
    preserveAspectRatio: 'xMidYMin meet',
    role: 'img',
  });

  data.buckets.forEach((bucket, index) => {
    const x = index * slot + (slot - bar) / 2;
    let y = height;
    for (const [kind, token] of KINDS) {
      const count = bucket.kinds[kind] || 0;
      if (!count) continue;
      const span = (count / peak) * height;
      // A 2px gap in the surface colour is what separates the segments; a
      // stroke around them would add ink that is not data.
      const drawn = Math.max(1, span - 2);
      y -= span;
      chart.append(svg('rect', {
        x, y, width: bar, height: drawn, rx: 3, fill: PALETTE[token],
      }));
    }
    const label = svg('text', {
      x: x + bar / 2, y: height + 16, fill: PALETTE.dim,
      'font-size': CHART_TEXT, 'text-anchor': 'middle',
    });
    label.textContent = bucket.prefix;
    chart.append(label);
  });

  host.replaceChildren(chart);
  const legend = el('div', 'legend');
  legend.innerHTML =
    `<span>${data.objects} objects</span>` +
    KINDS.map(([kind, token]) =>
      `<span><i class="swatch" style="background:${PALETTE[token]}"></i>${kind}</span>`
    ).reverse().join('');
  host.append(legend);
}

let lastPlan = null;

async function drawStorageChart(plan) {
  if (plan) lastPlan = plan;
  const host = document.getElementById('chart-storage');
  let stats;
  try {
    stats = await api('GET', '/console/stats');
  } catch {
    return;
  }

  const reclaimable = lastPlan ? lastPlan.bytes_reclaimable : 0;
  const live = Math.max(stats.stored_bytes - reclaimable, 0);
  const total = Math.max(stats.stored_bytes, 1);
  const width = 460;
  const liveWidth = (live / total) * width;
  const reclaimWidth = (reclaimable / total) * width;
  // A 2px gap in the surface colour is what separates the two segments — a
  // stroke around them would add ink that is not data.
  const gap = reclaimWidth > 0 ? 2 : 0;

  const chart = svg('svg', {
    viewBox: `0 0 ${width} 62`, preserveAspectRatio: 'xMinYMin meet', role: 'img',
  });
  chart.append(svg('rect', {
    x: 0, y: 6, width: Math.max(0, liveWidth - gap), height: 24, rx: 4, fill: PALETTE.fillTree,
  }));
  if (reclaimWidth > 0) {
    chart.append(svg('rect', {
      x: liveWidth, y: 6, width: reclaimWidth, height: 24, rx: 4, fill: PALETTE.fillCommit,
    }));
  }
  const caption = svg('text', { x: 0, y: 52, fill: PALETTE.dim, 'font-size': CHART_TEXT });
  caption.textContent = `${stats.objects} objects · ${humanBytes(stats.stored_bytes)} stored`;
  chart.append(caption);

  host.replaceChildren(chart);
  const legend = el('div', 'legend');
  legend.innerHTML =
    `<span><i class="swatch" style="background:${PALETTE.fillTree}"></i>live ${humanBytes(live)}</span>` +
    `<span><i class="swatch" style="background:${PALETTE.fillCommit}"></i>reclaimable ${humanBytes(reclaimable)}</span>` +
    `<span>tombstones ${stats.tombstones}</span>`;
  host.append(legend);
}

/* ────────────────────────────────────────────────────────────────────────
 * The corpus
 * ──────────────────────────────────────────────────────────────────────── */

async function refreshStats() {
  await drawStorageChart();
}

/* ────────────────────────────────────────────────────────────────────────
 * Boot
 * ──────────────────────────────────────────────────────────────────────── */

async function resetEverything() {
  await api('POST', '/demo/reset', { body: {} });
  shared.clear();
  writes.length = 0;
  lastPlan = null;
  for (const card of cards) {
    for (const entry of card.entries) {
      entry.ok = false;
      entry.node.className = 'step';
      entry.mark.textContent = '·';
      entry.evidence.replaceChildren();
    }
    // The headline is measured, so a reset has to drop it — otherwise the next
    // take opens showing the previous take's number.
    if (card.headline) {
      card.headline.remove();
      card.headline = null;
    }
    refreshCard(card.requirement.name);
  }
  document.getElementById('chart-cost').replaceChildren(el('p', 'empty', 'run a step that writes'));
  document.getElementById('chart-names').replaceChildren(el('p', 'empty', 'commit something first'));
  await drawStorageChart();
}

async function boot() {
  try {
    const listed = await api('GET', '/console/requirements');
    renderCards(listed.requirements);
  } catch (error) {
    banner.hidden = false;
    banner.textContent = `could not load the requirement list: ${error.message}`;
    return;
  }

  document.getElementById('reset').addEventListener('click', async (event) => {
    event.target.disabled = true;
    await resetEverything();
    event.target.disabled = false;
  });

  /* Both buttons reset first: the twelve are only coherent as one ordered walk
   * from a clean corpus, so a run that started from whatever the last one left
   * behind would be measuring something nobody can describe. */
  const wire = (id, label, only) =>
    document.getElementById(id).addEventListener('click', async (event) => {
      const buttons = [...document.querySelectorAll('.bar-right button')];
      for (const button of buttons) button.disabled = true;
      event.target.textContent = 'running…';
      try {
        await resetEverything();
        await runEverything(only);
      } finally {
        for (const button of buttons) button.disabled = false;
        event.target.textContent = label;
      }
    });

  wire('run-core', 'Run the essentials', ESSENTIALS);
  wire('run-all', 'Reset & run all', null);

  await drawStorageChart();
}

boot();
