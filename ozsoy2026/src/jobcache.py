"""Resumable Pool.map for the long campaigns.

Pool.map keeps every result in memory until the last job ends, so a campaign
cut short (the cloud container is sometimes restarted) loses all its work.
cached_map appends each finished job at once to a cache file under
results/cache/ (not tracked by git), keyed by the job's arguments; a rerun of
the same campaign skips the jobs already there and returns the results in job
order, exactly as pool.map would.

The cache file name holds the campaign tag, the function name, the hash of
the simulator source (tdsim.SOURCE_SHA) and a hash of the protocol constants
the caller passes (protocol(...)), so a change in either starts a new cache.
A change elsewhere (e.g. the chatter classifier) does not: delete
results/cache/ by hand after such a change.
"""
import hashlib
import os
import pickle

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "..", "results", "cache")


def protocol(*values):
    """Short hash of the constants that define a campaign's protocol."""
    return hashlib.sha1(repr(values).encode()).hexdigest()[:10]


class _Call:
    """Picklable wrapper that returns the job index with the result."""

    def __init__(self, fn):
        self.fn = fn

    def __call__(self, item):
        i, job = item
        return i, self.fn(job)


def _load(path):
    """Records of an existing cache; a record cut short by a restart is
    dropped and the file truncated after the last complete one."""
    done, good = {}, 0
    if not os.path.exists(path):
        return done
    with open(path, "rb") as fh:
        while True:
            try:
                k, r = pickle.load(fh)
            except Exception:          # EOF, or a record cut short
                break
            done[k] = r
            good = fh.tell()
    if os.path.getsize(path) > good:
        with open(path, "r+b") as fh:
            fh.truncate(good)
    return done


def cached_map(pool, fn, jobs, tag, chunksize=1):
    """pool.map(fn, jobs), resumable.  jobs must have a stable repr()."""
    from tdsim import SOURCE_SHA
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"{tag}_{fn.__name__}_{SOURCE_SHA[:12]}.pkl")
    keys = [hashlib.sha1(repr(j).encode()).hexdigest() for j in jobs]
    done = _load(path)
    todo = [(i, jobs[i]) for i, k in enumerate(keys) if k not in done]
    if len(todo) < len(jobs):
        print(f"[jobcache] {os.path.basename(path)}: {len(jobs) - len(todo)} of {len(jobs)} jobs from the cache",
              flush=True)
    if todo:
        with open(path, "ab") as fh:
            for i, r in pool.imap_unordered(_Call(fn), todo, chunksize):
                done[keys[i]] = r
                pickle.dump((keys[i], r), fh)
                fh.flush()
    return [done[k] for k in keys]
