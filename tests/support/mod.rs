//! Shared test doubles for memory tests.
//!
//! Included by several test binaries, each of which uses a subset; unused
//! helpers would otherwise warn per-crate.
#![allow(dead_code)]
//!
//! Both the integration tests and the eval harness need a `MemoryBackend`
//! that is deterministic, in-process, and controllable — no SQLite, no ONNX
//! runtime, no network. `FakeMemoryBackend` is that. It also records its
//! operations so tests can assert *what happened to the store*, not just what
//! the loop returned.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use ::agent::memory_integration::{
    effective_namespace, MemoryBackend, MemoryHit, RecallResult, RememberResult,
};

// ---------------------------------------------------------------------------
// Fake store
// ---------------------------------------------------------------------------

#[derive(Debug, Default)]
struct FakeStore {
    rows: Vec<MemoryHit>,
    next_id: u64,
}

/// A `MemoryBackend` over an in-process `Vec`.
///
/// Similarity is deterministic word-overlap scoring so that a query's relevance
/// to a row is stable across runs, platforms and optimizations — the same
/// property a real embedder is trusted to provide but never testably.
pub struct FakeMemoryBackend {
    store: Mutex<FakeStore>,
    /// Per-operation latency (sleep), applied before the store is touched.
    /// Exposes slow/unavailable backends deterministically.
    pub latency: Mutex<Duration>,
    /// When set, every recall and remember fails.
    pub broken: Mutex<bool>,
    /// Requests dispatched. Read with [`FakeMemoryBackend::dispatched`].
    dispatched: AtomicUsize,
}

impl Default for FakeMemoryBackend {
    fn default() -> Self {
        Self {
            store: Mutex::new(FakeStore::default()),
            latency: Mutex::new(Duration::ZERO),
            broken: Mutex::new(false),
            dispatched: AtomicUsize::new(0),
        }
    }
}

impl FakeMemoryBackend {
    pub fn new() -> Self {
        Self::default()
    }

    /// Convenience constructor for use with `Arc<dyn MemoryBackend>`.
    pub fn shared() -> Arc<Self> {
        Arc::new(Self::new())
    }

    /// Number of backend calls made so far (recall + remember).
    pub fn dispatched(&self) -> usize {
        self.dispatched.load(Ordering::Relaxed)
    }

    /// Insert a row directly, bypassing `remember`.
    pub fn seed(&self, id: &str, content: &str, score_hint: f32, namespace: Option<&str>) {
        let mut store = self.store.lock().unwrap();
        let hit = MemoryHit {
            id: id.to_string(),
            content: content.to_string(),
            score: score_hint,
            namespace: effective_namespace(namespace),
            memory_type: "fact".into(),
            source: Some("agent".into()),
            source_type: "seed".into(),
            tags: vec![],
            created_at: "2026-01-01T00:00:00Z".into(),
            deprecated: false,
        };
        store.rows.push(hit);
    }

    /// Rows that are still live (not deprecated).
    pub fn live_rows(&self) -> Vec<MemoryHit> {
        self.store
            .lock()
            .unwrap()
            .rows
            .iter()
            .filter(|h| !h.deprecated)
            .cloned()
            .collect()
    }

    /// All rows, including deprecated ones.
    pub fn all_rows(&self) -> Vec<MemoryHit> {
        self.store.lock().unwrap().rows.clone()
    }

    pub fn deprecate(&self, id: &str) {
        if let Some(r) = self
            .store
            .lock()
            .unwrap()
            .rows
            .iter_mut()
            .find(|h| h.id == id)
        {
            r.deprecated = true;
        }
    }

    pub fn set_broken(&self, broken: bool) {
        *self.broken.lock().unwrap() = broken;
    }

    pub fn set_latency(&self, latency: Duration) {
        *self.latency.lock().unwrap() = latency;
    }

    fn is_broken(&self) -> bool {
        *self.broken.lock().unwrap()
    }

    fn sleep(&self) {
        let d = *self.latency.lock().unwrap();
        if !d.is_zero() {
            std::thread::sleep(d);
        }
    }
}

/// Deterministic lexical similarity in `[0, 1]`.
///
/// Overlap coefficient — how much of the query's vocabulary the content
/// covers — so a near-verbatim restatement scores high (≥0.92, the dedup
/// threshold) while an unrelated sentence scores 0. Cheap, stable, and enough
/// to drive every decision a test makes; a real embedder is trusted for the
/// same property but can never be asserted on exactly.
pub fn lexical_similarity(query: &str, content: &str) -> f32 {
    use std::collections::BTreeSet;
    let words = |s: &str| -> BTreeSet<String> {
        s.split(|c: char| !c.is_alphanumeric())
            .filter(|w| !w.is_empty())
            .map(|w| w.to_lowercase())
            .collect()
    };
    let q = words(query);
    let c = words(content);
    if q.is_empty() || c.is_empty() {
        return 0.0;
    }
    let shared = q.intersection(&c).count() as f32;
    (shared / q.len() as f32).min(shared / c.len() as f32)
}

impl MemoryBackend for FakeMemoryBackend {
    fn recall(
        &self,
        query: &str,
        top_k: usize,
        _min_score: f32,
        namespace: Option<&str>,
    ) -> RecallResult {
        self.sleep();
        if self.is_broken() {
            return RecallResult::Degraded("fake backend is broken".into());
        }
        self.dispatched.fetch_add(1, Ordering::Relaxed);
        let store = self.store.lock().unwrap();
        let want = effective_namespace(namespace);
        let mut hits: Vec<MemoryHit> = store
            .rows
            .iter()
            .filter(|h| !h.deprecated)
            .filter(|h| h.namespace == want)
            .map(|h| {
                let mut h = h.clone();
                h.score = lexical_similarity(query, &h.content);
                h
            })
            .filter(|h| h.score >= _min_score)
            .collect();
        hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.id.cmp(&b.id))
        });
        hits.truncate(top_k);
        RecallResult::Hits(hits)
    }

    fn remember(
        &self,
        content: &str,
        tags: &[&str],
        namespace: Option<&str>,
        memory_type: Option<&str>,
        metadata: Option<serde_json::Value>,
    ) -> RememberResult {
        self.sleep();
        if self.is_broken() {
            return RememberResult::Failed("fake backend is broken".into());
        }
        self.dispatched.fetch_add(1, Ordering::Relaxed);
        let mut store = self.store.lock().unwrap();
        store.next_id += 1;
        let id = format!("mem-{:04}", store.next_id);
        store.rows.push(MemoryHit {
            id: id.clone(),
            content: content.to_string(),
            score: 1.0,
            namespace: effective_namespace(namespace),
            memory_type: memory_type.unwrap_or("note").to_string(),
            source: metadata
                .as_ref()
                .and_then(|m| m.get("source"))
                .and_then(|s| s.as_str())
                .map(|s| s.to_string()),
            source_type: metadata
                .as_ref()
                .and_then(|m| m.get("source_type"))
                .and_then(|s| s.as_str())
                .unwrap_or("unknown")
                .to_string(),
            tags: tags.iter().map(|t| t.to_string()).collect(),
            created_at: "2026-01-01T00:00:00Z".into(),
            deprecated: false,
        });
        RememberResult::Ok(id)
    }

    fn forget(&self, id: &str) -> Result<(), String> {
        let mut store = self.store.lock().unwrap();
        store.rows.retain(|h| h.id != id);
        Ok(())
    }

    fn soft_forget(&self, id: &str, reason: &str) -> Result<(), String> {
        let _ = reason;
        let mut store = self.store.lock().unwrap();
        match store.rows.iter_mut().find(|h| h.id == id) {
            Some(h) => {
                h.deprecated = true;
                Ok(())
            }
            None => Err(format!("memory not found: {id}")),
        }
    }

    fn list(&self, namespace: Option<&str>, limit: usize) -> Vec<MemoryHit> {
        let store = self.store.lock().unwrap();
        let want = effective_namespace(namespace);
        store
            .rows
            .iter()
            .filter(|h| !h.deprecated)
            .filter(|h| h.namespace == want)
            .take(limit)
            .cloned()
            .collect()
    }

    fn get(&self, id: &str) -> Option<MemoryHit> {
        let store = self.store.lock().unwrap();
        store
            .rows
            .iter()
            .find(|h| h.id == id || h.id.starts_with(id))
            .cloned()
    }
}

/// Arc a fake backend as the trait object `MemoryService` expects.
pub fn arc_backend(b: Arc<FakeMemoryBackend>) -> Arc<dyn MemoryBackend> {
    b
}

/// Convenience for one-off services.
pub fn service_with(
    backend: Arc<FakeMemoryBackend>,
    queue_bound: usize,
    timeout: Duration,
) -> (
    ::agent::memory_integration::MemoryService,
    Arc<FakeMemoryBackend>,
) {
    let svc = ::agent::memory_integration::MemoryService::new(
        arc_backend(backend.clone()),
        queue_bound,
        timeout,
    );
    (svc, backend)
}

/// Run the service once and shut it down, so the worker thread exits between
/// tests instead of accumulating for the length of the test binary.
pub async fn svc_roundtrip<T, F>(backend: Arc<FakeMemoryBackend>, f: F) -> T
where
    F: FnOnce(
        ::agent::memory_integration::MemoryService,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = T> + Send>>,
{
    let (svc, _) = service_with(backend, 16, Duration::from_secs(5));
    f(svc.clone()).await
}

/// Atomic counter helper for tests that need a shared tick.
#[derive(Debug, Default)]
pub struct Counter(pub AtomicU64);

impl Counter {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }
    pub fn bump(&self) -> u64 {
        self.0.fetch_add(1, Ordering::Relaxed) + 1
    }
    pub fn get(&self) -> u64 {
        self.0.load(Ordering::Relaxed)
    }
}
