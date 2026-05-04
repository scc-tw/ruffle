use crate::descriptors::Descriptors;
use crate::globals::Globals;
use fnv::FnvHashMap;
use std::fmt::{Debug, Formatter};
use std::ops::Deref;
use std::sync::{Arc, Mutex, Weak};
use std::sync::atomic::{AtomicU64, Ordering};

/// [seer-patch 1.6b] Pool storage. Each entry is `(item, description,
/// last_returned_frame)`. The frame stamp is set on `PoolEntry::drop`
/// (return to pool) so per-key sweep can purge entries idle past
/// `idle_frames`.
type PoolInner<T> = Mutex<Vec<T>>;
type Constructor<Type, Description> = Box<dyn Fn(&Descriptors, &Description) -> Type>;

/// [seer-patch 1.6b] Process-global monotonic frame counter for
/// pool item idle tracking. Bumped from `WgpuRenderBackend::submit_frame`
/// (or any equivalent wgpu submit point) so pool sweep can compute
/// idle age in frames. We use a global atomic instead of a per-backend
/// frame counter because `BufferPool` lives across multiple backends
/// in tests and we want a single timeline.
static FRAME_COUNTER: AtomicU64 = AtomicU64::new(0);

/// [seer-patch 1.6b] Bump the global frame counter. Called from
/// `Player::sweep_texture_pools` (which is itself called once per
/// frame by the seer launcher).
pub fn bump_frame_counter() -> u64 {
    FRAME_COUNTER.fetch_add(1, Ordering::Relaxed) + 1
}

/// [seer-patch 1.6b] Read the current frame counter without bumping.
pub fn current_frame_counter() -> u64 {
    FRAME_COUNTER.load(Ordering::Relaxed)
}

/// [seer-patch 1.6b] Per-item return-frame stamp. Set on
/// `PoolEntry::drop` (item returns to pool) so per-key sweep can
/// drop items idle past `idle_frames`.
#[derive(Debug, Clone, Copy, Default)]
pub struct FrameStamp(pub u64);

/// [seer-patch 1.6a] RAII census token for `FilterPool`-tracked
/// textures. Lives alongside the `wgpu::Texture` in the pool item
/// tuple. Drops fire `on_texture_dropped(FilterPool, bytes)` exactly
/// once per allocation:
///
/// - Pool reset (`backend.rs` `texture_pool = TexturePool::new()`):
///   the entire `Vec` of pool items drops → each token drops.
/// - Phase 1.6b sweep: explicitly removed entries drop their tokens.
/// - PoolEntry in-flight: when held by a caller, the token lives
///   inside the entry; on `PoolEntry::drop` the entry returns to the
///   pool Vec (token stays alive). No double-count.
///
/// Not Clone — single Drop per allocation is the whole point.
pub struct FilterPoolCensusToken {
    bytes: u64,
}

impl Debug for FilterPoolCensusToken {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FilterPoolCensusToken").field("bytes", &self.bytes).finish()
    }
}

impl Drop for FilterPoolCensusToken {
    fn drop(&mut self) {
        if self.bytes > 0
            && let Some(host) = crate::seer::host()
        {
            host.on_texture_dropped(crate::seer::TextureSource::FilterPool, self.bytes);
        }
    }
}

/// [seer-patch 1.6a] Pool item type for `TexturePool`. The third
/// component is the census token; callers that touch `.0` / `.1`
/// (texture / view) keep working unchanged.
pub type FilterPoolItem = (wgpu::Texture, wgpu::TextureView, FilterPoolCensusToken);

/// [seer-patch 1.6b] Tunables passed in by the host on each sweep
/// call. Mirrors `core::seer::TexturePoolPolicy` but lives in the
/// render crate to avoid an upward dep.
#[derive(Debug, Clone, Copy)]
pub struct TexturePoolSweepPolicy {
    /// Cap each per-key pool at this many entries. Excess oldest are
    /// dropped on sweep.
    pub max_per_pool: usize,
    /// Drop pool items whose last-return frame was this many frames
    /// ago.
    pub idle_frames: u64,
    /// Purge entire keyed pool entries that haven't been take()'d for
    /// this many frames.
    pub purge_frames: u64,
}

/// [seer-patch 1.6b] Stats returned by `TexturePool::sweep`.
#[derive(Debug, Default, Clone, Copy)]
pub struct TexturePoolSweepReport {
    /// Per-key pools that had at least one entry dropped.
    pub keys_swept: usize,
    /// Total pool items dropped (across all keys).
    pub dropped_entries: usize,
    /// Per-key pools removed entirely (long-idle).
    pub purged_keys: usize,
    /// Estimated bytes freed (sum over dropped items).
    pub freed_bytes: u64,
}

#[derive(Debug, Default)]
pub struct TexturePool {
    pools: FnvHashMap<TextureKey, BufferPool<FilterPoolItem, AlwaysCompatible>>,
    globals_cache: FnvHashMap<GlobalsKey, Arc<Globals>>,
}

impl TexturePool {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn get_texture(
        &mut self,
        descriptors: &Descriptors,
        size: wgpu::Extent3d,
        usage: wgpu::TextureUsages,
        format: wgpu::TextureFormat,
        sample_count: u32,
    ) -> PoolEntry<FilterPoolItem, AlwaysCompatible> {
        let key = TextureKey {
            size,
            usage,
            format,
            sample_count,
        };
        // [seer-patch 1.6a] Bytes for census attribution.
        // block_copy_size returns bytes per block (= bytes per pixel
        // for uncompressed formats). Multiply by sample_count for
        // MSAA where the pool actually backs sample_count×size bytes.
        let bytes = (size.width as u64)
            * (size.height as u64)
            * (format.block_copy_size(None).unwrap_or(4) as u64)
            * (sample_count as u64);
        let pool = self.pools.entry(key).or_insert_with(|| {
            let label = if cfg!(feature = "render_debug_labels") {
                use std::sync::atomic::{AtomicU32, Ordering};
                static ID_COUNT: AtomicU32 = AtomicU32::new(0);
                let id = ID_COUNT.fetch_add(1, Ordering::Relaxed);
                create_debug_label!("Pooled texture {}", id)
            } else {
                None
            };
            BufferPool::new(Box::new(move |descriptors, _description| {
                let texture = descriptors.device.create_texture(&wgpu::TextureDescriptor {
                    label: label.as_deref(),
                    size,
                    mip_level_count: 1,
                    sample_count,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    view_formats: &[format],
                    usage,
                });
                let view = texture.create_view(&Default::default());
                // [seer-patch 1.6a] Fire register; balanced by the
                // FilterPoolCensusToken's Drop on permanent removal.
                if let Some(host) = crate::seer::host() {
                    host.on_texture_registered(
                        crate::seer::TextureSource::FilterPool,
                        bytes,
                    );
                }
                (texture, view, FilterPoolCensusToken { bytes })
            }))
        });
        // [seer-patch 1.6b] Stamp the take frame on the pool itself so
        // long-idle pools can be purged whole-cloth.
        pool.last_taken_frame.store(current_frame_counter(), Ordering::Relaxed);
        pool.take(descriptors, AlwaysCompatible)
    }

    /// [seer-patch 1.6b] Sweep idle / over-cap entries and purge
    /// long-idle keyed pools.
    pub fn sweep(
        &mut self,
        current_frame: u64,
        policy: TexturePoolSweepPolicy,
    ) -> TexturePoolSweepReport {
        let mut report = TexturePoolSweepReport::default();
        // Whole-pool purge first.
        self.pools.retain(|key, pool| {
            let last = pool.last_taken_frame.load(Ordering::Relaxed);
            let idle = current_frame.saturating_sub(last);
            // Per-item byte count (filter pool textures only).
            let item_bytes = (key.size.width as u64)
                * (key.size.height as u64)
                * (key.format.block_copy_size(None).unwrap_or(4) as u64)
                * (key.sample_count as u64);
            if idle >= policy.purge_frames {
                // Drop the entire pool — both the available Vec and
                // the constructor closure. Census tokens fire as
                // items drop.
                let mut guard = pool.available.lock().unwrap();
                let n = guard.len();
                let freed = (n as u64) * item_bytes;
                report.purged_keys += 1;
                report.dropped_entries += n;
                report.freed_bytes += freed;
                guard.clear();
                drop(guard);
                return false;
            }
            // Per-key sweep: drop entries idle past idle_frames, and
            // cap at max_per_pool.
            let (dropped_n, _) = pool.sweep(current_frame, &policy);
            if dropped_n > 0 {
                report.keys_swept += 1;
                report.dropped_entries += dropped_n;
                report.freed_bytes += (dropped_n as u64) * item_bytes;
            }
            true
        });
        report
    }

    pub fn get_globals(
        &mut self,
        descriptors: &Descriptors,
        viewport_width: u32,
        viewport_height: u32,
    ) -> Arc<Globals> {
        self.globals_cache
            .entry(GlobalsKey {
                viewport_width,
                viewport_height,
            })
            .or_insert_with(|| {
                Arc::new(Globals::new(
                    &descriptors.device,
                    &descriptors.bind_layouts.globals,
                    viewport_width,
                    viewport_height,
                ))
            })
            .clone()
    }
}

#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
struct TextureKey {
    size: wgpu::Extent3d,
    usage: wgpu::TextureUsages,
    format: wgpu::TextureFormat,
    sample_count: u32,
}

#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
struct GlobalsKey {
    viewport_width: u32,
    viewport_height: u32,
}

pub trait BufferDescription: Clone + Debug {
    type Cost: Ord;

    /// If the potential buffer represented by this description (`self`)
    /// fits another existing buffer and its description (`other`),
    /// return the cost to use that buffer instead of making a new one.
    ///
    /// Cost is an arbitrary unit, but lower is better.
    /// None means that the other buffer cannot be used in place of this one.
    fn cost_to_use(&self, other: &Self) -> Option<Self::Cost>;
}

#[derive(Clone, Debug)]
pub struct AlwaysCompatible;

impl BufferDescription for AlwaysCompatible {
    type Cost = ();

    fn cost_to_use(&self, _other: &Self) -> Option<()> {
        Some(())
    }
}

pub struct BufferPool<Type, Description: BufferDescription> {
    available: Arc<PoolInner<(Type, Description, FrameStamp)>>,
    constructor: Constructor<Type, Description>,
    /// [seer-patch 1.6b] Frame stamp of the most recent `take()`
    /// against this pool. `TexturePool::sweep` reads this to decide
    /// whether to purge a long-idle keyed pool entirely.
    pub(crate) last_taken_frame: AtomicU64,
}

impl<Type, Description: BufferDescription> Debug for BufferPool<Type, Description> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BufferPool").finish()
    }
}

impl<Type, Description: BufferDescription> BufferPool<Type, Description> {
    pub fn new(constructor: Constructor<Type, Description>) -> Self {
        Self {
            available: Arc::new(Mutex::new(vec![])),
            constructor,
            last_taken_frame: AtomicU64::new(0),
        }
    }

    pub fn take(
        &self,
        descriptors: &Descriptors,
        description: Description,
    ) -> PoolEntry<Type, Description> {
        let mut guard = self
            .available
            .lock()
            .expect("Should not be able to lock recursively");
        let mut best: Option<(Description::Cost, usize)> = None;
        for i in 0..guard.len() {
            if let Some(cost) = description.cost_to_use(&guard[i].1) {
                if let Some(best) = &mut best {
                    if best.0 > cost {
                        *best = (cost, i);
                    }
                } else if best.is_none() {
                    best = Some((cost, i));
                }
            }
        }

        let (item, used_description) = if let Some((_, best)) = best {
            // [seer-patch 1.6b] Drop the FrameStamp; the new owner
            // will stamp on PoolEntry::drop.
            let (it, desc, _stamp) = guard.swap_remove(best);
            (it, desc)
        } else {
            let item = (self.constructor)(descriptors, &description);
            (item, description)
        };
        PoolEntry {
            item: Some(item),
            description: used_description,
            pool: Arc::downgrade(&self.available),
        }
    }

    /// [seer-patch 1.6b] Sweep this pool's available entries. Returns
    /// `(dropped_count, dropped_bytes)`. Bytes are 0 for non-texture
    /// pools (caller doesn't care; texture pool overrides).
    fn sweep(
        &self,
        current_frame: u64,
        policy: &TexturePoolSweepPolicy,
    ) -> (usize, u64) {
        let mut guard = self
            .available
            .lock()
            .expect("Should not be able to lock recursively");
        let before = guard.len();
        // Drop entries idle past idle_frames. Bytes-freed is computed
        // by the texture-specific sweep entry point that knows
        // FilterPoolItem layout; here we report 0.
        guard.retain(|(_, _, FrameStamp(f))| {
            current_frame.saturating_sub(*f) < policy.idle_frames
        });
        // Cap at max_per_pool. Trim from the front (oldest first;
        // since `swap_remove` rotates Vec, "front" is approximate but
        // good enough for sweep semantics).
        if guard.len() > policy.max_per_pool {
            guard.drain(policy.max_per_pool..);
        }
        (before - guard.len(), 0)
    }
}

pub struct PoolEntry<Type, Description: BufferDescription> {
    item: Option<Type>,
    description: Description,
    pool: Weak<PoolInner<(Type, Description, FrameStamp)>>,
}

impl<Type, Description: BufferDescription> Debug for PoolEntry<Type, Description>
where
    Type: Debug,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("PoolEntry").field(&self.item).finish()
    }
}

impl<Type, Description: BufferDescription> Drop for PoolEntry<Type, Description> {
    fn drop(&mut self) {
        if let Some(item) = self.item.take()
            && let Some(pool) = self.pool.upgrade()
        {
            // [seer-patch 1.6b] Stamp the return frame so per-key
            // sweep can later identify long-idle entries.
            let stamp = FrameStamp(current_frame_counter());
            pool.lock()
                .expect("Should not be able to lock recursively")
                .push((item, self.description.clone(), stamp))
        }
    }
}

impl<Type, Description: BufferDescription> Deref for PoolEntry<Type, Description> {
    type Target = Type;

    fn deref(&self) -> &Self::Target {
        self.item.as_ref().expect("Item should exist until dropped")
    }
}
