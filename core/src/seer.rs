//! [seer-patch] Host-supplied core context for seer-specific behaviour.
//!
//! Mirrors `ruffle_render_wgpu::seer` but for the core crate. The seer
//! launcher implements [`CoreSeerHost`] (alongside the wgpu-side
//! `SeerHost`) and installs it once at startup with [`install`].
//! Patched call sites query the installed host through [`host()`] and
//! fall through to upstream Ruffle behaviour when the slot is empty.
//!
//! ## Why a separate trait from the wgpu-side `SeerHost`
//!
//! The wgpu host trait lives in `ruffle_render_wgpu`. `ruffle_core`
//! cannot depend on `ruffle_render_wgpu` without inverting the
//! existing `core → render → render_wgpu` dependency chain. The
//! trait below is the `core`-layer parallel: same install-once /
//! `OnceLock` pattern, same defaults-preserve-upstream rule. A
//! single host type in `seer-flash` implements both traits.
//!
//! ## Cost when off
//!
//! Each query is one `OnceLock::get` (relaxed atomic load) plus, if
//! installed, one virtual call. With no host installed, the patched
//! call sites are byte-identical to upstream Ruffle.

use std::sync::{Arc, OnceLock};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

/// [seer-patch 1.6b/c] Process-global monotonic frame counter used
/// by both `TexturePool` sweep (filter intermediates) and
/// `BitmapCache` sweep (cacheAsBitmap render targets) to age idle
/// entries.
///
/// Bumped from `Player::sweep_texture_pools` (which the seer
/// launcher calls once per frame). Reading is `Relaxed` — these
/// counters are advisory; race-free monotonic.
static FRAME_COUNTER: AtomicU64 = AtomicU64::new(0);

/// [seer-patch 1.6b/c] Bump the global frame counter and return the
/// new value. Called once per frame from the launcher's sweep loop.
pub fn bump_frame_counter() -> u64 {
    FRAME_COUNTER.fetch_add(1, Ordering::Relaxed) + 1
}

/// [seer-patch 1.6b/c] Read the current frame counter without
/// bumping. Used by `BitmapCache::bump_last_drawn` (render path).
pub fn current_frame_counter() -> u64 {
    FRAME_COUNTER.load(Ordering::Relaxed)
}

/// Stats returned by `Player::sweep_idle_libraries`. Hosts use
/// these to format a `[swf-sweep]` log line and update memory-
/// pressure UI.
#[derive(Debug, Default, Clone, Copy)]
pub struct SweepReport {
    /// Number of `MovieLibrary` entries removed in this sweep.
    pub removed: usize,
    /// Estimated GPU bytes freed (sum of `bitmap_bytes()` across
    /// removed libraries, measured at remove time). The actual
    /// reclamation lags by one wgpu submit + one gc-arena cycle.
    pub freed_bitmap_bytes: u64,
}

/// Tunables for the layer-2 asset arena (per-SWF mimalloc-shaped
/// page eviction). When `Some(...)` is returned from
/// [`CoreSeerHost::asset_arena_policy`], `Player::post_frame_sweep`
/// becomes active and the host opts into per-frame liveness sweeps
/// of `MovieLibrary` entries.
///
/// `None` (the default) preserves upstream Ruffle behaviour: no
/// sweeps run, no libraries are evicted, and the only memory
/// reclamation comes from natural `Arc<SwfMovie>` drops.
#[derive(Debug, Clone)]
pub struct AssetArenaPolicy {
    /// A library is only considered abandonable if no character
    /// lookup has touched it for at least this long. Prevents us
    /// from evicting a SWF whose AS3 path is in a brief idle
    /// window between two uses (e.g. between two consecutive
    /// `Loader.load`s inside the same frame).
    pub idle_threshold: Duration,

    /// Skip libraries that contain at least one `Sound` character.
    /// Dropping them would kill mid-playback audio; conservative:
    /// we accept holding on to a SWF until its sounds finish
    /// rather than risk a glitch.
    pub never_evict_with_audio: bool,

    /// Skip the player's root SWF. Should always be `true` in
    /// practice; the option is here for completeness.
    pub never_evict_root: bool,
}

impl Default for AssetArenaPolicy {
    fn default() -> Self {
        Self {
            idle_threshold: Duration::from_secs(30),
            never_evict_with_audio: true,
            never_evict_root: true,
        }
    }
}

/// [seer-patch P1] Strategy for re-decoding an evicted bitmap on its
/// next sample.
///
/// `Sync` decodes inline on the render thread (simplest, but causes a
/// frame stutter — typically 10–100 ms for big JPEGs).
///
/// `Async` posts the decode to a background worker and renders a
/// transparent placeholder for the in-flight frame. The render thread
/// re-checks the slot on the next render and picks up the completed
/// upload. Recommended default.
#[derive(Debug, Clone, Copy)]
pub enum BitmapDecodeStrategy {
    /// Inline decode on the render thread.
    Sync,
    /// Background decode + transparent placeholder for the missed frame(s).
    Async,
}

impl Default for BitmapDecodeStrategy {
    fn default() -> Self {
        Self::Async
    }
}

/// [seer-patch P1] Tunables for the per-bitmap GPU-residency sweep
/// (`Player::sweep_idle_bitmaps`). When `Some(...)` is returned from
/// [`CoreSeerHost::bitmap_residency_policy`], the host opts into
/// dropping idle `BitmapHandle`s while keeping the source
/// `CompressedBitmap` (or `BitmapData::pixels`) alive for cheap
/// re-realisation.
///
/// `None` (the default) preserves upstream Ruffle behaviour: bitmap
/// GPU handles live for the lifetime of their `BitmapCharacter` /
/// `BitmapData` and are only freed by lib-level eviction or
/// gc-arena collection.
#[derive(Debug, Clone)]
pub struct BitmapResidencyPolicy {
    /// Drop a bitmap's GPU handle if it has not been sampled (via
    /// `bitmap_handle()` / `try_bitmap_handle()`) for at least this
    /// long. Recommended: 5–15 seconds for tutorial-fight workload.
    pub idle_threshold: Duration,

    /// How to handle the next sample after eviction. See
    /// [`BitmapDecodeStrategy`].
    pub decode_strategy: BitmapDecodeStrategy,
    // TODO(seer-patch P1): replace `idle_threshold` with a
    // congestion-aware controller (TCP-style AIMD on bitmap-bytes
    // headroom vs. budget). Track re-decode rate as the "loss"
    // signal — high rate ⇒ threshold too short, back off; low
    // rate ⇒ threshold can shrink toward target budget. Q3 in
    // docs/plans/bitmap-memory.md.
}

impl Default for BitmapResidencyPolicy {
    fn default() -> Self {
        Self {
            idle_threshold: Duration::from_secs(10),
            decode_strategy: BitmapDecodeStrategy::default(),
        }
    }
}

/// [seer-patch 1.6b] Tunables for the `TexturePool` (filter
/// intermediate / surface render-target) sweep. When `Some` is
/// returned from [`CoreSeerHost::texture_pool_policy`],
/// `Player::sweep_texture_pools` will cap each per-key pool, drop
/// idle entries, and purge whole-pool entries.
///
/// `None` (default) preserves upstream behaviour (unbounded pool).
#[derive(Debug, Clone, Copy)]
pub struct TexturePoolPolicy {
    /// Cap each per-key pool at this many entries. Most keys see
    /// at most ~4 simultaneous in-flight items (filter ping-pong +
    /// read + write). Default 4.
    pub max_per_pool: usize,
    /// Drop pool items not returned for this many frames. Default
    /// 60 (~1 s at 60 fps).
    pub idle_frames: u64,
    /// Drop the entire keyed pool if no `take()` for this many
    /// frames. Default 600 (~10 s).
    pub purge_frames: u64,
}

impl Default for TexturePoolPolicy {
    fn default() -> Self {
        Self {
            max_per_pool: 4,
            idle_frames: 60,
            purge_frames: 600,
        }
    }
}

/// [seer-patch 1.6b] Stats returned by `Player::sweep_texture_pools`.
#[derive(Debug, Default, Clone, Copy)]
pub struct TexturePoolSweepReport {
    /// Per-key pools that had at least one entry dropped.
    pub keys_swept: usize,
    /// Total pool items dropped (across all keys).
    pub dropped_entries: usize,
    /// Per-key pools removed entirely (long-idle).
    pub purged_keys: usize,
    /// Estimated bytes freed.
    pub freed_bytes: u64,
}

/// [seer-patch 1.6c] Tunables for the `BitmapCache` (cacheAsBitmap
/// render target) sweep. When `Some` is returned from
/// [`CoreSeerHost::bitmap_cache_policy`], `Player::sweep_idle_bitmap_caches`
/// drops `BitmapInfo` from caches that haven't been drawn for K
/// frames. Re-render on next visible frame regenerates the cache.
///
/// `None` (default) preserves upstream behaviour (cache lives until
/// display object drops or size changes).
#[derive(Debug, Clone, Copy)]
pub struct BitmapCachePolicy {
    /// Drop the cache if not drawn for this many frames. Default 300
    /// (~5 s at 60 fps). Cache is much cheaper to recreate than a
    /// JPEG-decoded library bitmap, so we can be more aggressive.
    pub idle_frames: u64,
    /// Skip eviction for caches smaller than this; cheap to keep.
    /// Default 256 KB.
    pub min_bytes_to_evict: u64,
}

impl Default for BitmapCachePolicy {
    fn default() -> Self {
        Self {
            idle_frames: 300,
            min_bytes_to_evict: 256 * 1024,
        }
    }
}

/// [seer-patch 1.6c] Stats returned by `Player::sweep_idle_bitmap_caches`.
#[derive(Debug, Default, Clone, Copy)]
pub struct BitmapCacheSweepReport {
    /// Number of `BitmapCache::bitmap` slots cleared.
    pub evicted: usize,
    /// Estimated bytes freed (sum of `width × height × 4`).
    pub freed_bytes: u64,
    /// Number of caches kept (recently drawn or below
    /// `min_bytes_to_evict`).
    pub kept: usize,
    /// Total caches with realised bitmaps inspected this sweep.
    pub total_realised: usize,
}

/// [seer-patch P1] Stats returned by `Player::sweep_idle_bitmaps`.
/// Hosts use these to format a `[bitmap-sweep]` log line and feed
/// the memory-pressure overlay.
///
/// Split into char- and data-side counters so logs can distinguish
/// "no bitmap display objects in tree" (data totals = 0) from
/// "found them but kept all of them" (data totals > 0, evicted_data
/// = 0). See `docs/plans/bitmap-memory.md` Q4 for context.
#[derive(Debug, Default, Clone, Copy)]
pub struct BitmapSweepReport {
    /// Number of `BitmapCharacter` GPU handles dropped this sweep.
    pub evicted_chars: usize,
    /// Number of `BitmapData` GPU handles dropped this sweep.
    pub evicted_data: usize,
    /// Estimated GPU bytes freed across `BitmapCharacter` evictions
    /// (sum of `width × height × 4`). Actual reclamation lags by
    /// one wgpu submit + one gc-arena cycle, same caveat as
    /// `SweepReport`.
    pub freed_char_bytes: u64,
    /// Estimated GPU bytes freed across `BitmapData` evictions.
    pub freed_data_bytes: u64,
    /// Number of realised `BitmapCharacter` handles kept this sweep
    /// (recently sampled within the idle threshold).
    pub kept_chars: usize,
    /// Number of realised `BitmapData` handles kept this sweep
    /// (recently sampled within the idle threshold OR dirty_state
    /// not Clean).
    pub kept_data: usize,
    /// Total realised `BitmapCharacter` handles inspected.
    pub total_chars_realised: usize,
    /// Total realised `BitmapData` handles inspected (visited via
    /// the display-tree walk).
    pub total_data_realised: usize,
    /// `BitmapData`s skipped because `dirty_state != Clean`. Counted
    /// separately from `kept_data` (which is "kept due to recent
    /// sample") so we can tell apart the two skip reasons. Subset
    /// of `total_data_realised`; not double-counted in `kept_data`.
    pub skipped_dirty: usize,
}

/// One snapshot of a `MovieLibrary`'s memory footprint. Returned
/// by `Player::library_report` so the host can render a memory-
/// pressure overlay or log line.
///
/// Three byte categories are tracked separately so we can tell at
/// a glance which storage layer dominates a given library:
///
///   * `bitmap_bytes` — **decoded** bitmaps already uploaded to
///     the GPU (lazy `BitmapHandle` realised). Lives in
///     `wgpu::Texture` / `VkDeviceMemory`. Released by dropping
///     the `MovieLibrary`.
///   * `compressed_bitmap_bytes` — **lazy** bitmaps still stored
///     as compressed source (`CompressedBitmap::Jpeg` /
///     `Lossless`). Lives in CPU heap. Comment in `character.rs`
///     notes a single SWF can hold ~10 GB if all decoded; staying
///     compressed is a deliberate memory tradeoff.
///   * `swf_data_bytes` — the entire `SwfMovie::data: Vec<u8>`
///     (raw SWF source bytes). Lives in CPU heap. Held alive by
///     the library's own `Arc<SwfMovie>` strong ref. Often
///     dominates because Ruffle re-reads tags lazily and never
///     drops the source bytes.
///
/// Character count breakdown (`shape_count` / `sprite_count` /
/// `font_count` / `text_count` / `sound_count`) helps locate
/// memory hidden in non-bitmap parsed state — tessellated
/// `Graphic`s, font glyph atlases, AVM2-bound `Sprite` timelines.
#[derive(Debug, Clone)]
pub struct LibraryEntry {
    /// The source SWF's URL, if known.
    pub url: Option<String>,
    /// Sum of `width × height × 4` for each `BitmapCharacter`
    /// whose `BitmapHandle` has been realised. Lives on GPU.
    pub bitmap_bytes: u64,
    /// Sum of source bytes (`CompressedBitmap::Jpeg.data.len()`
    /// + alpha + `Lossless.data`) across every `BitmapCharacter`,
    /// regardless of whether its handle has been realised. Lives
    /// in CPU heap.
    pub compressed_bitmap_bytes: u64,
    /// `SwfMovie::data().len()` — the raw SWF source bytes the
    /// library holds via its `Arc<SwfMovie>`. CPU heap.
    pub swf_data_bytes: u64,
    /// Total `Character` entries in the library.
    pub character_count: usize,
    /// `Character::Graphic` + `Character::MorphShape` — shape
    /// tessellations live in renderer-side caches keyed by these.
    pub shape_count: usize,
    /// `Character::MovieClip` — sub-timelines, each carrying its
    /// own AVM1/AVM2 frame action lists.
    pub sprite_count: usize,
    /// `Character::Font` — glyph atlases live keyed off these in
    /// the renderer's font cache.
    pub font_count: usize,
    /// `Character::Text` + `Character::EditText` — parsed text
    /// runs.
    pub text_count: usize,
    /// `Character::Sound` entries — non-zero blocks eviction when
    /// `never_evict_with_audio` is set.
    pub sound_count: usize,
    /// Time since any character lookup touched this library.
    pub idle_for: Duration,
    /// `Arc::strong_count(&swf) - 1` (subtract the MovieLibrary's
    /// own strong ref). `0` means the library is abandonable,
    /// subject to the idle / audio / root gates.
    pub external_refs: usize,
}

/// Host-side seer context for the `core` crate.
///
/// Every method has a default that preserves upstream Ruffle
/// behaviour. New patch points should land here as new defaulted
/// methods so existing host implementations keep compiling.
///
/// `Send + Sync + 'static` because the host is stored in a
/// process-global slot. In Ruffle's current single-threaded player
/// loop the bound is overkill; it's cheap and keeps the door open.
pub trait CoreSeerHost: Send + Sync + 'static {
    /// If `Some`, enables `Player::sweep_idle_libraries` to evict
    /// idle `MovieLibrary` entries when the host calls it.
    ///
    /// Default: `None` (no sweep, identical to upstream Ruffle).
    fn asset_arena_policy(&self) -> Option<AssetArenaPolicy> {
        None
    }

    /// Park the raw bytes of an evicted SWF in the host's warm
    /// cache (Phase 4 of the layer-2 plan). The slice is borrowed
    /// for the duration of the call; hosts that want to keep the
    /// bytes must copy them. The next `Loader.load` for the same
    /// URL can satisfy from this cache instead of re-fetching.
    ///
    /// `url` is the SWF's `SwfMovie::url()` — may be empty, in
    /// which case the host should skip caching (no key).
    ///
    /// Default: no-op (warm cache disabled — bytes are dropped
    /// when the SWF is evicted).
    fn park_warm_bytes(&self, _url: &str, _bytes: &[u8]) {}

    /// Try to satisfy a fetch from the warm cache. Returning
    /// `Some(bytes)` lets the navigator skip the network round-
    /// trip; returning `None` falls through to the normal fetch.
    ///
    /// Default: `None` (cache disabled). Hosts implement by
    /// looking up `url` in their warm cache and returning a
    /// clone of the byte buffer on hit.
    fn take_warm_bytes(&self, _url: &str) -> Option<Arc<[u8]>> {
        None
    }

    /// [seer-patch P1] If `Some`, enables `Player::sweep_idle_bitmaps`
    /// to drop idle GPU `BitmapHandle`s while keeping their source
    /// `CompressedBitmap` / `BitmapData::pixels` alive for cheap
    /// re-realisation on next sample.
    ///
    /// Default: `None` (no sweep, identical to upstream Ruffle).
    fn bitmap_residency_policy(&self) -> Option<BitmapResidencyPolicy> {
        None
    }

    /// [seer-patch P1 Day 3] Submit a `CompressedBitmap` for
    /// background decode. Returns an opaque request id the caller
    /// pairs with the originating `BitmapCharacter`. Future render
    /// frames poll via [`try_take_decoded_bitmap`] until the
    /// decode completes.
    ///
    /// Returns `None` to signal "async path unavailable; caller
    /// should fall back to inline sync decode". Hosts that don't
    /// implement async decode get the upstream behaviour.
    fn submit_async_decode(
        &self,
        _compressed: crate::character::CompressedBitmap,
    ) -> Option<u64> {
        None
    }

    /// [seer-patch P1 Day 3] Claim the result of a previously
    /// submitted async decode. Returns `None` if the decode is
    /// still in flight (or, rarely, failed — host logs failures and
    /// drops the entry; the caller's character keeps returning the
    /// placeholder until eviction-and-resample triggers a fresh
    /// sync attempt). Returns `Some(_)` on completed decode.
    ///
    /// Consumes the mailbox entry — duplicate calls return `None`.
    fn try_take_decoded_bitmap(
        &self,
        _id: u64,
    ) -> Option<ruffle_render::bitmap::Bitmap<'static>> {
        None
    }

    /// [seer-patch P1 Day 3] Lazy 1×1 transparent placeholder
    /// `BitmapHandle`. Returned from `BitmapCharacter::bitmap_handle`
    /// while an async decode is in flight, so the render path has
    /// something to sample from for the missed frame(s) instead of
    /// stalling on JPEG decode.
    ///
    /// Should be cheap to call (host caches a single shared handle
    /// after first registration). Returns `None` if the host doesn't
    /// implement the placeholder; caller falls back to sync decode.
    fn placeholder_bitmap(
        &self,
        _backend: &mut dyn ruffle_render::backend::RenderBackend,
    ) -> Option<ruffle_render::bitmap::BitmapHandle> {
        None
    }

    /// [seer-patch 1.6b] If `Some`, enables `Player::sweep_texture_pools`
    /// to bound the wgpu `TexturePool` (filter intermediate + render
    /// target pool) so peak concurrent demand doesn't get permanently
    /// retained.
    ///
    /// Default: `None` (no sweep, upstream behaviour).
    fn texture_pool_policy(&self) -> Option<TexturePoolPolicy> {
        None
    }

    /// [seer-patch 1.6c] If `Some`, enables `Player::sweep_idle_bitmap_caches`
    /// to drop `BitmapCache` (cacheAsBitmap render targets) that
    /// haven't been drawn in `idle_frames`. Re-render on next visible
    /// frame regenerates the cache transparently.
    ///
    /// Default: `None` (no sweep, upstream behaviour).
    fn bitmap_cache_policy(&self) -> Option<BitmapCachePolicy> {
        None
    }
}

/// A trivial host that returns every upstream default. Useful as a
/// starting point for hosts that only want to override one or two
/// hooks: derive your own type and only implement what you change.
pub struct DefaultCoreSeerHost;
impl CoreSeerHost for DefaultCoreSeerHost {}

static HOST: OnceLock<Arc<dyn CoreSeerHost>> = OnceLock::new();

/// Install the seer core host for this process. Idempotent: only
/// the first call wins, subsequent calls return `Err(host)` so the
/// caller can either drop it or panic. Call once during the host's
/// startup, before constructing any `Player`.
pub fn install(host: Arc<dyn CoreSeerHost>) -> Result<(), Arc<dyn CoreSeerHost>> {
    HOST.set(host)
}

/// The installed seer core host, if any. Patched call sites use
/// this to branch between the seer path and upstream Ruffle:
///
/// ```ignore
/// if let Some(policy) = crate::seer::host()
///     .and_then(|h| h.asset_arena_policy())
/// {
///     // seer path
/// } else {
///     // upstream Ruffle
/// }
/// ```
#[inline]
pub fn host() -> Option<&'static Arc<dyn CoreSeerHost>> {
    HOST.get()
}
