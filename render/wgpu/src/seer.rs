//! [seer-patch] Host-supplied context for seer-specific behaviour.
//!
//! This module is the single extension point through which the seer
//! launcher steers `ruffle_render_wgpu` away from upstream defaults.
//! It exists *only* in our fork; nothing in Ruffle proper looks at it.
//!
//! ## Shape
//!
//! - [`SeerHost`] is a trait the host crate (typically `seer-flash`)
//!   implements. Each method corresponds to one patched call site
//!   inside the wgpu backend, with a default that reproduces
//!   upstream Ruffle behaviour. Adding a new patch ⇒ adding a new
//!   method with a default ⇒ no breakage for hosts that don't care.
//!
//! - The host is registered once at process start with [`install`].
//!   Patched call sites read the installed host through [`host()`]
//!   and fall through to upstream Ruffle when the slot is empty.
//!
//! - [`Arc<dyn SeerHost>`] is the "ctx handle" — clone-cheap,
//!   thread-safe, hidden behind a single global slot so we don't
//!   have to thread it through every Ruffle constructor.
//!
//! ## Why a trait, not a struct of bools
//!
//! Bools are flat config; a trait lets a hook be a *computation*
//! (e.g. "decide async-readback strategy from current GPU pressure")
//! or a *callback* (e.g. "the host wants to be notified when a
//! buffer is mapped"). The trait shape leaves room for that without
//! a second migration. Hosts that just want flat bools implement
//! one-line overrides.
//!
//! ## Cost when off
//!
//! Each query is one `OnceLock::get` (relaxed atomic load) plus, if
//! installed, one virtual call. With no host installed the wgpu
//! backend is byte-identical to upstream Ruffle.

use std::sync::{Arc, OnceLock};

/// [seer-patch 1.6a] Origin of a `wgpu::Texture` allocation, surfaced
/// to the host so per-source memory gauges can isolate where GPU
/// memory pressure originates.
///
/// Phase 1's `BitmapCensus` only tracked the `Bitmap` source
/// (= `register_bitmap`). Live measurement (vmmap 2026-05-04) showed
/// the bulk of GPU memory was in `BitmapCache` render targets +
/// `FilterPool` intermediates instead — both invisible to
/// `BitmapCensus`. Phase 1.6a adds this source tag so the host can
/// attribute every `wgpu::Texture` to one of the buckets below.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum TextureSource {
    /// `WgpuRenderBackend::register_bitmap` — library bitmaps
    /// (`DefineBits`, `BitmapData::new_with_pixels`). Tracked also
    /// by the legacy `BitmapCensus` gauge for back-compat.
    Bitmap,
    /// `WgpuRenderBackend::create_empty_texture` — render targets
    /// for `displayObject.cacheAsBitmap = true` (or implicit caching
    /// via filters). Phase 1.6c targets these for idle-eviction.
    BitmapCache,
    /// `TexturePool::get_texture` — filter intermediates (blur, glow,
    /// drop shadow) and surface render targets. Phase 1.6b targets
    /// these for LRU cap + idle-purge.
    FilterPool,
    /// `Context3D::create_texture` — Stage3D back/front buffers and
    /// dynamic Context3D textures. Confirmed unused by 賽爾號 in
    /// Phase 1.6d audit, but tagged so we'd notice if a future SWF
    /// starts using Stage3D.
    Context3D,
    /// Pixel bender intermediate textures (`pixel_bender.rs`).
    /// Confirmed unused by 賽爾號; tagged for visibility.
    PixelBender,
    /// Mesh-attached textures, surface backbuffers, and any other
    /// site that doesn't fit the categories above. Small footprint;
    /// included for census reconciliation.
    Other,
}

impl TextureSource {
    /// Stable index into a fixed-size gauge array. Keep in sync with
    /// `TextureCensus::SOURCE_COUNT` on the host side.
    pub fn index(self) -> usize {
        match self {
            TextureSource::Bitmap => 0,
            TextureSource::BitmapCache => 1,
            TextureSource::FilterPool => 2,
            TextureSource::Context3D => 3,
            TextureSource::PixelBender => 4,
            TextureSource::Other => 5,
        }
    }

    /// Short tag for log lines / debug labels.
    pub fn short(self) -> &'static str {
        match self {
            TextureSource::Bitmap => "bitmap",
            TextureSource::BitmapCache => "cache",
            TextureSource::FilterPool => "pool",
            TextureSource::Context3D => "ctx3d",
            TextureSource::PixelBender => "pb",
            TextureSource::Other => "other",
        }
    }
}

/// Host-side seer context. Implementors decide how the seer launcher
/// wants the wgpu backend to deviate from upstream Ruffle.
///
/// Every method has a default that returns the upstream Ruffle
/// behaviour. New patch points should land here as new methods with
/// defaults so existing host implementations keep compiling.
///
/// `Send + Sync + 'static` because the host is stored in a process-
/// global slot accessed from any thread that drives the wgpu
/// backend (in seer-flash today that's a single thread, but the
/// bound is cheap and keeps the door open).
pub trait SeerHost: Send + Sync + 'static {
    /// Skip Ruffle's per-pixel `unmultiply_alpha_rgba` pass during
    /// `capture_frame` readback.
    ///
    /// Return `true` when the consumer of the captured `RgbaImage`
    /// accepts premultiplied alpha directly (Slint's
    /// `Image::from_rgba8_premultiplied`, GPU compositors, etc.).
    /// At 1280×720 the pass costs ~10 ms per readback; skipping it
    /// is the single largest win on the rendered-tick fast path.
    ///
    /// Default: `false` (preserve upstream Ruffle behaviour, where
    /// the consumer expects straight alpha).
    fn skip_unmultiply_on_capture(&self) -> bool {
        false
    }

    /// Notification that a `BitmapHandle` was registered with the
    /// renderer (one wgpu Texture allocated, `bytes` bytes of GPU
    /// memory committed). Fired once per call to
    /// `RenderBackend::register_bitmap`.
    ///
    /// Used by the seer-flash GPU memory monitor to track cumulative
    /// bitmap residency and emit a warning when bitmap-cache pressure
    /// approaches the documented leak threshold (Phase-1 launcher hit
    /// 5 GB of GPU bitmap textures across 200+ live SWFs over a
    /// 218 s session — see `docs/architecture.md` §5b.4).
    ///
    /// Default: no-op (preserve upstream Ruffle behaviour).
    fn on_bitmap_registered(&self, _width: u32, _height: u32, _bytes: u64) {}

    /// Notification that a `BitmapHandle`'s underlying GPU texture
    /// is being dropped. Fires from the wgpu backend's `Texture`
    /// destructor — by that point the wgpu resource has been
    /// signalled for release but Vulkan may still be holding the
    /// underlying device memory until the next command-queue flush.
    ///
    /// Pair with `on_bitmap_registered` to maintain a running census
    /// of live GPU bitmap bytes.
    ///
    /// Default: no-op.
    fn on_bitmap_dropped(&self, _bytes: u64) {}

    /// [seer-patch 1.6a] Notification that a `wgpu::Texture` was
    /// allocated with the given source attribution. Fires once per
    /// `Texture` wrapper construction (and once per `TexturePool`
    /// constructor invocation).
    ///
    /// `Bitmap`-source registrations *also* fire `on_bitmap_registered`
    /// (the two hooks coexist for back-compat); other sources fire
    /// only this hook.
    ///
    /// Default: no-op.
    fn on_texture_registered(&self, _source: TextureSource, _bytes: u64) {}

    /// [seer-patch 1.6a] Notification that a `wgpu::Texture` is
    /// being dropped. Pair with `on_texture_registered` to maintain
    /// per-source live byte gauges.
    ///
    /// Default: no-op.
    fn on_texture_dropped(&self, _source: TextureSource, _bytes: u64) {}
}

/// A trivial host that returns every upstream default. Useful as a
/// starting point for hosts that only want to override one or two
/// hooks: derive your own type and only implement what you change.
pub struct DefaultSeerHost;
impl SeerHost for DefaultSeerHost {}

static HOST: OnceLock<Arc<dyn SeerHost>> = OnceLock::new();

/// Install the seer host for this process. Idempotent: only the
/// first call wins, subsequent calls return `Err(host)` so the
/// caller can either drop it or panic. Call this once during the
/// host's startup, before constructing any `WgpuRenderBackend`.
pub fn install(host: Arc<dyn SeerHost>) -> Result<(), Arc<dyn SeerHost>> {
    HOST.set(host)
}

/// The installed seer host, if any. Patched call sites use this to
/// branch between the seer path and the upstream Ruffle path:
///
/// ```ignore
/// if seer::host().is_some_and(|h| h.skip_unmultiply_on_capture()) {
///     // seer path
/// } else {
///     // upstream Ruffle
/// }
/// ```
#[inline]
pub fn host() -> Option<&'static Arc<dyn SeerHost>> {
    HOST.get()
}
