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
