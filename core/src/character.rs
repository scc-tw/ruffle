use std::cell::RefCell;
use std::sync::Arc;
use std::time::Instant;

use crate::backend::audio::SoundHandle;
use crate::binary_data::BinaryData;
use crate::display_object::{
    Avm1Button, Avm2Button, BitmapClass, EditText, Graphic, MorphShape, MovieClip, Text, Video,
};
use crate::font::Font;
use gc_arena::barrier::unlock;
use gc_arena::lock::Lock;
use gc_arena::{Collect, Gc, Mutation};
use ruffle_render::backend::RenderBackend;
use ruffle_render::bitmap::{Bitmap as RenderBitmap, BitmapHandle, BitmapSize};
use ruffle_render::error::Error as RenderError;
use swf::DefineBitsLossless;

#[derive(Copy, Clone, Collect, Debug)]
#[collect(no_drop)]
pub enum Character<'gc> {
    EditText(EditText<'gc>),
    Graphic(Graphic<'gc>),
    MovieClip(MovieClip<'gc>),
    Bitmap(Gc<'gc, BitmapCharacter<'gc>>),
    Avm1Button(Avm1Button<'gc>),
    Avm2Button(Avm2Button<'gc>),
    Font(Font<'gc>),
    MorphShape(MorphShape<'gc>),
    Text(Text<'gc>),
    Sound(#[collect(require_static)] SoundHandle),
    Video(Video<'gc>),
    BinaryData(Gc<'gc, BinaryData>),
}

/// [seer-patch P1] Resettable GPU residency slot for a
/// `BitmapCharacter`. Replaces the upstream `OnceCell<BitmapHandle>`
/// to allow `Player::sweep_idle_bitmaps` to drop the GPU handle
/// while keeping `compressed` alive on CPU heap for cheap re-decode.
///
/// `last_sampled` is bumped from every `bitmap_handle()` call (which
/// is the access path for shape-pattern fills referencing this
/// character). The sweep evicts handles whose `last_sampled` is older
/// than `BitmapResidencyPolicy::idle_threshold`.
#[derive(Default, Debug)]
pub struct BitmapResidency {
    pub handle: Option<BitmapHandle>,
    pub last_sampled: Option<Instant>,
}

#[derive(Collect, Debug)]
#[collect(no_drop)]
pub struct BitmapCharacter<'gc> {
    #[collect(require_static)]
    compressed: CompressedBitmap,
    /// [seer-patch P1] Lazily constructed GPU handle + sampling clock.
    /// Used when performing fills with this bitmap (shape-pattern
    /// fills via `MovieLibrarySource::bitmap_handle`).
    /// `Player::sweep_idle_bitmaps` resets the inner `Option` once
    /// the handle has been idle past the residency policy threshold.
    #[collect(require_static)]
    residency: RefCell<BitmapResidency>,
    /// [seer-patch P1.5] Shared decoded pixel buffer + transparency
    /// flag. First call to `shared_pixels()` decodes `compressed`
    /// once and stores the result here as an `Arc`; subsequent
    /// calls return `Arc::clone` (no re-decode). Every `Bitmap::new`
    /// instantiation from this character's library entry shares the
    /// same `Arc<Vec<Color>>` — AS3 mutations COW via `Arc::make_mut`
    /// on the BitmapData side.
    ///
    /// Stored alongside transparency so callers don't have to
    /// re-derive from `CompressedBitmap`. `Color` is path-qualified
    /// to avoid a `use` dependency on bitmap_data internals.
    #[collect(require_static)]
    pixels_cache: RefCell<
        Option<(
            Arc<Vec<crate::bitmap::bitmap_data::Color>>,
            bool,
        )>,
    >,
    /// The bitmap class set by `SymbolClass` - this is used when we instantaite
    /// a `Bitmap` displayobject.
    avm2_class: Lock<BitmapClass<'gc>>,
}

impl<'gc> BitmapCharacter<'gc> {
    pub fn new(compressed: CompressedBitmap) -> Self {
        Self {
            compressed,
            residency: RefCell::new(BitmapResidency::default()),
            pixels_cache: RefCell::new(None),
            avm2_class: Lock::new(BitmapClass::NoSubclass),
        }
    }

    pub fn compressed(&self) -> &CompressedBitmap {
        &self.compressed
    }

    pub fn avm2_class(&self) -> BitmapClass<'gc> {
        self.avm2_class.get()
    }

    pub fn set_avm2_class(this: Gc<'gc, Self>, bitmap_class: BitmapClass<'gc>, mc: &Mutation<'gc>) {
        unlock!(Gc::write(mc, this), Self, avm2_class).set(bitmap_class);
    }

    pub fn bitmap_handle(
        &self,
        backend: &mut dyn RenderBackend,
    ) -> Result<BitmapHandle, RenderError> {
        // [seer-patch P1] Bump touch + return cached handle if present.
        // Drop borrow before any decode/register call so reentrant
        // `bitmap_handle()` (e.g., from inside a backend hook) doesn't
        // double-borrow.
        {
            let mut r = self.residency.borrow_mut();
            r.last_sampled = Some(Instant::now());
            if let Some(handle) = &r.handle {
                return Ok(handle.clone());
            }
        }
        let decoded = self.compressed.decode()?;
        let new_handle = backend.register_bitmap(decoded)?;
        // Re-borrow on store. If a concurrent reentrant call raced
        // and populated the slot in the meantime (unlikely on the
        // single-threaded AVM2 path, but defensive), prefer the
        // already-stored handle and let our fresh upload drop.
        {
            let mut r = self.residency.borrow_mut();
            if let Some(existing) = &r.handle {
                return Ok(existing.clone());
            }
            r.handle = Some(new_handle.clone());
        }
        Ok(new_handle)
    }

    /// [seer-patch] GPU residency in bytes if the lazy `BitmapHandle`
    /// has been realised, otherwise `None`. Used by
    /// `MovieLibrary::bitmap_bytes` to attribute decoded GPU memory
    /// per source SWF without having to walk the renderer's pools.
    /// Lazy (still compressed) bitmaps don't contribute — they live
    /// in heap and are accounted there.
    pub fn realised_bytes(&self) -> Option<u64> {
        if self.residency.borrow().handle.is_some() {
            let s = self.compressed.size();
            Some(u64::from(s.width) * u64::from(s.height) * 4)
        } else {
            None
        }
    }

    /// [seer-patch P1] Whether the lazy GPU handle is currently realised.
    pub fn is_realised(&self) -> bool {
        self.residency.borrow().handle.is_some()
    }

    /// [seer-patch P1] Wall-clock instant of the last `bitmap_handle()`
    /// call, or `None` if never realised. Used by the residency sweep
    /// to identify idle handles for eviction.
    pub fn last_sampled(&self) -> Option<Instant> {
        self.residency.borrow().last_sampled
    }

    /// [seer-patch P1] Drop the realised GPU handle. Returns `true` if
    /// a handle was present (i.e., the eviction did real work). The
    /// `compressed` source bytes stay alive; the next `bitmap_handle()`
    /// call will re-decode and re-upload.
    ///
    /// `last_sampled` is preserved as a hint for diagnostics — callers
    /// can tell apart "never realised" (None) from "evicted X seconds
    /// ago" (Some(t) with `is_realised() == false`).
    pub fn evict_gpu(&self) -> bool {
        self.residency.borrow_mut().handle.take().is_some()
    }

    /// [seer-patch] Source bytes still held in `CompressedBitmap`
    /// regardless of whether the lazy handle has been realised.
    /// Used by `MovieLibrary::compressed_bitmap_bytes` to attribute
    /// the heap-side cost of un-decoded image source per SWF.
    pub fn compressed_source_bytes(&self) -> u64 {
        self.compressed.source_bytes()
    }

    /// [seer-patch P1.5] Decode `compressed` once, return the result
    /// as a shared `Arc<Vec<Color>>`. First call decodes + populates
    /// the cache; subsequent calls return `Arc::clone` of the cached
    /// vec. Used by `library.rs::instantiate_display_object`'s
    /// `Character::Bitmap` arm so every `Bitmap` display object
    /// instantiated from this character shares one pixel buffer.
    ///
    /// AS3 mutations on the resulting `BitmapData` go through
    /// `Arc::make_mut`, producing an owned copy on first write —
    /// the cache here is unaffected (still shared by other
    /// non-mutated siblings).
    pub fn shared_pixels(
        &self,
    ) -> Result<
        (Arc<Vec<crate::bitmap::bitmap_data::Color>>, bool),
        RenderError,
    > {
        // Fast path: already decoded.
        if let Some((arc, transparent)) = self.pixels_cache.borrow().as_ref() {
            return Ok((Arc::clone(arc), *transparent));
        }
        // Decode. Drop borrow first — `compressed.decode()` doesn't
        // touch `pixels_cache` but be defensive about reentrancy.
        let bitmap = self.compressed.decode()?;
        let transparent = matches!(
            bitmap.format(),
            ruffle_render::bitmap::BitmapFormat::Rgba,
        );
        let pixels: Vec<crate::bitmap::bitmap_data::Color> = bitmap
            .as_colors()
            .map(crate::bitmap::bitmap_data::Color::from)
            .collect();
        let arc = Arc::new(pixels);
        // Store. If a concurrent reentrant call populated in the
        // meantime, prefer the existing entry to keep sharing
        // maximal (single allocation across siblings).
        let mut cache = self.pixels_cache.borrow_mut();
        if let Some((existing, t)) = cache.as_ref() {
            return Ok((Arc::clone(existing), *t));
        }
        *cache = Some((Arc::clone(&arc), transparent));
        Ok((arc, transparent))
    }

    /// [seer-patch P1.5] Drop the cached shared pixel `Arc`. Called
    /// from `Player::sweep_idle_bitmaps` (or future memory-pressure
    /// code) when no `BitmapData` siblings reference the cache and
    /// the character has been idle. Cheap to repopulate from
    /// `compressed` on next use.
    ///
    /// Returns `true` if a cache entry was present.
    pub fn evict_shared_pixels(&self) -> bool {
        self.pixels_cache.borrow_mut().take().is_some()
    }
}

/// Holds a bitmap from an SWF tag, plus the decoded width/height.
/// We avoid decompressing the image until it's actually needed - some pathological SWFS
/// like 'House' have thousands of highly-compressed (mostly empty) bitmaps, which can
/// take over 10GB of ram if we decompress them all during preloading.
#[derive(Clone, Debug)]
pub enum CompressedBitmap {
    Jpeg {
        data: Vec<u8>,
        alpha: Option<Vec<u8>>,
        width: u32,
        height: u32,
    },
    Lossless(DefineBitsLossless<'static>),
}

impl CompressedBitmap {
    /// [seer-patch] Bytes of source (compressed) image data still
    /// held in heap. For JPEGs this includes the alpha plane.
    /// For lossless tags this is the parsed `data` slice from the
    /// `DefineBitsLossless` tag.
    pub fn source_bytes(&self) -> u64 {
        match self {
            CompressedBitmap::Jpeg { data, alpha, .. } => {
                let alpha_len = alpha.as_ref().map(|a| a.len()).unwrap_or(0);
                (data.len() + alpha_len) as u64
            }
            CompressedBitmap::Lossless(define) => define.data.len() as u64,
        }
    }

    pub fn size(&self) -> BitmapSize {
        match self {
            CompressedBitmap::Jpeg { width, height, .. } => BitmapSize {
                width: *width,
                height: *height,
            },
            CompressedBitmap::Lossless(define_bits_lossless) => BitmapSize {
                width: define_bits_lossless.width.into(),
                height: define_bits_lossless.height.into(),
            },
        }
    }
    pub fn decode(&self) -> Result<RenderBitmap<'static>, RenderError> {
        match self {
            CompressedBitmap::Jpeg {
                data,
                alpha,
                width: _,
                height: _,
            } => ruffle_render::utils::decode_define_bits_jpeg(data, alpha.as_deref()),
            CompressedBitmap::Lossless(define_bits_lossless) => {
                ruffle_render::utils::decode_define_bits_lossless(define_bits_lossless)
            }
        }
    }
}
