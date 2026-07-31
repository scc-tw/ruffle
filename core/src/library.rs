use crate::avm1::{PropertyMap as Avm1PropertyMap, PropertyMap};
use crate::avm2::{Class as Avm2Class, Domain as Avm2Domain};
use crate::backend::audio::SoundHandle;
use crate::character::Character;

use crate::display_object::{Bitmap, Graphic, MorphShape, Text};
use crate::font::{Font, FontDescriptor, FontLike, FontQuery, FontType};
use crate::prelude::*;
use crate::string::AvmString;
use crate::tag_utils::SwfMovie;
use gc_arena::collect::Trace;
use gc_arena::{Collect, Mutation};
use ruffle_render::backend::RenderBackend;
use ruffle_render::bitmap::BitmapHandle;
use ruffle_render::utils::remove_invalid_jpeg_data;

use crate::backend::ui::{FontDefinition, UiBackend};
use crate::font::DefaultFont;
use fnv::{FnvHashMap, FnvHashSet};
use std::cell::Cell;
use std::collections::HashMap;
use std::sync::{Arc, Weak};
use std::time::{Duration, Instant};
use weak_table::{PtrWeakKeyHashMap, WeakValueHashMap, traits::WeakElement};

#[derive(Clone)]
struct MovieSymbol(Arc<SwfMovie>, CharacterId);

#[derive(Clone)]
struct WeakMovieSymbol(Weak<SwfMovie>, CharacterId);

impl WeakElement for WeakMovieSymbol {
    type Strong = MovieSymbol;

    fn new(view: &Self::Strong) -> Self {
        Self(Arc::downgrade(&view.0), view.1)
    }

    fn view(&self) -> Option<Self::Strong> {
        if let Some(strong) = self.0.upgrade() {
            Some(MovieSymbol(strong, self.1))
        } else {
            None
        }
    }
}

/// The mappings between class objects and library characters defined by
/// `SymbolClass`.
pub struct Avm2ClassRegistry<'gc> {
    /// A list of AVM2 class objects and the character IDs they are expected to
    /// instantiate.
    class_map: WeakValueHashMap<Avm2Class<'gc>, WeakMovieSymbol>,
}

unsafe impl<'gc> Collect<'gc> for Avm2ClassRegistry<'gc> {
    fn trace<C: Trace<'gc>>(&self, cc: &mut C) {
        for (k, _) in self.class_map.iter() {
            cc.trace(k);
        }
    }
}

impl Default for Avm2ClassRegistry<'_> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'gc> Avm2ClassRegistry<'gc> {
    pub fn new() -> Self {
        Self {
            class_map: WeakValueHashMap::new(),
        }
    }

    /// Retrieve the library symbol for a given AVM2 class object.
    ///
    /// A value of `None` indicates that this AVM2 class is not associated with
    /// a library symbol.
    pub fn class_symbol(&self, class_def: Avm2Class<'gc>) -> Option<(Arc<SwfMovie>, CharacterId)> {
        match self.class_map.get(&class_def) {
            Some(MovieSymbol(movie, symbol)) => Some((movie, symbol)),
            None => None,
        }
    }

    /// Associate an AVM2 class definition with a given library symbol.
    pub fn set_class_symbol(
        &mut self,
        class_def: Avm2Class<'gc>,
        movie: Arc<SwfMovie>,
        symbol: CharacterId,
    ) {
        if let Some(old) = self.class_map.get(&class_def) {
            if Arc::ptr_eq(&movie, &old.0) && symbol != old.1 {
                // Flash player actually allows using the same class in multiple SymbolClass
                // entries in the same swf, with *different* symbol ids. Whichever one
                // is processed first will *win*, and the second one will be ignored.
                // We still log a warning, since we wouldn't expect this to happen outside
                // of deliberately crafted SWFs.
                tracing::warn!(
                    "Tried to overwrite class {:?} id={:?} with symbol id={:?} from same movie",
                    class_def,
                    old.1,
                    symbol,
                );
            }
            // If we're trying to overwrite the class with a symbol from a *different* SwfMovie,
            // then just ignore it. This handles the case where a Loader has a class that shadows
            // a class in the main swf (possibly with a different ApplicationDomain). This will
            // result in the original class from the parent being used, even when the child swf
            // instantiates the clip on the timeline.
            return;
        }
        self.class_map.insert(class_def, MovieSymbol(movie, symbol));
    }
}

/// [seer-patch] Per-variant character counts produced by
/// `MovieLibrary::character_counts`. The rest of `Character`
/// variants (Avm1Button, Avm2Button, Video, BinaryData) fold into
/// `other`; they're either rare or carry negligible bytes.
#[derive(Debug, Default, Clone, Copy)]
pub struct CharacterCounts {
    pub bitmap: usize,
    pub shape: usize,
    pub sprite: usize,
    pub font: usize,
    pub text: usize,
    pub sound: usize,
    pub other: usize,
}

/// Symbol library for a single given SWF.
#[derive(Collect)]
#[collect(no_drop)]
pub struct MovieLibrary<'gc> {
    swf: Arc<SwfMovie>,
    characters: HashMap<CharacterId, Character<'gc>>,
    export_characters: Avm1PropertyMap<'gc, CharacterId>,
    imported_assets: HashMap<AvmString<'gc>, CharacterId>,
    jpeg_tables: Option<Vec<u8>>,
    fonts: FontMap<'gc>,
    avm2_domain: Option<Avm2Domain<'gc>>,
    /// [seer-patch] Wall-clock at last character lookup. Bumped from
    /// every `character_by_*` / `instantiate_*` / `get_*` path so
    /// `Library::abandonable_movies` can find idle SWFs whose
    /// `MovieLibrary` is safe to evict (releasing every decoded
    /// `BitmapHandle` / Font glyph atlas / etc. owned by it).
    /// `Cell` is fine because gc-arena is single-threaded.
    #[collect(require_static)]
    last_touch: Cell<Instant>,
}

impl<'gc> MovieLibrary<'gc> {
    pub fn new(swf: Arc<SwfMovie>) -> Self {
        Self {
            swf,
            characters: HashMap::new(),
            imported_assets: HashMap::new(),
            export_characters: Avm1PropertyMap::new(),
            jpeg_tables: None,
            fonts: Default::default(),
            avm2_domain: None,
            last_touch: Cell::new(Instant::now()),
        }
    }

    /// [seer-patch] Bump the touch clock. Called from every lookup
    /// path so an active SWF stays "warm" and out of
    /// `Library::abandonable_movies`'s output.
    #[inline]
    fn touch(&self) {
        self.last_touch.set(Instant::now());
    }

    /// [seer-patch] When this library was last accessed.
    pub fn last_touch(&self) -> Instant {
        self.last_touch.get()
    }

    /// [seer-patch] The source SWF this library was parsed from.
    pub fn swf(&self) -> &Arc<SwfMovie> {
        &self.swf
    }

    /// [seer-patch] Sum of decoded GPU bytes (width × height × 4)
    /// across every `BitmapCharacter` in this library whose lazy
    /// handle has been realised. Bitmaps still in compressed form
    /// don't contribute (no GPU memory committed yet).
    pub fn bitmap_bytes(&self) -> u64 {
        let mut total = 0u64;
        for ch in self.characters.values() {
            if let Character::Bitmap(bc) = ch
                && let Some(b) = bc.realised_bytes()
            {
                total = total.saturating_add(b);
            }
        }
        total
    }

    /// [seer-patch] Sum of compressed source bytes
    /// (`CompressedBitmap::Jpeg.data + alpha`, or
    /// `Lossless.data`) across every `BitmapCharacter`. Lives on
    /// CPU heap regardless of whether the GPU handle has been
    /// realised, so this number is **additive** with
    /// `bitmap_bytes`, not exclusive: a bitmap currently on GPU
    /// also still has its compressed source in heap.
    pub fn compressed_bitmap_bytes(&self) -> u64 {
        let mut total = 0u64;
        for ch in self.characters.values() {
            if let Character::Bitmap(bc) = ch {
                total = total.saturating_add(bc.compressed_source_bytes());
            }
        }
        total
    }

    /// [seer-patch] Length of the raw SWF source bytes the library
    /// keeps alive via its `Arc<SwfMovie>`. CPU heap.
    pub fn swf_data_bytes(&self) -> u64 {
        self.swf.data().len() as u64
    }

    /// [seer-patch] Per-variant character counts. Useful for
    /// triangulating where non-bitmap memory lives (e.g. lots of
    /// `Graphic`s ⇒ tessellation cache pressure; lots of
    /// `MovieClip`s ⇒ AVM2 vtable / timeline weight).
    pub fn character_counts(&self) -> CharacterCounts {
        let mut c = CharacterCounts::default();
        for ch in self.characters.values() {
            match ch {
                Character::Bitmap(_) => c.bitmap += 1,
                Character::Graphic(_) | Character::MorphShape(_) => c.shape += 1,
                Character::MovieClip(_) => c.sprite += 1,
                Character::Font(_) => c.font += 1,
                Character::Text(_) | Character::EditText(_) => c.text += 1,
                Character::Sound(_) => c.sound += 1,
                _ => c.other += 1,
            }
        }
        c
    }

    /// [seer-patch] Count of `Character::Sound` entries. A non-zero
    /// count blocks eviction when
    /// `AssetArenaPolicy::never_evict_with_audio` is set, since
    /// dropping the library would release the registered
    /// `SoundHandle` and could cut mid-playback audio.
    pub fn sound_count(&self) -> usize {
        self.characters
            .values()
            .filter(|c| matches!(c, Character::Sound(_)))
            .count()
    }

    /// Drop GPU shape handles that have not been rendered recently while
    /// retaining all SWF character and AVM2 domain state.
    pub fn evict_idle_shape_handles(&self, now: Instant, idle_threshold: Duration) -> usize {
        self.characters
            .values()
            .map(|character| match character {
                Character::Graphic(graphic) => {
                    usize::from(graphic.evict_shape_if_idle(now, idle_threshold))
                }
                Character::MorphShape(morph) => morph.evict_shapes_if_idle(now, idle_threshold),
                _ => 0,
            })
            .sum()
    }

    /// Registers a character; returns `true` if successful, or `false` if a character with
    /// the given ID already exists.
    pub fn register_character(&mut self, id: CharacterId, character: Character<'gc>) -> bool {
        use std::collections::hash_map::Entry;
        // [seer-patch] Treat registration as a touch — a freshly
        // parsed library is warm. If the SWF is loaded but never
        // displayed, the idle clock starts ticking from here.
        self.touch();
        match self.characters.entry(id) {
            Entry::Vacant(e) => {
                if let Character::Font(font) = character {
                    self.fonts.register(font);
                }
                e.insert(character);
                true
            }
            Entry::Occupied(_) => {
                tracing::error!("Character ID collision: Tried to register ID {} twice", id);
                false
            }
        }
    }

    /// Registers an export name for a given character ID.
    /// This character will then be instantiable from AVM1.
    pub fn register_export(&mut self, id: CharacterId, export_name: AvmString<'gc>) {
        let character_exists = self.contains_character(id);
        debug_assert!(character_exists);
        if !character_exists {
            tracing::error!(
                "Tried to register export '{export_name}' for a non-existent character {id}"
            );
            return;
        }

        self.export_characters.insert(export_name, id, false);
    }

    pub fn characters(&self) -> &HashMap<CharacterId, Character<'gc>> {
        &self.characters
    }

    pub fn export_characters(&self) -> &PropertyMap<'gc, CharacterId> {
        &self.export_characters
    }

    pub fn contains_character(&self, id: CharacterId) -> bool {
        self.characters.contains_key(&id)
    }

    pub fn character_by_id(&self, id: CharacterId) -> Option<Character<'gc>> {
        self.touch();
        self.characters.get(&id).copied()
    }

    pub fn character_by_export_name(
        &self,
        name: AvmString<'gc>,
    ) -> Option<(CharacterId, Character<'gc>)> {
        self.touch();
        if let Some(id) = self.export_characters.get(name, false)
            && let Some(character) = self.characters.get(id)
        {
            return Some((*id, *character));
        }
        None
    }

    pub fn character_id_by_import_name(&self, name: AvmString<'gc>) -> Option<CharacterId> {
        self.touch();
        self.imported_assets.get(&name).copied()
    }

    pub fn register_import(&mut self, name: AvmString<'gc>, id: CharacterId) {
        self.imported_assets.insert(name, id);
    }

    /// Instantiates the library item with the given character ID into a display object.
    /// The object must then be post-instantiated before being used.
    pub fn instantiate_by_id(
        &self,
        id: CharacterId,
        mc: &Mutation<'gc>,
    ) -> Option<DisplayObject<'gc>> {
        self.touch();
        if let Some(&character) = self.characters.get(&id) {
            self.instantiate_display_object(id, character, mc)
        } else {
            tracing::error!("Tried to instantiate non-registered character ID {}", id);
            None
        }
    }

    /// Instantiates the library item with the given export name into a display object.
    /// The object must then be post-instantiated before being used.
    pub fn instantiate_by_export_name(
        &self,
        export_name: AvmString<'gc>,
        mc: &Mutation<'gc>,
    ) -> Option<DisplayObject<'gc>> {
        // `character_by_export_name` already touches — no double bump.
        if let Some((id, character)) = self.character_by_export_name(export_name) {
            self.instantiate_display_object(id, character, mc)
        } else {
            tracing::error!(
                "Tried to instantiate non-registered character {}",
                export_name
            );
            None
        }
    }

    /// Instantiates the given character into a display object.
    /// The object must then be post-instantiated before being used.
    fn instantiate_display_object(
        &self,
        id: CharacterId,
        character: Character<'gc>,
        mc: &Mutation<'gc>,
    ) -> Option<DisplayObject<'gc>> {
        match character {
            Character::Bitmap(bitmap) => {
                let avm2_class = bitmap.avm2_class();
                // [seer-patch P1.5] Use the character's shared pixel
                // Arc instead of decoding fresh per instantiation.
                // First call decodes once and caches; sibling
                // instantiations share via Arc::clone (no decode,
                // no pixel-Vec duplication). AS3 mutations COW.
                let (pixels, transparency) = bitmap.shared_pixels().unwrap();
                let size = bitmap.compressed().size();
                let bitmap_do = Bitmap::new_with_shared_pixels(
                    mc,
                    id,
                    size.width,
                    size.height,
                    transparency,
                    pixels,
                    &self.swf,
                );
                bitmap_do.set_avm2_bitmapdata_class(mc, avm2_class);
                Some(bitmap_do.instantiate(mc))
            }
            Character::EditText(edit_text) => Some(edit_text.instantiate(mc)),
            Character::Graphic(graphic) => Some(graphic.instantiate(mc)),
            Character::MorphShape(morph_shape) => Some(morph_shape.instantiate(mc)),
            Character::MovieClip(movie_clip) => Some(movie_clip.instantiate(mc)),
            Character::Avm1Button(button) => Some(button.instantiate(mc)),
            Character::Avm2Button(button) => Some(button.instantiate(mc)),
            Character::Text(text) => Some(text.instantiate(mc)),
            Character::Video(video) => Some(video.instantiate(mc)),
            _ => {
                // Cannot instantiate non-display object
                None
            }
        }
    }

    pub fn get_font(&self, id: CharacterId) -> Option<Font<'gc>> {
        self.touch();
        if let Some(&Character::Font(font)) = self.characters.get(&id) {
            Some(font)
        } else {
            None
        }
    }

    pub fn embedded_fonts(&self) -> Vec<Font<'gc>> {
        // [seer-patch] AS3 `Font.enumerateFonts(false)` lands here.
        // Treat as a live use so font-heavy libs don't false-idle.
        self.touch();
        self.fonts.all()
    }

    /// Returns the `Graphic` with the given character ID.
    /// Returns `None` if the ID does not exist or is not a `Graphic`.
    pub fn get_graphic(&self, id: CharacterId) -> Option<Graphic<'gc>> {
        self.touch();
        if let Some(&Character::Graphic(graphic)) = self.characters.get(&id) {
            Some(graphic)
        } else {
            None
        }
    }

    /// Returns the `MorphShape` with the given character ID.
    /// Returns `None` if the ID does not exist or is not a `MorphShape`.
    pub fn get_morph_shape(&self, id: CharacterId) -> Option<MorphShape<'gc>> {
        self.touch();
        if let Some(&Character::MorphShape(morph_shape)) = self.characters.get(&id) {
            Some(morph_shape)
        } else {
            None
        }
    }

    pub fn get_sound(&self, id: CharacterId) -> Option<SoundHandle> {
        self.touch();
        if let Some(Character::Sound(sound)) = self.characters.get(&id) {
            Some(*sound)
        } else {
            None
        }
    }

    /// Returns the `Text` with the given character ID.
    /// Returns `None` if the ID does not exist or is not a `Text`.
    pub fn get_text(&self, id: CharacterId) -> Option<Text<'gc>> {
        self.touch();
        if let Some(&Character::Text(text)) = self.characters.get(&id) {
            Some(text)
        } else {
            None
        }
    }

    pub fn set_jpeg_tables(&mut self, data: &[u8]) {
        if self.jpeg_tables.is_some() {
            // SWF spec says there should only be one JPEGTables tag.
            // TODO: What is the behavior when there are multiples?
            tracing::warn!("SWF contains multiple JPEGTables tags");
            return;
        }
        // Some SWFs have a JPEGTables tag with 0 length; ignore these.
        // (Does this happen when there is only a single DefineBits tag?)
        self.jpeg_tables = if data.is_empty() {
            None
        } else {
            Some(remove_invalid_jpeg_data(data).to_vec())
        }
    }

    pub fn jpeg_tables(&self) -> Option<&[u8]> {
        self.jpeg_tables.as_ref().map(|data| &data[..])
    }

    pub fn set_avm2_domain(&mut self, avm2_domain: Avm2Domain<'gc>) {
        self.avm2_domain = Some(avm2_domain);
    }

    /// Get the AVM2 domain this movie runs under.
    ///
    /// Note that the presence of an AVM2 domain does *not* indicate that this
    /// movie provides AVM2 code. For example, a movie may have been loaded by
    /// AVM2 code into a particular domain, even though it turned out to be
    /// an AVM1 movie, and thus this domain is unused.
    pub fn avm2_domain(&self) -> Avm2Domain<'gc> {
        // [seer-patch] AVM2 hot path: every `loaderInfo.applicationDomain`
        // access and every class resolution walks through here. Without
        // this touch, an actively-running AS3 SWF can be falsely judged
        // idle by `Library::abandonable_movies` while it is still
        // resolving classes from its own domain.
        self.touch();
        self.avm2_domain.unwrap()
    }

    pub fn try_avm2_domain(&self) -> Option<Avm2Domain<'gc>> {
        // [seer-patch] Same reasoning as `avm2_domain`. Bump the touch
        // clock unconditionally — even a `None` return is a probe that
        // proves the library is still being interrogated.
        self.touch();
        self.avm2_domain
    }
}

pub struct MovieLibrarySource<'a, 'gc> {
    pub library: &'a MovieLibrary<'gc>,
}

impl ruffle_render::bitmap::BitmapSource for MovieLibrarySource<'_, '_> {
    fn bitmap_size(&self, id: u16) -> Option<ruffle_render::bitmap::BitmapSize> {
        if let Some(Character::Bitmap(bitmap)) = self.library.characters.get(&id) {
            Some(bitmap.compressed().size())
        } else {
            None
        }
    }

    fn bitmap_handle(&self, id: u16, backend: &mut dyn RenderBackend) -> Option<BitmapHandle> {
        let Some(Character::Bitmap(bitmap)) = self.library.characters.get(&id) else {
            return None;
        };

        match bitmap.bitmap_handle(backend) {
            Ok(handle) => Some(handle),
            Err(e) => {
                tracing::error!("Failed to register bitmap character {id}: {e}");
                None
            }
        }
    }
}

struct MovieLibraries<'gc>(PtrWeakKeyHashMap<Weak<SwfMovie>, MovieLibrary<'gc>>);

unsafe impl<'gc> Collect<'gc> for MovieLibraries<'gc> {
    #[inline]
    fn trace<C: Trace<'gc>>(&self, cc: &mut C) {
        for (_, val) in self.0.iter() {
            cc.trace(val);
        }
    }
}

impl<'gc> MovieLibraries<'gc> {
    fn new() -> Self {
        Self(PtrWeakKeyHashMap::new())
    }

    fn get(&self, key: &Arc<SwfMovie>) -> Option<&MovieLibrary<'gc>> {
        self.0.get(key)
    }

    fn get_or_insert_mut(&mut self, movie: Arc<SwfMovie>) -> &mut MovieLibrary<'gc> {
        self.0
            .entry(movie.clone())
            .or_insert_with(|| MovieLibrary::new(movie))
    }

    fn known_movies(&self) -> impl Iterator<Item = Arc<SwfMovie>> {
        self.0.keys()
    }

    /// [seer-patch] Iterate `(Arc<SwfMovie>, &MovieLibrary)` pairs.
    /// `PtrWeakKeyHashMap::iter` upgrades each weak key for us; the
    /// resulting `Arc` does *not* extend lifetime past upstream's
    /// existing `Arc<SwfMovie>` strong refs (the library always
    /// holds one already), so iteration is just bookkeeping.
    fn iter(&self) -> impl Iterator<Item = (Arc<SwfMovie>, &MovieLibrary<'gc>)> + '_ {
        self.0.iter()
    }

    /// [seer-patch] Remove the entry whose key Arc is pointer-equal
    /// to `key`. Returning the value lets callers inspect / move it
    /// before drop if they want. Used by `Library::remove_movie`.
    fn remove(&mut self, key: &Arc<SwfMovie>) -> Option<MovieLibrary<'gc>> {
        self.0.remove(key)
    }
}

/// Symbol library for multiple movies.
#[derive(Collect)]
#[collect(no_drop)]
pub struct Library<'gc> {
    /// All the movie libraries.
    movie_libraries: MovieLibraries<'gc>,

    /// A cache of seen device fonts.
    // TODO: Descriptors shouldn't be stored in fonts. Fonts should be a list that we iterate and ask "do you match". A font can have zero or many names.
    device_fonts: FontMap<'gc>,

    /// "Global" embedded fonts, registered through AVM2 `Font.registerFont`.
    /// These should be checked before any Movie-specific library's own fonts.
    global_fonts: FontMap<'gc>,

    /// A set of which fonts we've asked from the backend already, to help with negative caching.
    /// If we've asked for a specific font, record it here and don't ask again.
    font_lookup_cache: FnvHashSet<FontQuery>,

    /// Cached font sort queries.
    font_sort_cache: FnvHashMap<FontQuery, Vec<Font<'gc>>>,

    /// The implementation names of each default font.
    default_font_names: FnvHashMap<DefaultFont, Vec<String>>,

    /// The cached list of implementations per default font.
    default_font_cache: FnvHashMap<(DefaultFont, bool, bool), Vec<Font<'gc>>>,

    /// A list of the symbols associated with specific AVM2 constructor
    /// prototypes.
    avm2_class_registry: Avm2ClassRegistry<'gc>,
}

impl<'gc> Library<'gc> {
    pub fn empty() -> Self {
        Self {
            movie_libraries: MovieLibraries::new(),
            device_fonts: Default::default(),
            global_fonts: Default::default(),
            font_lookup_cache: Default::default(),
            font_sort_cache: Default::default(),
            default_font_names: Default::default(),
            default_font_cache: Default::default(),
            avm2_class_registry: Default::default(),
        }
    }

    pub fn library_for_movie(&self, movie: Arc<SwfMovie>) -> Option<&MovieLibrary<'gc>> {
        self.movie_libraries.get(&movie)
    }

    pub fn library_for_movie_mut(&mut self, movie: Arc<SwfMovie>) -> &mut MovieLibrary<'gc> {
        self.movie_libraries.get_or_insert_mut(movie)
    }

    pub fn known_movies(&self) -> impl Iterator<Item = Arc<SwfMovie>> {
        self.movie_libraries.known_movies()
    }

    /// Returns the default Font implementations behind the built in names (ie `_sans`)
    pub fn default_font(
        &mut self,
        name: DefaultFont,
        is_bold: bool,
        is_italic: bool,
        ui: &dyn UiBackend,
        renderer: &mut dyn RenderBackend,
        gc_context: &Mutation<'gc>,
    ) -> Vec<Font<'gc>> {
        // Can't use entry api here as we want to use self for `load_device_font`.
        // Cache the value as this will be looked up a lot, and font lookup by name can be expensive if lots of fonts exist.
        if let Some(cache) = self.default_font_cache.get(&(name, is_bold, is_italic)) {
            return cache.clone();
        }

        let mut result = vec![];
        // First try to find any exactly matching fonts.
        for name in self.default_font_names.entry(name).or_default().clone() {
            let query = FontQuery::new(FontType::Device, name, is_bold, is_italic);
            if let Some(font) = self.get_or_load_exact_device_font(&query, ui, renderer, gc_context)
            {
                result.push(font);
                break; // TODO: Return multiple fonts when it's needed.
            }
        }

        // Nothing found, try a compatible font.
        if result.is_empty() {
            for name in self.default_font_names.entry(name).or_default().clone() {
                let query = FontQuery::new(FontType::Device, name, is_bold, is_italic);
                if let Some(font) = self.device_fonts.find(&query) {
                    result.push(font);
                    break; // TODO: Return multiple fonts when it's needed.
                }
            }
        }

        self.default_font_cache
            .insert((name, is_bold, is_italic), result.clone());
        result
    }

    /// Returns the device font exactly matching the requested options.
    fn get_or_load_exact_device_font(
        &mut self,
        query: &FontQuery,
        ui: &dyn UiBackend,
        renderer: &mut dyn RenderBackend,
        gc_context: &Mutation<'gc>,
    ) -> Option<Font<'gc>> {
        // If we have the exact matching font already, use that
        // TODO: We should instead ask each font if it matches a given name. Partial matches are allowed, and fonts may have any amount of names.
        if let Some(font) = self.device_fonts.get(query) {
            return Some(*font);
        }

        // We don't have this font already. Did we ask for it before?
        let new_request = self.font_lookup_cache.insert(query.clone());
        if new_request {
            // First time asking for this font, see if our backend can provide anything relevant
            ui.load_device_font(query, &mut |definition| {
                self.register_device_font(gc_context, renderer, definition)
            });

            // Check again. A backend may or may not have provided some new fonts,
            // and they may or may not be relevant to the one we're asking for.
            if let Some(font) = self.device_fonts.get(query) {
                return Some(*font);
            }

            let name = &query.name;
            let is_bold = query.is_bold;
            let is_italic = query.is_italic;
            warn!("Unknown device font \"{name}\" (bold: {is_bold}, italic: {is_italic})");
        }

        None
    }

    /// Returns the device font compatible with the requested options.
    pub fn get_or_load_device_font(
        &mut self,
        name: &str,
        is_bold: bool,
        is_italic: bool,
        ui: &dyn UiBackend,
        renderer: &mut dyn RenderBackend,
        gc_context: &Mutation<'gc>,
    ) -> Option<Font<'gc>> {
        let query = FontQuery::new(FontType::Device, name.to_owned(), is_bold, is_italic);

        // Try to find an exactly matching font.
        if let Some(font) = self.get_or_load_exact_device_font(&query, ui, renderer, gc_context) {
            return Some(font);
        }

        // Fallback: Try to find an existing font to re-use instead of giving up.
        self.device_fonts.find(&query)
    }

    fn sort_device_fonts(
        &mut self,
        query: &FontQuery,
        ui: &dyn UiBackend,
        renderer: &mut dyn RenderBackend,
        gc_context: &Mutation<'gc>,
    ) -> Vec<Font<'gc>> {
        // First, ask the backend to sort the fonts for us.
        let fonts = ui.sort_device_fonts(query, &mut |definition| {
            self.register_device_font(gc_context, renderer, definition)
        });

        let fonts: Vec<Font<'gc>> = fonts
            .iter()
            .filter_map(|font_query| self.device_fonts.get(font_query))
            .copied()
            .collect();

        if !fonts.is_empty() {
            return fonts;
        }

        // When the backend failed (or doesn't support sorting fonts), fall back
        // to loading one font only without sorting.
        let font = self.get_or_load_device_font(
            &query.name,
            query.is_bold,
            query.is_italic,
            ui,
            renderer,
            gc_context,
        );
        font.map(|font| vec![font]).unwrap_or_default()
    }

    pub fn get_or_sort_device_fonts(
        &mut self,
        name: &str,
        is_bold: bool,
        is_italic: bool,
        ui: &dyn UiBackend,
        renderer: &mut dyn RenderBackend,
        gc_context: &Mutation<'gc>,
    ) -> Vec<Font<'gc>> {
        // TODO We should be able to return a &Vec here, but (1) the borrow
        //   checker is too strict and doesn't allow if branching, and
        //   (2) there's no way to insert a value and get a reference to
        //   it at the same time.
        let query = FontQuery::new(FontType::Device, name.to_owned(), is_bold, is_italic);
        if let Some(fonts) = self.font_sort_cache.get(&query) {
            return fonts.clone();
        }

        let fonts = self.sort_device_fonts(&query, ui, renderer, gc_context);
        self.font_sort_cache.insert(query, fonts.clone());
        fonts
    }

    pub fn set_default_font(&mut self, font: DefaultFont, names: Vec<String>) {
        self.default_font_names.insert(font, names);
        self.default_font_cache.clear();
    }

    pub fn register_device_font(
        &mut self,
        gc_context: &Mutation<'gc>,
        renderer: &mut dyn RenderBackend,
        definition: FontDefinition<'_>,
    ) {
        match definition {
            FontDefinition::SwfTag(tag, encoding) => {
                let font =
                    Font::from_swf_tag(gc_context, renderer, tag, encoding, FontType::Device);
                let name = font.descriptor().name().to_owned();
                let is_bold = font.descriptor().bold();
                let is_italic = font.descriptor().italic();
                tracing::debug!(
                    "Loaded new device font \"{name}\" (bold: {is_bold}, italic: {is_italic}) from swf tag"
                );
                self.device_fonts.register(font);
            }
            FontDefinition::FontFile {
                name,
                is_bold,
                is_italic,
                data,
                index,
            } => {
                let descriptor = FontDescriptor::from_parts(&name, is_bold, is_italic);
                if let Ok(font) =
                    Font::from_font_file(gc_context, descriptor, data, index, FontType::Device)
                {
                    let name = font.descriptor().name().to_owned();
                    tracing::debug!(
                        "Loaded new device font \"{name}\" (bold: {is_bold}, italic: {is_italic}) from file"
                    );
                    self.device_fonts.register(font);
                } else {
                    warn!("Failed to load device font from file");
                }
            }
            FontDefinition::ExternalRenderer {
                name,
                is_bold,
                is_italic,
                font_renderer,
            } => {
                let descriptor = FontDescriptor::from_parts(&name, is_bold, is_italic);
                let font = Font::from_renderer(gc_context, descriptor, font_renderer);
                tracing::debug!(
                    "Loaded new externally rendered font \"{name}\" (bold: {is_bold}, italic: {is_italic})"
                );
                self.device_fonts.register(font);
            }
        }
        self.default_font_cache.clear();
    }

    /// Find a font by it's name and parameters.
    pub fn get_embedded_font_by_name(
        &self,
        name: &str,
        font_type: FontType,
        is_bold: bool,
        is_italic: bool,
        movie: Option<Arc<SwfMovie>>,
    ) -> Option<Font<'gc>> {
        let query = FontQuery::new(font_type, name.to_owned(), is_bold, is_italic);
        if let Some(font) = self.global_fonts.find(&query) {
            return Some(font);
        }
        if let Some(movie) = movie
            && let Some(library) = self.library_for_movie(movie)
            && let Some(font) = library.fonts.find(&query)
        {
            return Some(font);
        }
        None
    }

    pub fn global_fonts(&self) -> Vec<Font<'gc>> {
        self.global_fonts.all()
    }

    pub fn register_global_font(&mut self, font: Font<'gc>) {
        self.global_fonts.register(font);
    }

    /// Get the AVM2 class registry.
    pub fn avm2_class_registry(&self) -> &Avm2ClassRegistry<'gc> {
        &self.avm2_class_registry
    }

    /// Mutate the AVM2 class registry.
    pub fn avm2_class_registry_mut(&mut self) -> &mut Avm2ClassRegistry<'gc> {
        &mut self.avm2_class_registry
    }

    /// [seer-patch] Snapshot of every `MovieLibrary`'s memory
    /// footprint. Cheap (walks each library's character HashMap
    /// once); intended for the periodic
    /// `CoreSeerHost::on_library_report` callback.
    pub fn library_entries(&self) -> Vec<crate::seer::LibraryEntry> {
        let now = Instant::now();
        let mut out = Vec::with_capacity(self.movie_libraries.0.len());
        for (swf_arc, lib) in self.movie_libraries.iter() {
            let url = url_of_swf(&swf_arc);
            // Subtract:
            //   1 for the Arc we just upgraded from the WeakKey,
            //   1 for `MovieLibrary::swf` (always held strong).
            // What remains *includes both* live DisplayObject
            // clones and AVM2-internal clones held by Scripts in
            // `lib.avm2_domain` — this number is diagnostic only
            // and is **not** the eviction gate (that uses display
            // tree presence; see `abandonable_movies`).
            let strong = Arc::strong_count(&swf_arc).saturating_sub(2);
            let counts = lib.character_counts();
            out.push(crate::seer::LibraryEntry {
                url,
                bitmap_bytes: lib.bitmap_bytes(),
                compressed_bitmap_bytes: lib.compressed_bitmap_bytes(),
                swf_data_bytes: lib.swf_data_bytes(),
                character_count: lib.characters.len(),
                shape_count: counts.shape,
                sprite_count: counts.sprite,
                font_count: counts.font,
                text_count: counts.text,
                sound_count: counts.sound,
                idle_for: now.duration_since(lib.last_touch.get()),
                external_refs: strong,
            });
        }
        out
    }

    /// [seer-patch] Identify `MovieLibrary` entries safe to evict.
    /// "Safe" means three things, all required:
    ///
    ///  1. **Not referenced by any live DisplayObject.** Caller
    ///     supplies `live_movies`, the set of `SwfMovie` pointers
    ///     reachable from the stage's render tree (see
    ///     `Player::collect_live_movies`). If our library's movie
    ///     is in that set, an active MovieClip / Bitmap / etc. is
    ///     still mounted and a future `place_object` tag could
    ///     resolve a character ID — eviction would dangle.
    ///
    ///     **Note on liveness signal choice.** An earlier version of
    ///     this check used `Arc::strong_count(&swf) > 2` (assuming
    ///     "extra refs ⇒ external holder"). That was wrong for AVM2
    ///     SWFs: `lib.avm2_domain` is a `Gc<DomainData>` whose
    ///     `defs: PropertyMap<Script>` contains Scripts that each
    ///     hold an `Arc<SwfMovie>` clone (via `TranslationUnit`).
    ///     Every AS3 SWF therefore has `strong_count = 2 + N` where
    ///     N is the number of scripts loaded — strictly > 2 — and
    ///     no AVM2 SWF was ever evictable. The display-tree set is
    ///     immune to that internal cycle.
    ///
    ///  2. **Idle for at least `idle_threshold`.** Prevents us from
    ///     evicting between two consecutive uses inside the same
    ///     frame (e.g. two `Loader.load`s that resolve to the
    ///     same library).
    ///
    ///  3. **No `Sound` characters when `audio_guard`.** Dropping
    ///     a library with a registered `SoundHandle` could cut
    ///     mid-playback audio, depending on the audio backend's
    ///     handle lifecycle.
    ///
    /// Returns the SwfMovie key for each evictable library so the
    /// caller can pass it to `remove_movie` in a separate pass
    /// (we don't mutate during iteration).
    pub fn abandonable_movies(
        &self,
        live_movies: &FnvHashSet<*const SwfMovie>,
        idle_threshold: Duration,
        audio_guard: bool,
    ) -> Vec<Arc<SwfMovie>> {
        let now = Instant::now();
        let mut out = Vec::new();
        for (swf_arc, lib) in self.movie_libraries.iter() {
            // Display-tree presence is the authoritative
            // "in-use" signal — see method doc above.
            if live_movies.contains(&Arc::as_ptr(&swf_arc)) {
                continue;
            }
            if now.duration_since(lib.last_touch.get()) < idle_threshold {
                continue;
            }
            if audio_guard && lib.sound_count() > 0 {
                continue;
            }
            // [seer-patch B1 2026-05-05] Skip libraries with a set
            // AVM2 ApplicationDomain. Reason: AS3 code outside the
            // stage display tree (e.g. PetAssetsManager /
            // SkillAssetsManager in PetFightDLL_201308) caches
            // `Class` references obtained from `applicationDomain`.
            // Each such Class is bound to a SymbolClass in the
            // owning movie's library; instantiating it produces a
            // `MovieClip` whose `self.movie() == swf_arc`. If we
            // evict the library here, a later access via
            // `library_for_movie_mut(swf_arc)` lazy-creates an
            // empty entry with no domain. Preload of that entry's
            // DoAbc2 / SymbolClass tags then panics at
            // `library.rs::avm2_domain` (`unwrap` on None) — see
            // the 16:36:31 panic chain in seer-run.log on the
            // second fight load.
            //
            // The library owns this domain, so the Option does not
            // become None when external Class references disappear.
            // This deliberately retains every AVM2 library until a
            // sound reachability signal exists. Bitmap and shape
            // residency sweeps reclaim their GPU resources without
            // invalidating cached classes or SymbolClass identity.
            if lib.avm2_domain.is_some() {
                continue;
            }
            out.push(swf_arc);
        }
        out
    }

    /// [seer-patch] Remove and drop a `MovieLibrary` entry. Returns
    /// `true` on hit. Dropping the entry releases every
    /// `Character<'gc>` it owned; the underlying gc-arena
    /// allocations become eligible for collection on the next
    /// sweep, at which point each `BitmapCharacter`'s
    /// residency slot drops, releasing its `wgpu::Texture`
    /// (subject to the renderer's queued-free flush — see
    /// `RenderBackend::empty_submit`).
    pub fn remove_movie(&mut self, movie: &Arc<SwfMovie>) -> bool {
        self.movie_libraries.remove(movie).is_some()
    }

    /// [seer-patch P1] Iterate every loaded `MovieLibrary` for the
    /// per-bitmap residency sweep. The returned references are
    /// borrowed from the underlying `PtrWeakKeyHashMap`; do not
    /// outlive `&self`.
    pub fn iter_libraries(&self) -> impl Iterator<Item = &MovieLibrary<'gc>> + '_ {
        self.movie_libraries.0.iter().map(|(_, lib)| lib)
    }
}

/// [seer-patch] Best-effort URL extraction for memory reports.
fn url_of_swf(swf: &Arc<SwfMovie>) -> Option<String> {
    let u = swf.url();
    if u.is_empty() {
        None
    } else {
        Some(u.to_string())
    }
}

#[derive(Collect, Default)]
#[collect(no_drop)]
struct FontMap<'gc>(FnvHashMap<FontQuery, Font<'gc>>);

impl<'gc> FontMap<'gc> {
    pub fn register(&mut self, font: Font<'gc>) {
        let descriptor = font.descriptor();
        self.0
            .entry(FontQuery::from_descriptor(font.font_type(), descriptor))
            .or_insert(font);
    }

    pub fn get(&self, font_query: &FontQuery) -> Option<&Font<'gc>> {
        self.0.get(font_query)
    }

    pub fn find(&self, font_query: &FontQuery) -> Option<Font<'gc>> {
        // The order here is specific, and tested in `tests/swfs/fonts/embed_matching/fallback_preferences`

        // Exact match
        if let Some(font) = self.get(font_query) {
            return Some(*font);
        }

        let is_italic = font_query.is_italic;
        let is_bold = font_query.is_bold;

        let mut fallback_query = font_query.clone();
        if is_italic ^ is_bold {
            // If one is set (but not both), then try upgrading to bold italic...
            fallback_query.is_bold = true;
            fallback_query.is_italic = true;
            if let Some(font) = self.get(&fallback_query) {
                return Some(*font);
            }

            // and then downgrading to regular
            fallback_query.is_bold = false;
            fallback_query.is_italic = false;
            if let Some(font) = self.get(&fallback_query) {
                return Some(*font);
            }

            // and then finally whichever one we don't have set
            fallback_query.is_bold = !is_bold;
            fallback_query.is_italic = !is_italic;
            if let Some(font) = self.get(&fallback_query) {
                return Some(*font);
            }
        } else {
            // We don't have an exact match and we were either looking for regular or bold-italic

            if is_italic && is_bold {
                // Do we have regular? (unless we already looked for it)
                fallback_query.is_bold = false;
                fallback_query.is_italic = false;
                if let Some(font) = self.get(&fallback_query) {
                    return Some(*font);
                }
            }

            // Do we have bold?
            fallback_query.is_bold = true;
            fallback_query.is_italic = false;
            if let Some(font) = self.get(&fallback_query) {
                return Some(*font);
            }

            // Do we have italic?
            fallback_query.is_bold = false;
            fallback_query.is_italic = true;
            if let Some(font) = self.get(&fallback_query) {
                return Some(*font);
            }

            if !is_bold && !is_italic {
                // Do we have bold italic? (unless we already looked for it)
                fallback_query.is_bold = true;
                fallback_query.is_italic = true;
                if let Some(font) = self.get(&fallback_query) {
                    return Some(*font);
                }
            }
        }

        None
    }

    pub fn all(&self) -> Vec<Font<'gc>> {
        self.0.values().copied().collect()
    }
}
