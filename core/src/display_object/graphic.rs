use crate::avm1::Object as Avm1Object;
use crate::avm2::{
    Activation as Avm2Activation, Avm2, ClassObject as Avm2ClassObject,
    StageObject as Avm2StageObject,
};
use crate::context::{RenderContext, UpdateContext};
use crate::display_object::{BoundsMode, DisplayObjectBase};
use crate::drawing::Drawing;
use crate::library::MovieLibrarySource;
use crate::prelude::*;
use crate::tag_utils::SwfMovie;
use crate::vminterface::Instantiator;
use core::fmt;
use gc_arena::barrier::unlock;
use gc_arena::lock::Lock;
use gc_arena::{Collect, Gc, Mutation};
use ruffle_common::utils::HasPrefixField;
use ruffle_render::backend::ShapeHandle;
use ruffle_render::commands::CommandHandler;
use std::cell::{Cell, OnceCell, RefCell, RefMut};
use std::sync::Arc;
use std::time::Duration;
use web_time::Instant;

#[derive(Clone, Collect, Copy)]
#[collect(no_drop)]
pub struct Graphic<'gc>(Gc<'gc, GraphicData<'gc>>);

impl fmt::Debug for Graphic<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Graphic")
            .field("ptr", &Gc::as_ptr(self.0))
            .finish()
    }
}

#[derive(Clone, Collect, HasPrefixField)]
#[collect(no_drop)]
#[repr(C, align(8))]
pub struct GraphicData<'gc> {
    base: DisplayObjectBase<'gc>,
    shared: Lock<Gc<'gc, GraphicShared>>,
    class: Lock<Option<Avm2ClassObject<'gc>>>,
    avm2_object: Lock<Option<Avm2StageObject<'gc>>>,
    /// This is lazily allocated on demand, to make `GraphicData` smaller in the common case.
    #[collect(require_static)]
    drawing: OnceCell<Box<RefCell<Drawing>>>,
}

impl<'gc> Graphic<'gc> {
    /// Construct a `Graphic` from it's associated `Shape` tag.
    pub fn from_swf_tag(
        context: &mut UpdateContext<'gc>,
        swf_shape: swf::Shape,
        movie: Arc<SwfMovie>,
    ) -> Self {
        let shared = GraphicShared::from_swf_shape(swf_shape, movie);

        Graphic(Gc::new(
            context.gc(),
            GraphicData {
                base: Default::default(),
                shared: Lock::new(Gc::new(context.gc(), shared)),
                class: Lock::new(None),
                avm2_object: Lock::new(None),
                drawing: OnceCell::new(),
            },
        ))
    }

    /// Construct an empty `Graphic`.
    pub fn empty(context: &mut UpdateContext<'gc>) -> Self {
        let shared = GraphicShared {
            id: 0,
            bounds: Default::default(),
            render_handle: RefCell::new(None),
            last_rendered: Cell::new(None),
            shape: swf::Shape {
                version: 32,
                id: 0,
                shape_bounds: Default::default(),
                edge_bounds: Default::default(),
                flags: swf::ShapeFlag::empty(),
                styles: swf::ShapeStyles {
                    fill_styles: Vec::new(),
                    line_styles: Vec::new(),
                },
                shape: Vec::new(),
            },
            movie: context.root_swf.clone(),
        };

        Graphic(Gc::new(
            context.gc(),
            GraphicData {
                base: Default::default(),
                shared: Lock::new(Gc::new(context.gc(), shared)),
                class: Lock::new(None),
                avm2_object: Lock::new(None),
                drawing: OnceCell::new(),
            },
        ))
    }

    pub fn drawing_mut(&self) -> RefMut<'_, Drawing> {
        self.0.drawing.get_or_init(Default::default).borrow_mut()
    }

    pub fn set_avm2_class(self, mc: &Mutation<'gc>, class: Avm2ClassObject<'gc>) {
        unlock!(Gc::write(mc, self.0), GraphicData, class).set(Some(class));
    }

    /// Drop the lazily-created GPU shape after it has not been rendered for
    /// `idle_threshold`. The source SWF shape remains available for rebuild.
    pub fn evict_shape_if_idle(self, now: Instant, idle_threshold: Duration) -> bool {
        self.0.shared.get().evict_shape_if_idle(now, idle_threshold)
    }

    fn set_shared(self, mc: &Mutation<'gc>, shared: Gc<'gc, GraphicShared>) {
        unlock!(Gc::write(mc, self.0), GraphicData, shared).set(shared);
    }
}

impl<'gc> TDisplayObject<'gc> for Graphic<'gc> {
    fn base(self) -> Gc<'gc, DisplayObjectBase<'gc>> {
        HasPrefixField::as_prefix_gc(self.0)
    }

    fn instantiate(self, gc_context: &Mutation<'gc>) -> DisplayObject<'gc> {
        Self(Gc::new(gc_context, self.0.as_ref().clone())).into()
    }

    fn id(self) -> CharacterId {
        self.0.shared.get().id
    }

    fn self_bounds(self, _mode: BoundsMode) -> Rectangle<Twips> {
        if let Some(drawing) = self.0.drawing.get() {
            drawing.borrow().self_bounds()
        } else {
            self.0.shared.get().bounds
        }
    }

    fn construct_frame(self, context: &mut UpdateContext<'gc>) {
        if self.movie().is_action_script_3() && self.object2().is_none() {
            let class_object = self
                .0
                .class
                .get()
                .unwrap_or_else(|| context.avm2.classes().shape);

            let mut activation = Avm2Activation::from_nothing(context);

            match Avm2StageObject::for_display_object_childless(
                &mut activation,
                self.into(),
                class_object,
            ) {
                Ok(object) => self.set_object2(activation.context, object),
                Err(err) => {
                    Avm2::uncaught_error(
                        &mut activation,
                        Some(self.into()),
                        err,
                        "Error running AVM2 construction for shape",
                    );
                }
            }

            self.on_construction_complete(context);
        }
    }

    fn replace_with(self, context: &mut UpdateContext<'gc>, id: CharacterId) {
        // Static assets like Graphics can replace themselves via a PlaceObject tag with PlaceObjectAction::Replace.
        // This does not create a new instance, but instead swaps out the underlying static data to point to the new art.
        if let Some(new_graphic) = context
            .library
            .library_for_movie_mut(self.movie())
            .get_graphic(id)
        {
            self.set_shared(context.gc(), new_graphic.0.shared.get());
        } else {
            tracing::warn!("PlaceObject: expected Graphic at character ID {}", id);
        }
        self.invalidate_cached_bitmap();
    }

    fn render_self(self, context: &mut RenderContext) {
        if !context.is_offscreen
            && !self
                .world_bounds(BoundsMode::Engine)
                .intersects(&context.stage.view_bounds())
        {
            // Off-screen; culled
            return;
        }

        if let Some(drawing) = self.0.drawing.get() {
            drawing.borrow().render(context);
        } else {
            let shared = self.0.shared.get();
            shared.last_rendered.set(Some(Instant::now()));
            let render_handle = if let Some(handle) = shared.render_handle.borrow().clone() {
                handle
            } else {
                let library = context
                    .library
                    .library_for_movie(shared.movie.clone())
                    .unwrap();
                let handle = context
                    .renderer
                    .register_shape((&shared.shape).into(), &MovieLibrarySource { library });
                *shared.render_handle.borrow_mut() = Some(handle.clone());
                handle
            };
            context
                .commands
                .render_shape(render_handle, context.transform_stack.transform())
        }
    }

    fn hit_test_shape(
        self,
        _context: &mut UpdateContext<'gc>,
        point: Point<Twips>,
        options: HitTestOptions,
    ) -> bool {
        // Transform point to local coordinates and test.
        if (!options.contains(HitTestOptions::SKIP_INVISIBLE) || self.visible())
            && self.world_bounds(BoundsMode::Engine).contains(point)
        {
            let Some(local_matrix) = self.global_to_local_matrix() else {
                return false;
            };
            let point = local_matrix * point;
            if let Some(drawing) = self.0.drawing.get() {
                if drawing.borrow().hit_test(point, &local_matrix) {
                    return true;
                }
            } else {
                let shape = &self.0.shared.get().shape;
                return ruffle_render::shape_utils::shape_hit_test(shape, point, &local_matrix);
            }
        }

        false
    }

    fn post_instantiation(
        self,
        context: &mut UpdateContext<'gc>,
        _init_object: Option<Avm1Object<'gc>>,
        _instantiated_by: Instantiator,
        _run_frame: bool,
    ) {
        if self.movie().is_action_script_3() {
            self.set_default_instance_name(context);
        }
    }

    fn movie(self) -> Arc<SwfMovie> {
        self.0.shared.get().movie.clone()
    }

    fn object1(self) -> Option<Avm1Object<'gc>> {
        None
    }

    fn object2(self) -> Option<Avm2StageObject<'gc>> {
        self.0.avm2_object.get()
    }

    fn set_object2(self, context: &mut UpdateContext<'gc>, to: Avm2StageObject<'gc>) {
        let mc = context.gc();
        unlock!(Gc::write(mc, self.0), GraphicData, avm2_object).set(Some(to));
    }

    fn as_drawing(&self) -> Option<RefMut<'_, Drawing>> {
        Some(self.drawing_mut())
    }
}

/// Data shared between all instances of a Graphic.
#[derive(Collect)]
#[collect(require_static)]
struct GraphicShared {
    id: CharacterId,
    shape: swf::Shape,
    render_handle: RefCell<Option<ShapeHandle>>,
    last_rendered: Cell<Option<Instant>>,
    bounds: Rectangle<Twips>,
    movie: Arc<SwfMovie>,
}

impl GraphicShared {
    fn from_swf_shape(shape: swf::Shape, movie: Arc<SwfMovie>) -> Self {
        Self {
            id: shape.id,
            bounds: shape.shape_bounds,
            render_handle: RefCell::new(None),
            last_rendered: Cell::new(None),
            shape,
            movie,
        }
    }

    fn evict_shape_if_idle(&self, now: Instant, idle_threshold: Duration) -> bool {
        let Some(last_rendered) = self.last_rendered.get() else {
            return false;
        };
        if now.duration_since(last_rendered) < idle_threshold {
            return false;
        }
        self.render_handle.borrow_mut().take().is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ruffle_render::backend::ShapeHandleImpl;

    #[derive(Debug)]
    struct TestShapeHandle;

    impl ShapeHandleImpl for TestShapeHandle {}

    fn test_shape() -> swf::Shape {
        swf::Shape {
            version: 32,
            id: 1,
            shape_bounds: Default::default(),
            edge_bounds: Default::default(),
            flags: swf::ShapeFlag::empty(),
            styles: swf::ShapeStyles {
                fill_styles: Vec::new(),
                line_styles: Vec::new(),
            },
            shape: Vec::new(),
        }
    }

    #[test]
    fn swf_shapes_are_lazy_and_idle_handles_are_evictable() {
        let shared =
            GraphicShared::from_swf_shape(test_shape(), Arc::new(SwfMovie::empty(32, None)));
        assert!(shared.render_handle.borrow().is_none());

        *shared.render_handle.borrow_mut() = Some(ShapeHandle(Arc::new(TestShapeHandle)));
        let now = Instant::now();
        shared.last_rendered.set(Some(now));
        assert!(!shared.evict_shape_if_idle(now, Duration::from_secs(30)));

        shared
            .last_rendered
            .set(Some(now - Duration::from_secs(31)));
        assert!(shared.evict_shape_if_idle(now, Duration::from_secs(30)));
        assert!(shared.render_handle.borrow().is_none());
    }
}
