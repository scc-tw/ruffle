//! `flash.media.Sound` builtin/prototype

use crate::avm2::Avm2;
use crate::avm2::Error;
use crate::avm2::activation::Activation;
use crate::avm2::error::{make_error_2037, make_error_2084};
use crate::avm2::globals::methods::flash_media_sound as sound_methods;
use crate::avm2::globals::slots::flash_net_url_request as url_request_slots;
use crate::avm2::object::{
    EventObject, QueuedPlay, SoundChannelObject, SoundLoadingState, TObject as _,
};
use crate::avm2::parameters::ParametersExt;
use crate::avm2::value::Value;
use crate::backend::navigator::Request;
use crate::character::Character;
use crate::display_object::SoundTransform;
use swf::{SoundEvent, SoundInfo};

pub use crate::avm2::object::sound_allocator;

/// Implements `flash.media.Sound`'s 'init' method. which is called from the constructor.
pub fn init<'gc>(
    activation: &mut Activation<'_, 'gc>,
    this: Value<'gc>,
    args: &[Value<'gc>],
) -> Result<Value<'gc>, Error<'gc>> {
    if let Some(sound_object) = this.as_object().and_then(|o| o.as_sound_object()) {
        let class_def = this.instance_class(activation);

        if let Some((movie, symbol)) = activation
            .context
            .library
            .avm2_class_registry()
            .class_symbol(class_def)
        {
            if let Some(Character::Sound(sound)) = activation
                .context
                .library
                .library_for_movie_mut(movie)
                .character_by_id(symbol)
            {
                sound_object.set_sound(activation.context, sound);
            } else {
                tracing::warn!(
                    "Attempted to construct subclass of Sound, {}, which is associated with non-Sound character {}",
                    class_def.name().local_name(),
                    symbol
                );
            }
        }
    }

    if args.try_get_object(0).is_some() {
        this.call_method(sound_methods::LOAD, args, activation)?;
    }

    Ok(Value::Undefined)
}

/// Implements `Sound.bytesTotal`
pub fn get_bytes_total<'gc>(
    activation: &mut Activation<'_, 'gc>,
    this: Value<'gc>,
    _args: &[Value<'gc>],
) -> Result<Value<'gc>, Error<'gc>> {
    let this = this.as_object().unwrap();

    if let Some(sound) = this.as_sound_object() {
        if let Some(sound_handle) = sound.sound_handle()
            && let Some(length) = activation.context.audio.get_sound_size(sound_handle)
        {
            return Ok((length).into());
        }
        return Ok(0.into());
    }

    Ok(Value::Undefined)
}

pub fn get_bytes_loaded<'gc>(
    activation: &mut Activation<'_, 'gc>,
    this: Value<'gc>,
    args: &[Value<'gc>],
) -> Result<Value<'gc>, Error<'gc>> {
    // Ruffle's audio backend does not surface progressive download progress;
    // once a Sound finishes loading it has bytesLoaded == bytesTotal, which
    // is what AS3 progress UIs end up observing in practice.
    get_bytes_total(activation, this, args)
}

/// Implements `Sound.isBuffering`
pub fn get_is_buffering<'gc>(
    _activation: &mut Activation<'_, 'gc>,
    _this: Value<'gc>,
    _args: &[Value<'gc>],
) -> Result<Value<'gc>, Error<'gc>> {
    // Network-loaded sounds are decoded synchronously once the bytes arrive,
    // so there is never a buffering state to report.
    Ok(false.into())
}

/// Implements `Sound.isURLInaccessible`
pub fn get_is_url_inaccessible<'gc>(
    _activation: &mut Activation<'_, 'gc>,
    _this: Value<'gc>,
    _args: &[Value<'gc>],
) -> Result<Value<'gc>, Error<'gc>> {
    // Ruffle does not enforce cross-domain policies on Sound URLs.
    Ok(false.into())
}

/// Implements `Sound.url`
pub fn get_url<'gc>(
    _activation: &mut Activation<'_, 'gc>,
    _this: Value<'gc>,
    _args: &[Value<'gc>],
) -> Result<Value<'gc>, Error<'gc>> {
    // We do not currently track the source URL on the Sound object.
    Ok(Value::Null)
}

/// Implements `Sound.length`
pub fn get_length<'gc>(
    activation: &mut Activation<'_, 'gc>,
    this: Value<'gc>,
    _args: &[Value<'gc>],
) -> Result<Value<'gc>, Error<'gc>> {
    let this = this.as_object().unwrap();

    if let Some(sound) = this.as_sound_object() {
        if let Some(sound_handle) = sound.sound_handle()
            && let Some(duration) = activation.context.audio.get_sound_duration(sound_handle)
        {
            return Ok((duration.as_millis()).into());
        }
        return Ok(0.into());
    }

    Ok(Value::Undefined)
}

/// Implements `Sound.play`
pub fn play<'gc>(
    activation: &mut Activation<'_, 'gc>,
    this: Value<'gc>,
    args: &[Value<'gc>],
) -> Result<Value<'gc>, Error<'gc>> {
    let this = this.as_object().unwrap();

    if let Some(sound_object) = this.as_sound_object() {
        let position = args.get_f64(0);
        let num_loops = args.get_i32(1);
        let sound_transform = args.try_get_object(2);

        let in_sample = if position > 0.0 {
            Some((position / 1000.0 * 44100.0) as u32)
        } else {
            None
        };

        let sound_info = SoundInfo {
            event: SoundEvent::Start,
            in_sample,
            out_sample: None,
            num_loops: num_loops.max(1) as u16,
            envelope: None,
        };

        let sound_transform = if let Some(sound_transform) = sound_transform {
            Some(SoundTransform::from_avm2_object(sound_transform))
        } else {
            None
        };

        let sound_channel = SoundChannelObject::empty(activation);

        let queued_play = QueuedPlay {
            position,
            sound_info,
            sound_transform,
            sound_channel,
        };
        if sound_object.play(queued_play, activation) {
            return Ok(sound_channel.into());
        }
        // If we start playing a loaded sound with an invalid position,
        // this method returns `null`
        return Ok(Value::Null);
    }

    Ok(Value::Null)
}

/// `Sound.extract` — fills a ByteArray with PCM samples from the loaded
/// sound. Ruffle does not yet expose decoded PCM through the audio backend,
/// so we satisfy callers by writing the requested number of zero-bytes.
pub fn extract<'gc>(
    activation: &mut Activation<'_, 'gc>,
    _this: Value<'gc>,
    args: &[Value<'gc>],
) -> Result<Value<'gc>, Error<'gc>> {
    let bytearray = args.try_get_object(0);
    let length = args.get_f64(1);

    if let Some(bytearray) = bytearray
        && let Some(mut bytearray) = bytearray.as_bytearray_mut()
    {
        bytearray
            .write_bytes(vec![0u8; length.ceil() as usize].as_slice())
            .map_err(|e| e.to_avm(activation))?;
    }

    Ok(Value::Undefined)
}

/// `Sound.close` — Adobe spec: closes the network stream backing this
/// Sound. We do not maintain such a stream after loading completes, so
/// this is effectively a no-op.
pub fn close<'gc>(
    _activation: &mut Activation<'_, 'gc>,
    _this: Value<'gc>,
    _args: &[Value<'gc>],
) -> Result<Value<'gc>, Error<'gc>> {
    Ok(Value::Undefined)
}

/// `Sound.load`
pub fn load<'gc>(
    activation: &mut Activation<'_, 'gc>,
    this: Value<'gc>,
    args: &[Value<'gc>],
) -> Result<Value<'gc>, Error<'gc>> {
    let this_object = this.as_object().unwrap();

    let this = this_object.as_sound_object().unwrap();
    if this.loading_state() != SoundLoadingState::New {
        return Err(make_error_2037(activation));
    }

    let url_request = match args.try_get_object(0) {
        Some(request) => request,
        // FP ignores calls of `load(null)`
        None => return Ok(Value::Undefined),
    };

    let url = url_request
        .get_slot(url_request_slots::_URL)
        .coerce_to_string(activation)?;

    // SoundLoaderContext (buffer time, checkPolicyFile) is currently
    // ignored — Ruffle decodes the entire stream up-front and does not
    // enforce cross-domain policies.
    let _ = args.try_get_object(1);

    let future = crate::loader::load_sound_avm2(
        activation.context,
        this,
        // FIXME: Set options from the `URLRequest`.
        Request::get(url.to_string()),
    );
    activation.context.navigator.spawn_future(future);
    this.set_loading_state(SoundLoadingState::Loading);

    Ok(Value::Undefined)
}

/// `Sound.loadCompressedDataFromByteArray`
pub fn load_compressed_data_from_byte_array<'gc>(
    activation: &mut Activation<'_, 'gc>,
    this: Value<'gc>,
    args: &[Value<'gc>],
) -> Result<Value<'gc>, Error<'gc>> {
    let this_object = this.as_object().unwrap();

    let this = this_object.as_sound_object().unwrap();
    if this.loading_state() == SoundLoadingState::Loaded {
        return Ok(Value::Undefined);
    }

    let bytearray = args.get_object(activation, 0, "bytes")?;
    let bytes_length = args.get_u32(1);
    let bytearray = bytearray.as_bytearray().unwrap();

    let bytes = if let Ok(bytes) = bytearray.read_bytes(bytes_length as usize) {
        bytes
    } else {
        // This is the error Flash throws
        return Err(make_error_2084(activation));
    };

    // FIXME - determine the actual error thrown by Flash Player
    let handle = activation.context.audio.register_mp3(bytes).map_err(|e| {
        Error::rust_error(format!("Failed to register sound from bytearray: {e:?}").into())
    })?;

    let progress_evt =
        EventObject::progress_event(activation, "progress", bytes.len(), bytes.len());

    Avm2::dispatch_event(activation.context, progress_evt, this_object);

    this.read_and_call_id3_event(activation, bytes);
    this.set_sound(activation.context, handle);

    Ok(Value::Undefined)
}

/// `Sound.loadPCMFromByteArray`
pub fn load_pcm_from_byte_array<'gc>(
    activation: &mut Activation<'_, 'gc>,
    this: Value<'gc>,
    _args: &[Value<'gc>],
) -> Result<Value<'gc>, Error<'gc>> {
    let this_object = this.as_object().unwrap();

    let this = this_object.as_sound_object().unwrap();
    if this.loading_state() == SoundLoadingState::Loaded {
        return Ok(Value::Undefined);
    }

    // PCM-from-ByteArray ingestion is not yet wired through the audio
    // backend — flag the sound as loaded so subsequent load*() calls do
    // not loop, but no audio data is registered.
    this.set_loading_state(SoundLoadingState::Loaded);
    let _ = activation;

    Ok(Value::Undefined)
}

/// Implements `Sound.id3`
pub fn get_id3<'gc>(
    _activation: &mut Activation<'_, 'gc>,
    this: Value<'gc>,
    _args: &[Value<'gc>],
) -> Result<Value<'gc>, Error<'gc>> {
    let this = this.as_object().unwrap();

    if let Some(id3) = this.as_sound_object().unwrap().id3() {
        Ok(id3.into())
    } else {
        Ok(Value::Null)
    }
}
