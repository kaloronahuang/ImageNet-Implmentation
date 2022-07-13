# ImageNetDataset.py
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy
from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.types as types
import nvidia.dali.fn as fn

@pipeline_def
def ImageNetDALIPipeline(data_dir, crop, size, shard_id, num_shards, is_training):
    images, labels = fn.readers.file(file_root=data_dir,
                                     shard_id=shard_id,
                                     num_shards=num_shards,
                                     random_shuffle=is_training,
                                     pad_last_batch=True,
                                     name="Reader")
    dali_device = 'gpu'
    decoder_device = 'mixed'
    # preallocate the space for JPEGs;
    device_memory_padding = 211025920
    host_memory_padding = 140544512
    preallocate_width_hint = 5980
    preallocate_height_hint = 6430

    if is_training:
        images = fn.decoders.image_random_crop(images, 
            device=decoder_device, output_type=types.RGB,
            device_memory_padding=device_memory_padding,
            host_memory_padding=host_memory_padding,
            preallocate_width_hint=preallocate_width_hint,
            preallocate_height_hint=preallocate_height_hint,
            random_aspect_ratio=[0.8, 1.25],
            random_area=[0.1, 1.0],
            num_attempts=100
        )
        images = fn.resize(images,
                            device=dali_device,
                            resize_x=crop,
                            resize_y=crop,
                            interp_type=types.INTERP_TRIANGULAR)
        mirror = fn.random.coin_flip(probability=0.5)
    else:
        images = fn.decoders.image(images,
                                   device=decoder_device,
                                   output_type=types.RGB)
        images = fn.resize(images,
                           device=dali_device,
                           size=size,
                           mode="not_smaller",
                           interp_type=types.INTERP_TRIANGULAR)
        mirror = False
    images = fn.crop_mirror_normalize(images.gpu(),
                                      dtype=types.FLOAT,
                                      output_layout="CHW",
                                      crop=(crop, crop),
                                      mean=[0, 0, 0],
                                      std=[1, 1, 1],
                                      mirror=mirror)
    labels = labels.gpu()
    return images, labels