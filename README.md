# About zipformer-streaming-ovep-python-static
This Python pipeline is to show how to run [Zipformer Streaming](https://huggingface.co/stdo/PengChengStarling/tree/main) ASR on Intel CPU/GPU/NPU thru [ONNX Runtime](https://github.com/microsoft/onnxruntime) + [OpenVINO Execution Provider](https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html)

This implementation is forked from [Zipformer](https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/zipformer/onnx_pretrained-streaming.py) of [Icefall](https://github.com/k2-fsa/icefall/tree/master) project 

Audio samples ("```en.wav```", "```ja.wav```" and "```zh.wav```") are downloaded from [Hugging Face sherpa-onnx-streaming-zipformer-ar_en_id_ja_ru_th_vi_zh-2025-02-10](https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-ar_en_id_ja_ru_th_vi_zh-2025-02-10/tree/main/test_wavs)


## Key features
* On-line (streaming) mode
* Support Chinese, English, Russian, Vietnamese, Japanese, Thai, Indonesian, and Arabic
* Models are converted to static (mainly for NPU)

# Quick Steps
## Download and export models
Visit https://huggingface.co/stdo/PengChengStarling/tree/main, download the following files
```
decoder-epoch-75-avg-11-chunk-16-left-128.onnx
encoder-epoch-75-avg-11-chunk-16-left-128.onnx
joiner-epoch-75-avg-11-chunk-16-left-128.onnx
tokens.txt
```
Run the following commands install dependencies
```
pip install -r requirements.txt
```
## Run
Usage
```
usage: onnx_pretrained-streaming.py [-h] --encoder-model-filename ENCODER_MODEL_FILENAME
                                    --decoder-model-filename DECODER_MODEL_FILENAME
                                    --joiner-model-filename JOINER_MODEL_FILENAME [--tokens TOKENS] [--device DEVICE]
                                    [--use-mic] [--loopback]
                                    [sound_file]

positional arguments:
  sound_file            The input sound file to transcribe when not using --use-mic. Supported formats are those
                        supported by torchaudio.load(). For example, wav and flac are supported. The sample rate has
                        to be 16kHz. (default: )

options:
  -h, --help            show this help message and exit
  --encoder-model-filename ENCODER_MODEL_FILENAME
                        Path to the encoder onnx model. (default: None)
  --decoder-model-filename DECODER_MODEL_FILENAME
                        Path to the decoder onnx model. (default: None)
  --joiner-model-filename JOINER_MODEL_FILENAME
                        Path to the joiner onnx model. (default: None)
  --tokens TOKENS       Path to tokens.txt. (default: None)
  --device DEVICE       Execution device. Use 'CPU', 'GPU', 'NPU' for OpenVINO. If not specified, default
                        CPUExecutionProvider will be used. (default: None)
  --use-mic             If set, capture audio from microphone instead of reading a sound file. (default: False)
  --loopback            In mic mode, audio will be loop backed from mic to speaker. In file mode, the sound file will
                        be played synchronously while being processed. (default: False)
```

Run on CPU, input from audio file
```
python onnx_pretrained-streaming.py \
--encoder-model-filename encoder-epoch-75-avg-11-chunk-16-left-128.onnx \
--decoder-model-filename decoder-epoch-75-avg-11-chunk-16-left-128.onnx \
--joiner-model-filename joiner-epoch-75-avg-11-chunk-16-left-128.onnx \
--tokens tokens.txt \
--device CPU
zh.wav
```
Run on CPU, input from microphone
```
python onnx_pretrained-streaming.py \
--encoder-model-filename encoder-epoch-75-avg-11-chunk-16-left-128.onnx \
--decoder-model-filename decoder-epoch-75-avg-11-chunk-16-left-128.onnx \
--joiner-model-filename joiner-epoch-75-avg-11-chunk-16-left-128.onnx \
--tokens tokens.txt \
--device CPU
--use-mic
```
Device can be "CPU", "GPU", and "NPU". If --device is not specified, CPUExecutionProvider will be used by default

:warning:[NOTE] The 1st time running on NPU will take long time (about 3 minutes) on model compiling. [OpenVINO Model Caching](https://docs.openvino.ai/2025/openvino-workflow/running-inference/optimize-inference/optimizing-latency/model-caching-overview.html) has been enabled for NPU to ease the issue. This feature will cache compiled models. Although the 1st run still takes long, but later runs can be faster as model compilation has been skipped.
## Tested devices
The pipeline has been verified working on a ```Intel(R) Core(TM) Ultra 7 268V (Lunar Lake)``` system, with
* ```iGPU: Intel(R) Arc(TM) 140V GPU, driver 32.0.101.8247 (10/22/2025)```
* ```NPU: Intel(R) AI Boost, driver 32.0.100.4404 (11/7/2025)```
### Result
| Sample             | CPU | GPU | NPU |
|--------------------|-----|-----|-----|
| English (en.mp3)   | OK  | OK  | Not yet  |
| Japanese (ja.mp3)  | OK  | OK  | Not yet  |
| Chinese (zh.mp3)   | OK  | OK  | Not yet  |

### Sample log
```
(python313_general) C:\GitHub\zipformer-streaming>python onnx_pretrained-streaming.py --encoder-model-filename encoder-epoch-75-avg-11-chunk-16-left-128.onnx --decoder-model-filename decoder-epoch-75-avg-11-chunk-16-left-128.onnx --joiner-model-filename joiner-epoch-75-avg-11-chunk-16-left-128.onnx --tokens tokens.txt --loopback zh.wav
2025-12-16 14:46:16,079 INFO [onnx_pretrained-streaming.py:524] {'encoder_model_filename': 'encoder-epoch-75-avg-11-chunk-16-left-128.onnx', 'decoder_model_filename': 'decoder-epoch-75-avg-11-chunk-16-left-128.onnx', 'joiner_model_filename': 'joiner-epoch-75-avg-11-chunk-16-left-128.onnx', 'tokens': 'tokens.txt', 'device': None, 'use_mic': False, 'loopback': True, 'sound_file': 'zh.wav'}
Device: Using Default CPUExecutionProvider.
2025-12-16 14:46:18,945 INFO [onnx_pretrained-streaming.py:198] encoder_meta={'encoder_dims': '192,384,768,1024,768,384', 'version': '1', 'model_type': 'zipformer2', 'model_author': 'k2-fsa', 'comment': 'streaming zipformer2', 'decode_chunk_len': '32', 'num_encoder_layers': '2,2,4,5,4,2', 'T': '45', 'cnn_module_kernels': '31,31,15,15,15,31', 'left_context_len': '128,64,32,16,32,64', 'query_head_dims': '32,32,32,32,32,32', 'value_head_dims': '12,12,12,12,12,12', 'num_heads': '4,4,4,8,4,4'}
2025-12-16 14:46:18,946 INFO [onnx_pretrained-streaming.py:225] decode_chunk_len: 32
2025-12-16 14:46:18,946 INFO [onnx_pretrained-streaming.py:226] T: 45
2025-12-16 14:46:18,946 INFO [onnx_pretrained-streaming.py:227] num_encoder_layers: [2, 2, 4, 5, 4, 2]
2025-12-16 14:46:18,946 INFO [onnx_pretrained-streaming.py:228] encoder_dims: [192, 384, 768, 1024, 768, 384]
2025-12-16 14:46:18,947 INFO [onnx_pretrained-streaming.py:229] cnn_module_kernels: [31, 31, 15, 15, 15, 31]
2025-12-16 14:46:18,947 INFO [onnx_pretrained-streaming.py:230] left_context_len: [128, 64, 32, 16, 32, 64]
2025-12-16 14:46:18,947 INFO [onnx_pretrained-streaming.py:231] query_head_dims: [32, 32, 32, 32, 32, 32]
2025-12-16 14:46:18,947 INFO [onnx_pretrained-streaming.py:232] value_head_dims: [12, 12, 12, 12, 12, 12]
2025-12-16 14:46:18,947 INFO [onnx_pretrained-streaming.py:233] num_heads: [4, 4, 4, 8, 4, 4]
2025-12-16 14:46:18,991 INFO [onnx_pretrained-streaming.py:291] context_size: 2
2025-12-16 14:46:18,992 INFO [onnx_pretrained-streaming.py:292] vocab_size: 16000
2025-12-16 14:46:19,054 INFO [onnx_pretrained-streaming.py:305] joiner_dim: 512
2025-12-16 14:46:19,054 INFO [onnx_pretrained-streaming.py:538] Constructing Fbank computer
[output device]
{'name': 'Speakers (Realtek(R) Audio)', 'index': 4, 'hostapi': 0, 'max_input_channels': 0, 'max_output_channels': 2, 'default_low_input_latency': 0.09, 'default_low_output_latency': 0.09, 'default_high_input_latency': 0.18, 'default_high_output_latency': 0.18, 'default_samplerate': 44100.0}
2025-12-16 14:46:19,779 INFO [onnx_pretrained-streaming.py:665] Reading sound file: zh.wav
C:\Python\python313_general\Lib\site-packages\torchaudio\_backend\utils.py:213: UserWarning: In 2.9, this function's implementation will be changed to use torchaudio.load_with_torchcodec` under the hood. Some parameters like ``normalize``, ``format``, ``buffer_size``, and ``backend`` will be ignored. We recommend that you port your code to rely directly on TorchCodec's decoder instead: https://docs.pytorch.org/torchcodec/stable/generated/torchcodec.decoders.AudioDecoder.html#torchcodec.decoders.AudioDecoder.
  warnings.warn(
2025-12-16 14:46:20,917 INFO [onnx_pretrained-streaming.py:722] Partial: 对
2025-12-16 14:46:21,909 INFO [onnx_pretrained-streaming.py:722] Partial: 对我做
2025-12-16 14:46:22,008 INFO [onnx_pretrained-streaming.py:722] Partial: 对我做了介绍
2025-12-16 14:46:23,001 INFO [onnx_pretrained-streaming.py:722] Partial: 对我做了介绍那么
2025-12-16 14:46:23,097 INFO [onnx_pretrained-streaming.py:722] Partial: 对我做了介绍那么我想
2025-12-16 14:46:23,908 INFO [onnx_pretrained-streaming.py:722] Partial: 对我做了介绍那么我想说的
2025-12-16 14:46:24,027 INFO [onnx_pretrained-streaming.py:722] Partial: 对我做了介绍那么我想说的是
2025-12-16 14:46:24,183 INFO [onnx_pretrained-streaming.py:722] Partial: 对我做了介绍那么我想说的是大
2025-12-16 14:46:24,319 INFO [onnx_pretrained-streaming.py:722] Partial: 对我做了介绍那么我想说的是大家如果
2025-12-16 14:46:24,935 INFO [onnx_pretrained-streaming.py:722] Partial: 对我做了介绍那么我想说的是大家如果对我的
2025-12-16 14:46:25,047 INFO [onnx_pretrained-streaming.py:722] Partial: 对我做了介绍那么我想说的是大家如果对我的研究
2025-12-16 14:46:25,159 INFO [onnx_pretrained-streaming.py:722] Partial: 对我做了介绍那么我想说的是大家如果对我的研究感兴趣
2025-12-16 14:46:26,004 INFO [onnx_pretrained-streaming.py:722] Partial: 对我做了介绍那么我想说的是大家如果对我的研究感兴趣呢
2025-12-16 14:46:26,127 INFO [onnx_pretrained-streaming.py:730] zh.wav
2025-12-16 14:46:26,127 INFO [onnx_pretrained-streaming.py:731] 对我做了介绍那么我想说的是大家如果对我的研究感兴趣呢
2025-12-16 14:46:26,128 INFO [onnx_pretrained-streaming.py:733] Decoding Done
```
[Full log](https://github.com/luke-lin-vmc/sense-voice-ovep-python-static/blob/main/log_full.txt) (from scratch) is provided for reference

## Known issues
If the following warning appears when running the pipeline thru OVEP for the 1st time
```
C:\Users\...\site-packages\onnxruntime\capi\onnxruntime_inference_collection.py:123:
User Warning: Specified provider 'OpenVINOExecutionProvider' is not in available provider names.
Available providers: 'AzureExecutionProvider, CPUExecutionProvider'
```
This would be caused by that both ```onnxruntime``` and ```onnxruntime-openvino``` are installed. Solution is to remove both of them then re-install ```onnxruntime-openvino```
```
pip uninstall -y onnxruntime onnxruntime-openvino
pip install onnxruntime-openvino~=1.23.0
```
Or simply to re-install ```onnxruntime-openvino``` if you would like to keep ```onnxruntime```
```
pip uninstall -y onnxruntime-openvino
pip install onnxruntime-openvino~=1.23.0
```
